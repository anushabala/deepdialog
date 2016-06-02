from argparse import ArgumentParser
import os, sys
import string
import random
import json
import math
from collections import defaultdict
import transcript_utils
from chat.modeling.lexicon import Lexicon, load_scenarios
from chat.modeling import dialogue_tracker
from chat.lib import sample_utils, logstats

def create_example(lexicon, scenario, agent, args, summary_map):
    # Return example (JSON) and lexical mapping
    ex = {
        'scenario_id': scenario['uuid'],
        'agent': agent,
    }
    tracker = dialogue_tracker.DialogueTracker(lexicon, scenario, agent, args, None, summary_map)
    tracker.executor.kb.dump()
    dialogue = transcript['dialogue']
    for i, (who, utterance) in enumerate(dialogue):
        end_turn = i+1 == len(dialogue) or dialogue[i+1][0] != who
        tracker.parse_add(who, dialogue_tracker.utterance_to_tokens(utterance), end_turn)
    logstats.update_summary(summary_map['num_states'], len(tracker.states))
    states = tracker.get_states()

    #print '%s states: max_prob=%s' % (len(states), states[0].prob if len(states) > 0 else 'n/a')
    print '%s states' % (len(states),)
    for state in states:
        state.dump()
    if len(states) == 0:  # Failed to generate this example
        return None, None

    weights = [state.weight() for state in states]
    def messages_to_json(state):
        return [message.to_json() for message in state.messages]
    ex['states'] = [{'weight': weight, 'messages': messages_to_json(state)} for state, weight in zip(states, weights)]

    return ex, sum(weights)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--transcripts', type=str, help='Directory containing chat transcripts', nargs='+', required=True)
    parser.add_argument('--out-prefix', type=str, help='Write output transcripts to this', required=True)
    parser.add_argument('--random', type=int, help='Random seed', default=1)
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to process')
    parser.add_argument('--input-offset', type=int, help='Start here', default=0)
    parser.add_argument('--train-frac', type=float, help='Fraction of examples to use for training', default=0.9)
    parser.add_argument('--agents', type=int, help='Fraction of examples to use for training', nargs='+', default=[0, 1])
    dialogue_tracker.add_arguments(parser)

    args = parser.parse_args()
    logstats.init(args.out_prefix + 'stats.json')
    logstats.add_args('options', args)
    random.seed(args.random)

    # Create lexicon
    scenarios = load_scenarios(args.scenarios)
    lexicon = Lexicon(scenarios)
    #lexicon.test()

    train_examples = []
    dev_examples = []
    num_empty_examples = 0
    paths = []
    for transcript_dir in args.transcripts:
        for name in os.listdir(transcript_dir):
            f = os.path.join(transcript_dir, name)
            paths.append(f)

    summary_map = defaultdict(dict)
    total_log_prob = 0
    for i, f in enumerate(sorted(paths)[args.input_offset:]):
        print '### Reading %d/%d: %s' % (i, len(paths), f)
        transcript = transcript_utils.parse_transcript(scenarios, f, include_bots=False)
        if transcript is None:
            continue
        valid, reason = transcript_utils.is_transcript_valid(transcript)
        if not valid:
            #print 'Invalid transcript because of %s: %s' % (reason, f)
            continue

        r = random.random()
        for agent in args.agents:
            scenario_id = transcript['scenario']
            ex, weight = create_example(lexicon, scenarios[scenario_id], agent, args, summary_map)
            logstats.update(summary_map)
            if ex:
                total_log_prob += math.log(weight)
                is_train = r < args.train_frac
                examples = train_examples if is_train else dev_examples
                examples.append(ex)
            else:
                num_empty_examples += 1
        if args.max_examples is not None and i >= args.max_examples:
            break

    print 'Created %d train examples, %d dev examples, threw away %d empty examples, total_log_prob = %s' % \
        (len(train_examples), len(dev_examples), num_empty_examples, total_log_prob)
    logstats.add('num_train_examples', len(train_examples))
    logstats.add('num_dev_examples', len(dev_examples))
    logstats.add('num_empty_examples', num_empty_examples)
    logstats.add('total_log_prob', total_log_prob)
    logstats.dump_summary_map(summary_map)

    # Write examples
    def output(name, examples):
        path = args.out_prefix + name
        print 'Outputting to %s: %d entries' % (path, len(examples))
        with open(path, 'w') as out:
            out.write(json.dumps(examples))
    output('train.json', train_examples)
    output('dev.json', dev_examples)
    #output('entity_phrase.json', dict((':'.join(k), v) for k, v in lex_mapping.items()))
