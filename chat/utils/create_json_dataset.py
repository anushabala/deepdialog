from argparse import ArgumentParser
from transcript_utils import load_scenarios, parse_transcript, is_transcript_valid
import os, sys
import string
import random
import json
import math
from collections import defaultdict
from chat.modeling.lexicon import Lexicon
from chat.modeling.dialogue_tracker import DialogueTracker
from chat.modeling.sample_utils import sample_candidates, normalize_weights
from chat.nn import logstats

def new_lex_mapping():
    return defaultdict(lambda : defaultdict(float))
def update_lex_mapping(target, source):
    for k1, m in source.items():
        for k2, v in m.items():
            target[k1][k2] += v

def create_example(lexicon, scenario, agent, args, stats):
    # Return example (JSON) and lexical mapping
    ex = {
        'scenario_id': scenario['uuid'],
        'agent': agent,
    }
    tracker = DialogueTracker(lexicon, scenario, agent, args, stats)
    tracker.executor.kb.dump()
    for (who, utterance) in transcript['dialogue']:
        tracker.parse_add(who, utterance)
    stats['num_states'].append(len(tracker.states))
    states = tracker.get_states()

    #print '%s states: max_prob=%s' % (len(states), states[0].prob if len(states) > 0 else 'n/a')
    print '%s states' % (len(states),)
    for state in states:
        state.dump()
    if len(states) == 0:  # Failed to generate this example
        return None, None

    weights = [state.weight() for state in states]
    probs = normalize_weights(weights)
    def messages_to_json(state):
        return [message.to_json() for message in state.messages]
    ex['states'] = [{'prob': prob, 'messages': messages_to_json(state)} for state, prob in zip(states, probs)]

    return ex, sum(weights)

def summarize_stats(stats):
    return '%s / %s / %s (%d)' % (min(stats), 1.0 * sum(stats) / len(stats), max(stats), len(stats))

def verify_example(lexicon, scenarios, args, ex):
    # Verify that that formulas we output can actually recover the raw utterance (or something close).
    # Also shows an example of how the DialogueTracker should be used in practice.
    scenario = scenarios[ex['scenario_id']]
    agent = ex['agent']
    tracker = DialogueTracker(lexicon, scenario, agent, args, {})
    #tracker.executor.dump_kb()
    for state in ex['states']:
        #print '### possible state'
        for message in state['messages']:
            who = message['who']
            formula_token_candidates = message['formula_token_candidates']
            formula_tokens = map(sample_candidates, formula_token_candidates)
            # If not agent, provide entity tokens (otherwise, no way to guess people not in our KB).
            entity_tokens = None if who == agent else message['entity_tokens']
            utterance = tracker.generate_add(who, formula_tokens, entity_tokens)
            orig_utterance = ' '.join(message['raw_tokens'])
            #print '---  formula: %s' % ' '.join(formula_tokens)
            #print '   generated: %s' % utterance
            #print '    original: %s%s' % (orig_utterance, ' (DIFF)' if utterance != orig_utterance else '')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--scenarios', type=str, help='File containing JSON scenarios', required=True)
    parser.add_argument('--transcripts', type=str, help='Directory containing chat transcripts', nargs='+', required=True)
    parser.add_argument('--out-prefix', type=str, help='Write output transcripts to this', required=True)
    parser.add_argument('--beam-size', type=int, help='Maximum number of candidate states to generate per agent/scenario', default=5)
    parser.add_argument('--random', type=int, help='Random seed', default=1)
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to process')
    parser.add_argument('--input-offset', type=int, help='Start here', default=0)
    parser.add_argument('--formulas-mode', type=str, help='Which formulas to include', default='full')
    parser.add_argument('--train-frac', type=float, help='Fraction of examples to use for training', default=0.9)
    parser.add_argument('--agents', type=int, help='Fraction of examples to use for training', nargs='+', default=[0, 1])

    args = parser.parse_args()
    logstats.init(args.out_prefix + 'stats.json')
    random.seed(args.random)
    scenarios = load_scenarios(args.scenarios)
    lexicon = Lexicon(scenarios)
    lexicon.test()

    train_examples = []
    dev_examples = []
    num_empty_examples = 0
    paths = []
    for transcript_dir in args.transcripts:
        for name in os.listdir(transcript_dir):
            f = os.path.join(transcript_dir, name)
            paths.append(f)

    stats = defaultdict(list)
    total_log_prob = 0
    for i, f in enumerate(sorted(paths)[args.input_offset:]):
        print '### Reading %d/%d: %s' % (i, len(paths), f)
        transcript = parse_transcript(scenarios, f, include_bots=False)
        if transcript is None:
            continue
        valid, reason = is_transcript_valid(transcript)
        if not valid:
            #print 'Invalid transcript because of %s: %s' % (reason, f)
            continue

        r = random.random()
        for agent in args.agents:
            scenario_id = transcript['scenario']
            ex, weight = create_example(lexicon, scenarios[scenario_id], agent, args, stats)
            if ex:
                #verify_example(lexicon, scenarios, args, ex)
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
    for key, values in stats.items():
        logstats.add(key, 1.0 * sum(values) / len(values))
        print key, '=', summarize_stats(values)

    # Write examples
    def output(name, examples):
        path = args.out_prefix + name
        print 'Outputting to %s: %d entries' % (path, len(examples))
        with open(path, 'w') as out:
            out.write(json.dumps(examples))
    output('train.json', train_examples)
    output('dev.json', dev_examples)
    #output('entity_phrase.json', dict((':'.join(k), v) for k, v in lex_mapping.items()))
