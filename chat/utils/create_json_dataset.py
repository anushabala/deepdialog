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

def create_example(lexicon, scenario, agent, args):
    #print '######'
    ex = {
        'scenario_id': scenario['uuid'],
        'agent': agent,
    }
    tracker = DialogueTracker(lexicon, scenario, agent, args)
    for (who, utterance) in transcript['dialogue']:
        tracker.parse_add(who, utterance)
    states = tracker.get_states()
    #print '%s states: max_prob=%s' % (len(states), states[0].prob if len(states) > 0 else 'n/a')
    print '%s states' % (len(states),)
    for state in states:
        state.dump()
    if len(states) == 0:  # Failed to generate this example
        return None
    ex['seqs'] = [state.to_json() for state in states]
    return ex

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--scenarios', type=str, help='File containing JSON scenarios', required=True)
    parser.add_argument('--transcripts', type=str, help='Directory containing chat transcripts', nargs='+', required=True)
    parser.add_argument('--out-prefix', type=str, help='Write output transcripts to this', required=True)
    parser.add_argument('--beam-size', type=int, help='Maximum number of candidate states to generate per agent/scenario', default=5)
    parser.add_argument('--random', type=int, help='Random seed', default=1)

    args = parser.parse_args()
    random.seed(args.random)
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

    total_log_prob = 0
    for i, f in enumerate(sorted(paths)):
        print '### Reading %d/%d: %s' % (i, len(paths), f)
        transcript = parse_transcript(f, include_bots=False)
        if transcript is None:
            continue
        valid, reason = is_transcript_valid(transcript)
        if not valid:
            #print 'Invalid transcript because of %s: %s' % (reason, f)
            continue

        r = random.random()
        for agent in [0, 1]:
            scenario_id = transcript['scenario']
            ex = create_example(lexicon, scenarios[scenario_id], agent, args)
            if ex:
                total_log_prob += math.log(ex['seqs'][0]['prob'])
                examples = train_examples if r < 0.9 else dev_examples
                examples.append(ex)
            else:
                num_empty_examples += 1

    print 'Created %d train examples, %d dev examples, threw away %d empty examples, total_log_prob = %s' % \
        (len(train_examples), len(dev_examples), num_empty_examples, total_log_prob)
    def output(name, examples):
        path = args.out_prefix + '.' + name
        print 'Outputting to %s: %d examples' % (path, len(examples))
        with open(path, 'w') as out:
            out.write(json.dumps(examples))
    output('train.json', train_examples)
    output('dev.json', dev_examples)
