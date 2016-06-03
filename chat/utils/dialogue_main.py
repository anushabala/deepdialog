import argparse
import random
import json
from chat.modeling.lexicon import Lexicon, load_scenarios
from chat.modeling import dialogue_tracker
from chat.modeling import recurrent_box
from chat.lib import sample_utils, logstats
from chat.lib.cpt import ConditionalProbabilityTable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training data (JSON)', required=True)
    parser.add_argument('--dev', help='Development data (JSON)')
    parser.add_argument('--n', help='n-gram', type=int, default=2)
    parser.add_argument('--stats-file', help='File to output stats')
    parser.add_argument('--random', help='Random seed', type=int, default=1)
    dialogue_tracker.add_arguments(parser)
    args = parser.parse_args()
    random.seed(args.random)
    logstats.init(args.stats_file)
    logstats.add_args('options', args)

    scenarios = load_scenarios(args.scenarios)
    lexicon = Lexicon(scenarios)
    train_examples = json.load(open(args.train))
    dev_examples = json.load(open(args.dev))
    model = recurrent_box.learn_ngram_model(args.n, train_examples)

    def evaluate(mode, examples):
        print 'evaluate(%s): %d examples' % (mode, len(examples))
        summary_map = {}
        for ex in examples:
            scenario = scenarios[ex['scenario_id']]
            ex_summary_map = dialogue_tracker.evaluate_example(scenario, lexicon, args, ex, lambda : recurrent_box.NgramBox(model))
            logstats.update_summary_map(summary_map, ex_summary_map)
            logstats.add(mode, summary_map)
        print 'evaluate(%s): %s' % (mode, logstats.summary_map_to_str(summary_map))

    evaluate('train', train_examples)
    evaluate('dev', dev_examples)

    def interact():
        for scenario in scenarios.values():
            agent = 0
            box = recurrent_box.NgramBox(model)
            tracker = dialogue_tracker.DialogueTracker(lexicon, scenario, agent, args, box, None)
            def say(partner_utterance):
                partner_tokens = dialogue_tracker.utterance_to_tokens(partner_utterance)
                print 'who=%s: %s' % (1 - agent, partner_tokens)
                tracker.parse_add(1 - agent, partner_tokens, end_turn=True)
                while True:
                    agent_tokens, end_turn = tracker.generate_add(agent)
                    print 'who=%s: %s, end_turn=%s' % (agent, map(str, agent_tokens), end_turn)
                    if end_turn:
                        break
            tracker.executor.kb.dump()
            say('hello')
            break
    #interact()
