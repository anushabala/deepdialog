import argparse
import json
from transcript_utils import load_scenarios
from chat.modeling.lexicon import Lexicon
from chat.modeling import dialogue_tracker
from chat.modeling.recurrent_box import BigramBox, learn_bigram_model
from chat.lib import sample_utils
from chat.lib.cpt import ConditionalProbabilityTable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training data (JSON)', required=True)
    dialogue_tracker.add_arguments(parser)
    args = parser.parse_args()

    model = learn_bigram_model(json.load(open(args.train)))
    scenarios = load_scenarios(args.scenarios)
    lexicon = Lexicon(scenarios)

    for scenario in scenarios.values():
        agent = 0
        box = BigramBox(model)
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
