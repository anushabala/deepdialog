import os, sys
import string
import random
import json
from collections import defaultdict
from chat.modeling.executor import Executor, is_entity

class DialogueState(object):
    '''
    Corresponds to one interpretation of the history.
    '''
    def __init__(self):
        self.seq = []  # List of (who, raw_tokens, entity_tokens, formula_tokens) tuples
        self.mentions = [[], []]  # List of entities mentioned for agent and opp
        self.prob = 1.0

    def extend(self, who, raw_tokens, entity_tokens, formula_tokens, prob):
        new = DialogueState()
        new.seq = self.seq + [(who, raw_tokens, entity_tokens, formula_tokens)]
        new.mentions = [self.mentions[0], self.mentions[1]]
        new.prob = self.prob * prob
        for token in entity_tokens:
            if is_entity(token):
                new.mentions[who] = new.mentions[who] + [token]
        return new

    def token_to_str(self, token, top_level=True):
        # ['And', ['FriendOf', 'A'], ['HasCompany', ['Get', ['MentionOf', 'B'], '1']]] => !And(FriendsOf(A),HasCompany(Get(MentionOf(B),1)))
        if not is_entity(token):
            return token 
        return ('!' if top_level else '') + token[0] + '(' + ','.join(self.token_to_str(x, False) for x in token[1:]) + ')'

    def dump(self):
        print '>>> prob=%s' % self.prob
        for who, raw_tokens, entity_tokens, formula_tokens in self.seq:
            print '  who=%s: %s | %s' % (who, ' '.join(raw_tokens), ' '.join(map(self.token_to_str, formula_tokens)))

    def to_json(self):
        return {'prob': self.prob, 'seq': [
            {
                'who': who,
                'raw_tokens': raw_tokens,
                'entity_tokens': entity_tokens,
                'formula_tokens': map(self.token_to_str, tokens),
            } \
            for (who, raw_tokens, entity_tokens, tokens) in self.seq
        ]}

class DialogueTracker(object):
    '''
    Stores potentially many interpretations.
    '''
    def __init__(self, lexicon, scenario, agent, args):
        self.lexicon = lexicon
        self.scenario = scenario
        self.agent = agent
        self.args = args
        self.executor = Executor(scenario, agent)
        self.states = [DialogueState()]  # Possible dialogue states that we could be in.

    def get_states(self):
        return self.states

    def parse_add(self, who, utterance):
        utterance = utterance.encode('utf-8').lower()
        utterance = utterance.translate(string.maketrans('', ''), string.punctuation)  # Remove punctuation
        tokens = utterance.split(' ')
        print '##### parse_add who=%s %s' % (who, tokens)

        # Convert (some) tokens into entities
        candidates = self.convert_tokens_to_entities(tokens)
        #for c in candidates:
            #print self.agent, who, utterance, '=>', c

        # Convert entities into logical forms
        new_states = []
        for si, state in enumerate(self.states):
            for c in candidates:
                #print '---- state %s/%s: %s' % (si, len(self.states), c)
                for cc, prob in self.convert_entities_to_formulas(state, who, c):
                    new_states.append(state.extend(who, tokens, c, cc, prob))
        self.states = new_states

        # Sort and prune!
        self.states = sorted(self.states, key=lambda s: s.prob, reverse=True)
        self.states = self.states[:self.args.beam_size]

    def generate_add(self, who, formula_tokens):
        # Called when the model generates formula_tokens and we want to actually want to render
        # Try to execute in the following states
        for state in self.states:
            # TODO
            pass
            

    def convert_tokens_to_entities(self, tokens):
        # Example: ['i', 'work', 'at', 'apple'] => [['i', 'work', 'at', ('apple', 'company')], ...]
        i = 0
        candidates = [[]]
        while i < len(tokens):
            # Find longest phrase (if any) that matches an entity
            for l in range(5, 0, -1):
                phrase = ' '.join(tokens[i:i+l])
                results = self.lexicon.lookup(phrase)
                if len(results) > 0:
                    i += l
                    break
            if not results:
                results = [tokens[i]]
                i += 1
            new_candidates = []
            for c in candidates:
                for r in results:
                    new_candidates.append(c + [r])
            candidates = new_candidates
        return candidates

    def convert_entities_to_formulas(self, state, who, tokens):
        # Example: ['i', 'work', 'at', ('apple', 'company')] => [['i', 'work', 'at', ('Get', ('CompanyOf', 'A'), '1')], ...]
        # Each candidate formula is associated with a probability of generating the actual entity given the formula,
        # assuming a uniform distribution over entities.
        candidates = [([], 1.0)]
        for i, token in enumerate(tokens):
            # Go through all the entity tokens and abstract them into logical forms.
            results = []
            if not is_entity(token):
                results.append((token, 1)) 
            else:
                #print '- %s' % (token,)
                # Find all the ways that we could execute to this entity
                for formula in self.executor.generate_formulas(state, who, tokens, i):
                    pred_token = self.executor.execute(state, who, tokens, i, formula)
                    # If the executor returns a list containing the answer, then return it, weighting it properly.
                    if pred_token and token in pred_token:
                        prob = 1.0 / len(pred_token)
                        #print '  %s =[%s * %s]=> %s' % (formula, state.prob, prob, pred_token[0])
                        results.append((formula, prob))
            if len(results) == 0:
                print 'WARNING: no way to generate %s' % (token,)
                self.executor.dump_kb()
                state.dump()
            new_candidates = []
            for c, prob in candidates:
                for r, p in results:
                    new_candidates.append((c + [r], prob * p))
            candidates = new_candidates
            candidates = sorted(candidates, key=lambda (c, p): p, reverse=True)
            candidates = candidates[:self.args.beam_size]
        return candidates
