import os, sys
import string
import random
import json
from collections import defaultdict
from chat.modeling.executor import Executor, is_entity

def parse_formula(buf, i=0):  # Return thing parsed
    n = len(buf)
    pos = [i]
    def find(c):
        i = buf.find(c, pos[0])
        return i if i != -1 else n
    def recurse():
        i = min(find('('), find(')'), find(','))
        result = buf[pos[0]:i]
        pos[0] = i
        if i < n:
            if buf[pos[0]] == '(':
                result = [result]
                while buf[pos[0]] != ')':
                    pos[0] += 1
                    result.append(recurse())
                pos[0] += 1
        return result
    return recurse()

#print parse_formula('asdf(3452,23(23),3f(3(f)),f)')

class DialogueState(object):
    '''
    Corresponds to one interpretation of the history.
    '''
    def __init__(self):
        self.seq = []  # List of (who, raw_tokens, entity_tokens, formula_tokens) tuples
        self.mentions = [[], []]  # List of entities mentioned for agent and opp
        self.lex_pairs = []
        self.prob = 1.0

    def extend(self, who, raw_tokens, entity_tokens, formula_tokens, prob, lex_pairs):
        new = DialogueState()
        new.seq = self.seq + [(who, raw_tokens, entity_tokens, formula_tokens)]
        new.mentions = [self.mentions[0], self.mentions[1]]
        new.prob = self.prob * prob
        new.lex_pairs = self.lex_pairs + lex_pairs  # Collect lexical mappings
        for token in entity_tokens: # Update mentions of entities
            if is_entity(token):
                new.mentions[who] = new.mentions[who] + [token]
        return new

    def formula_token_to_str(self, token, top_level=True):
        # ['And', ['FriendOf', 'A'], ['HasCompany', ['Get', ['MentionOf', 'B'], '1']]] => !And(FriendsOf(A),HasCompany(Get(MentionOf(B),1)))
        if not is_entity(token):
            return token
        return ('!' if top_level else '') + token[0] + '(' + ','.join(self.formula_token_to_str(x, False) for x in token[1:]) + ')'

    def dump(self):
        print '>>> prob=%s' % self.prob
        for who, raw_tokens, entity_tokens, formula_tokens in self.seq:
            print '  who=%s: %s | %s' % (who, ' '.join(raw_tokens), ' '.join(map(self.formula_token_to_str, formula_tokens)))

    def to_json(self):
        return {'prob': self.prob, 'seq': [
            {
                'who': who,
                'raw_tokens': raw_tokens,
                'entity_tokens': entity_tokens,
                'formula_tokens': map(self.formula_token_to_str, formula_tokens),
            } \
            for (who, raw_tokens, entity_tokens, formula_tokens) in self.seq
        ]}

class DialogueTracker(object):
    '''
    Stores potentially many interpretations.
    '''
    def __init__(self, lexicon, scenario, agent, args, stats):
        self.lexicon = lexicon
        self.scenario = scenario
        self.agent = agent
        self.args = args
        self.executor = Executor(scenario, agent, args)
        self.states = [DialogueState()]  # Possible dialogue states that we could be in.
        self.stats = stats

    def get_states(self):
        return self.states

    def parse_add(self, who, utterance):
        utterance = utterance.encode('utf-8').lower()
        utterance = utterance.translate(string.maketrans('', ''), string.punctuation)  # Remove punctuation
        raw_tokens = utterance.split(' ')
        print '##### parse_add who=%s %s' % (who, raw_tokens)

        # Convert (some) tokens into entities
        candidates = self.convert_raw_to_entities(raw_tokens)
        #for c in candidates:
            #print self.agent, who, utterance, '=>', c

        # Convert entities into logical forms
        new_states = []
        for si, state in enumerate(self.states):
            for entity_tokens, lex_pairs in candidates:
                #print '---- state %s/%s: %s' % (si, len(self.states), entity_tokens)
                for formula_tokens, prob in self.convert_entities_to_formulas(state, who, entity_tokens):
                    new_states.append(state.extend(who, raw_tokens, entity_tokens, formula_tokens, prob, lex_pairs))
        self.states = new_states
        self.stats['num_states'].append(len(self.states))

        # Sort and prune!
        self.states = sorted(self.states, key=lambda s: s.prob, reverse=True)
        self.states = self.states[:self.args.beam_size]

    def generate_add(self, who, str_formula_tokens, init_entity_tokens=None):
        # Returns the raw utterance and updates the dialogue tracker.
        # Should be really passing in multiple generated_tokens.
        # Example: ['i', 'went', 'to', '!SchoolOf(A)']
        # Called when the model generates formula_tokens and we want to actually want to render to an utterance.
        is_formula = [s.startswith('!') for s in str_formula_tokens]
        formula_tokens = [parse_formula(s, 1) if s.startswith('!') else s for s in str_formula_tokens]

        new_states = []
        for state in self.states:
            if init_entity_tokens:
                entity_tokens = list(init_entity_tokens)
            else:
                entity_tokens = [formula_tokens[i] if not is_formula[i] else None for i in range(len(formula_tokens))]
            prob = 1.0

            # Need to take multiple passes since some formulas depend on existence of following entities (e.g., 'two at Facebook')
            while True:
                changed = False
                for i in range(len(entity_tokens)):
                    if entity_tokens[i] is not None:
                        continue
                    formula = formula_tokens[i]
                    # Execute: SchoolOf(A) => 'university of pennsylvania'
                    choices = self.executor.execute(state, who, entity_tokens, i, formula)
                    if choices is None:  # Error or just not ready to execute (e.g., if formula has NextMention)
                        continue
                    if len(choices) == 0:  # Probably we got derailed...fail carefully
                        continue
                    if any(c is None for c in choices):
                        continue
                    #print formula, '=>', choices
                    entity_tokens[i] = random.choice(choices)
                    prob *= 1.0 / len(choices)
                    changed = True
                if not changed:
                    break

            #print 'RRRR', formula_tokens
            if any(token is None for token in entity_tokens):
                print 'Failed execution of %s, pruning state' % str_formula_tokens
                continue

            # Now generate the raw token from the entity from the lexical mapping
            # Example: 'university of pennsylvania' => 'upenn'
            # For now, just write things out explicitly.  Later, incorporate lexical mapping
            raw_tokens = [token[0] if is_entity(token) else token for token in entity_tokens]
            new_states.append(state.extend(who, raw_tokens, entity_tokens, formula_tokens, prob, []))

        self.states = new_states

        if len(self.states) == 0:
            return '(fail)'

        # Return an utterance
        state = self.states[0]  # Choose a state (probably should be highest prob)
        raw_tokens = state.seq[-1][1]  # Last utterance, get tokens
        return ' '.join(raw_tokens)

    def convert_raw_to_entities(self, raw_tokens):
        # Example: ['i', 'work', 'at', 'apple'] => [(['i', 'work', 'at', ('apple', 'company')], [('apple', 'company'), 'apple'], ...]
        i = 0
        candidates = [([], [])]  # Each candidate is a (list of entity tokens, list of (entity, raw))
        while i < len(raw_tokens):
            # Find longest phrase (if any) that matches an entity
            for l in range(5, 0, -1):
                phrase = ' '.join(raw_tokens[i:i+l])
                results = self.lexicon.lookup(phrase)
                if len(results) > 0:
                    j = i + l
                    break
            if not results:
                results = [raw_tokens[i]]
                j = i + 1
            new_candidates = []
            for entity_tokens, lex_pairs in candidates:
                for r in results:
                    new_lex_pairs = lex_pairs + [(r, ' '.join(raw_tokens[i:j]))] if is_entity(r) else lex_pairs
                    new_candidates.append((entity_tokens + [r], new_lex_pairs))
            i = j
            candidates = new_candidates
        return candidates

    def convert_entities_to_formulas(self, state, who, entity_tokens):
        # Example: ['i', 'work', 'at', ('apple', 'company')] => [['i', 'work', 'at', ('Get', ('CompanyOf', 'A'), '1')], ...]
        # Each candidate formula is associated with a probability of generating the actual entity given the formula,
        # assuming a uniform distribution over entities.
        candidates = [([], 1.0)]
        for i, token in enumerate(entity_tokens):
            # Go through all the entity tokens and abstract them into logical forms.
            results = []
            if not is_entity(token):
                results.append((token, 1))
            else:
                #print '- %s' % (token,)
                # Find all the ways that we could execute to this entity
                formulas = list(self.executor.generate_formulas(state, who, entity_tokens, i))
                for formula in formulas:
                    pred_token = self.executor.execute(state, who, entity_tokens, i, formula)
                    # If the executor returns a list containing the answer, then return it, weighting it properly.
                    if pred_token and token in pred_token:
                        prob = 1.0 / len(pred_token)
                        #print '  %s =[%s * %s]=> %s' % (formula, state.prob, prob, pred_token[0])
                        results.append((formula, prob))
                self.stats['num_formulas_per_token'].append(len(formulas))
                self.stats['num_valid_formulas_per_token'].append(len(results))
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
