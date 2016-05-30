import os, sys
import string
import random
import json
from collections import defaultdict
import re
from chat.modeling.executor import Executor, is_entity
from chat.modeling.sample_utils import normalize_weights, sorted_candidates

# ['And', ['FriendOf', 'A'], ['HasCompany', ['Get', ['MentionOf', 'B'], '1']]]
#  <=>
# '!And(FriendsOf(A),HasCompany(Get(MentionOf(B),1)))'

def parse_formula(buf):
    assert buf[0] == '!'
    n = len(buf)
    pos = [1]
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

def render_formula(formula):
    def recurse(formula):
        if not is_entity(formula):
            return formula
        return  formula[0] + '(' + ','.join(map(recurse, formula[1:])) + ')'
    return '!' + recurse(formula)

#print parse_formula('asdf(3452,23(23),3f(3(f)),f)')

class Message(object):
    '''
    Represents a single message.
    Corresponds to one interpretation of the raw tokens as entity tokens,
    but stores all possible formulas that lead to the resulting entities.
    Note:
    - who: person who sent the message
      e.g., 0 or 1
    - raw_tokens: raw words
      e.g., ['studied', 'comp sci']
    - entity_tokens: entities
      e.g., ['studied', ('computer science', 'major')]
    - formula_token_candidates: list of candidate formulas for each token
      e.g., [[('studied', 1)], [(('MajorOf', 'A'), 1), (('MajorOf', ('FriendOf', 'A')), 0.2)]]
    entity_tokens and formula_tokens are aligned.
    '''
    def __init__(self, who, raw_tokens, entity_tokens, formula_token_candidates):
        self.who = who
        self.raw_tokens = raw_tokens
        self.entity_tokens = entity_tokens
        self.formula_token_candidates = formula_token_candidates
        # Compute weight
        self.weight = 1
        for i in range(len(formula_token_candidates)):
            # Marginalize over candidates at each position
            psi = 0
            for token, weight in formula_token_candidates[i]:
                psi += weight
            self.weight *= psi 

    def to_json(self):
        def render(candidates):
            candidates = sorted_candidates(candidates)
            str_tokens = [render_formula(token) if is_entity(token) else token for token, weight in candidates]
            probs = normalize_weights([weight for token, weight in candidates])
            return zip(str_tokens, probs)
                 
        return {
            'who': self.who,
            'raw_tokens': self.raw_tokens,
            'entity_tokens': self.entity_tokens,
            'formula_token_candidates': map(render, self.formula_token_candidates),
        }

class DialogueState(object):
    '''
    Stores a list of messages.
    '''
    def __init__(self):
        self.messages = []  # List of messages
        self.mentions = [[], []]  # List of entities mentioned for agent (A) and partner (B)

    def extend(self, message):
        new = DialogueState()
        new.messages = self.messages + [message]
        # Update mentions of entities (only for efficiency)
        new.mentions = [self.mentions[0], self.mentions[1]]
        for token in message.entity_tokens:
            if is_entity(token):
                new.mentions[message.who] = new.mentions[message.who] + [token]
        return new

    def weight(self):
        w = 1
        for message in self.messages:
            w *= message.weight
        return w

    def dump(self):
        def render_candidates(candidates):
            return ','.join( \
                (render_formula(token) if is_entity(token) else token) + ('' if weight == 1 else ':%.3f' % weight) \
                for token, weight in sorted_candidates(candidates) \
            )
        print '>>> weight=%s' % self.weight()
        for message in self.messages:
            print '  who=%s: %s | %s' % (
                message.who,
                ' '.join(message.raw_tokens),
                ' '.join(map(render_candidates, message.formula_token_candidates))
            )

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

        # utterance = utterance.translate(string.maketrans('', ''), string.punctuation)  # Remove punctuation
        # raw_tokens = utterance.split(' ')

        raw_tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
        print '##### parse_add who=%s orig_utterance=%s %s' % (who, utterance, raw_tokens)

        # Convert (some) tokens into entities
        entity_candidates = self.convert_raw_to_entities(raw_tokens)
        #for entity_tokens in entity_candidates:
            #print '>>', entity_tokens

        new_states = []
        for si, state in enumerate(self.states):
            #print '---- state %s/%s: %s' % (si, len(self.states), entity_tokens)
            for entity_tokens in entity_candidates:
                formula_token_candidates = self.convert_entities_to_formulas(state, who, entity_tokens)
                if not formula_token_candidates:
                    continue
                message = Message(who, raw_tokens, entity_tokens, formula_token_candidates)
                new_states.append(state.extend(message))
        self.states = new_states

        # Sort and prune!
        self.states = sorted(self.states, key=lambda s: s.weight, reverse=True)
        self.states = self.states[:self.args.beam_size]

    def compute_state_probs(self):
        return normalize_weights([state.weight() for state in self.states])

    def generate_add(self, who, str_formula_tokens, init_entity_tokens=None):
        # Returns the raw utterance and updates the dialogue tracker.
        # Returns None if execution fails.
        # Example of str_formula_tokens: ['i', 'went', 'to', '!SchoolOf(A)']
        # Called when the RNN model generates formula_tokens and we want to actually want to render to an utterance.
        # Assume there is only one state.
        if len(self.states) == 0:
            return None
        if len(self.states) > 1:
            raise Exception('Can only handle the one state case')
        is_formula = [s.startswith('!') for s in str_formula_tokens]
        formula_tokens = [parse_formula(s) if s.startswith('!') else s for s in str_formula_tokens]
        formula_weights = [1.0] * len(formula_tokens)

        state = self.states[0]
        if init_entity_tokens:
            entity_tokens = list(init_entity_tokens)
        else:
            entity_tokens = [formula_tokens[i] if not is_formula[i] else None for i in range(len(formula_tokens))]

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
                formula_weights[i] = 1.0 / len(choices)
                changed = True
            if not changed:
                break

        if any(token is None for token in entity_tokens):
            print 'WARNING: failed execution of %s' % str_formula_tokens
            # Note: don't update state in this case.
            return None

        # Now generate the raw token from the entity from the lexical mapping
        # Example: 'university of pennsylvania' => 'upenn'
        # For now, just write things out explicitly.  Later, incorporate lexical mapping
        raw_tokens = [token[0] if is_entity(token) else token for token in entity_tokens]

        # Update the state
        formula_token_candidates = map(lambda x : [x], zip(formula_tokens, formula_weights))
        message = Message(who, raw_tokens, entity_tokens, formula_token_candidates)
        state = state.extend(message)
        self.states = [state]

        # Return the utterance
        return ' '.join(raw_tokens)

    def convert_raw_to_entities(self, raw_tokens):
        '''
        Return a list of Message's corresponding to raw_tokens.
        '''
        # Example: ['i', 'work', 'at', 'apple'] => [(['i', 'work', 'at', ('apple', 'company')], [('apple', 'company'), 'apple'], ...]
        i = 0
        candidates = [[]]  # Each candidate is a (list of entity tokens)
        while i < len(raw_tokens):
            # Find longest phrase (if any) that matches an entity
            for l in range(5, 0, -1):
                phrase = ' '.join(raw_tokens[i:i+l])
                results = self.lexicon.lookup(phrase)
                if len(results) > 0:
                    j = i + l
                    break
            if not results:  # No entity match, just treat as normal token
                results = [raw_tokens[i]]
                j = i + 1
            new_candidates = []
            for entity_tokens in candidates:
                for r in results:
                    new_candidates.append(entity_tokens + [r])
            i = j
            candidates = new_candidates

        return candidates

    def convert_entities_to_formulas(self, state, who, entity_tokens):
        '''
        Example:
            ['i', 'work', 'at', ('apple', 'company')] =>
            [['i', 'work', 'at', ('Get', ('CompanyOf', 'A'), '1')], ...]
        Each candidate formula is associated with a probability of generating the actual entity given the formula,
        assuming a uniform distribution over entities.

        Return a list of messages
        '''
        formula_token_candidates = []

        for i, token in enumerate(entity_tokens):
            candidates = []  # List of formula tokens that could be at position i
            formula_token_candidates.append(candidates)
            if not is_entity(token):
                candidates.append((token, 1))
            else:
                #print '- %s' % (token,)
                # Find all the ways that we could execute to this entity
                formulas = list(self.executor.generate_formulas(state, who, entity_tokens, i))
                for formula in formulas:
                    pred_token = self.executor.execute(state, who, entity_tokens, i, formula)
                    # If the executor returns a list containing the answer, then return it, weighting it properly.
                    if pred_token and token in pred_token:
                        prob = 1.0 / len(pred_token)
                        #print '  %s: %s | %s' % (formula, prob, [x for x in pred_token if x])
                        candidates.append((formula, prob))
                self.stats['num_formulas_per_token'].append(len(formulas))
                self.stats['num_consistent_formulas_per_token'].append(len(candidates))
            if len(candidates) == 0:
                print 'WARNING: no way to generate %s' % (token,)
                self.executor.kb.dump()
                state.dump()
                return None

        return formula_token_candidates
