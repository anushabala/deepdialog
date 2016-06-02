import os, sys
import string
import random
import json
from collections import defaultdict
import re
from chat.modeling.executor import Executor, is_entity
from chat.lib import sample_utils
from chat.nn import vocabulary
from chat.modeling import tokens as mytokens

def utterance_to_tokens(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # utterance = utterance.translate(string.maketrans('', ''), string.punctuation)  # Remove punctuation
    # raw_tokens = utterance.split(' ')
    tokens = re.findall(r"[\w']+|[.,!?;]", utterance)
    return tokens 

### Serializing and deserializing formulas

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

def is_str_formula(formula):
    return formula.startswith('!') and len(formula) > 1

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
            if not is_entity(formula_token_candidates[i]):
                continue
            psi = 0
            for token, weight in formula_token_candidates[i]:
                psi += weight
            self.weight *= psi

    def to_json(self):
        def render(candidates):
            if isinstance(candidates, list):
                candidates = sample_utils.sorted_candidates(candidates)
                return [(render_formula(token), weight) for token, weight in candidates]
            return candidates  # single word

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
            if not isinstance(candidates, list):
                return candidates  # Single word
            return ','.join( \
                render_formula(token) + ('' if weight == 1 else ':%.3f' % weight) \
                for token, weight in sample_utils.sorted_candidates(candidates) \
            )
        print '>>> weight=%s' % self.weight()
        for message in self.messages:
            print '  who=%s: %s | %s' % (
                message.who,
                ' '.join(message.raw_tokens),
                ' '.join(map(render_candidates, message.formula_token_candidates))
            )

def add_arguments(parser):
    parser.add_argument('--scenarios', type=str, help='File containing JSON scenarios', required=True)
    parser.add_argument('--formulas-mode', type=str, help='Which formulas to include (verbatim, basic, full) see executor.py)', default='full')
    parser.add_argument('--beam-size', type=int, help='Maximum number of candidate states to generate per agent/scenario', default=5)

class DialogueTracker(object):
    '''
    Stores potentially many interpretations.
    Main methods:
    DialogueTracker(, ..., box)
    - parse_add(who, tokens, end_turn): when receive an utterance
    - tokens, end_turn = generate_add(who): when want to send an utterance
    '''
    def __init__(self, lexicon, scenario, agent, args, box, stats):
        self.lexicon = lexicon
        self.scenario = scenario
        self.agent = agent
        self.args = args
        self.executor = Executor(scenario, agent, args)
        self.box = box
        self.stats = stats
        self.states = [DialogueState()]  # Possible dialogue states that we could be in.

    def get_states(self):
        return self.states

    def parse_add(self, who, raw_tokens, end_turn):
        #print '##### parse_add who=%s raw_tokens=%s end_turn=%s' % (who, raw_tokens, end_turn)

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

        # Update the RecurrentBox.
        if self.box:
            # Choose most probable state.
            state = self.states[0]
            self.states = [state]
            # Go through the positions and choose the formula that's most
            # likely under a combination of the weight p(x|z) and the RNN probability p(z).
            message = state.messages[-1]
            if len(state.messages) == 1 and message.who == self.agent:  # If agent starting, then pad
                self.box.observe(mytokens.END_TURN, write=False)
            self.box.observe(mytokens.SAY, write=False)
            for i, candidates in enumerate(message.formula_token_candidates):
                if not isinstance(candidates, list):
                    token = candidates
                else:
                    distrib = dict(self.box.generate())
                    # Reweight by p(z)
                    candidates = [(token, distrib.get(token, 0) * weight) for token, weight in candidates]
                    token = sample_utils.sample_candidates(candidates)[0]
                # Commit to that choice
                self.box.observe(token, write=False)
            self.box.observe(mytokens.END_TURN if end_turn else mytokens.END, write=False)

    def compute_state_probs(self):
        return sample_utils.normalize_weights([state.weight() for state in self.states])

    def generate_add(self, who):
        MAX_TOKENS = 20
        '''
        Returns (tokens, end_turn) updates the dialogue tracker.
        Returns None if execution fails.
        Assume there is only one state.
        '''
        if len(self.states) == 0:
            return None, None
        if len(self.states) > 1:
            raise Exception('Can only handle the one state case')

        state = self.states[0]
        entity_tokens = []
        formula_tokens = []
        str_formula_tokens = []
        formula_weights = []
        def try_execute(formula):
            if 'NextMention' in formula:  # Hack: allow NextMention
                return True
            i = len(entity_tokens)
            choices = self.executor.execute(state, who, entity_tokens, i, formula)
            #print 'try_execute', formula, choices
            if choices is None:  # Error or just not ready to execute (e.g., if formula has NextMention)
                return False
            if len(choices) == 0:  # Probably we got derailed...fail carefully
                return False
            if any(c is None for c in choices):  # Rule out MagicType
                return False
            return True

        def execute(i):
            if entity_tokens[i] is not None:
                return
            # Execute the i-th formula and try to fill it out
            formula = formula_tokens[i]
            # Execute: SchoolOf(A) => 'university of pennsylvania'
            choices = self.executor.execute(state, who, entity_tokens, i, formula)
            #print 'execute', formula, choices
            entity_tokens[i] = random.choice(choices)
            formula_weights[i] *= 1.0 / len(choices)

        # Generate a formula list
        # Ideally, we want to reject options which can't be executed properly,
        # but this might result in cyclic dependencies
        self.box.observe(mytokens.SAY, write=True)
        end_turn = True
        while len(formula_tokens) < MAX_TOKENS:
            # Get candidate next tokens (formulas)
            candidates = self.box.generate()
            #print 'GEN %s => %s' % (self.box.prev_token, candidates)
            if len(candidates) == 0:
                print 'WARNING: no candidates!'
                self.box.observe(mytokens.END_TURN, write=True)  # Force an end
                break

            # Filter formulas that don't execute and try to convert to formula
            num_good = num_bad = 0
            for i, (str_formula, weight) in enumerate(candidates):
                if is_str_formula(str_formula):  # e.g., !SchoolOfA
                    formula = parse_formula(str_formula)
                    if not try_execute(formula):  # Zero out formulas that don't work
                        weight = 0
                        num_bad += 1
                    else:
                        num_good += 1
                    candidates[i] = ((str_formula, formula), weight)
                else:  # e.g., 'attend'
                    candidates[i] = ((str_formula, None), weight)
            #print 'generate_add: %d candidates, %d good formulas, %d bad formulas' % (len(candidates), num_good, num_bad)

            # Choose a formula
            if sum(weight for (token, formula), weight in candidates) == 0:
                print 'WARNING: no valid candidates among ', candidates
                break
            (token, formula), weight = sample_utils.sample_candidates(candidates)
            # Commit to that choice
            self.box.observe(token, write=True)
            if token == mytokens.END or token == mytokens.END_TURN:
                end_turn = (token == mytokens.END_TURN)
                break

            str_formula_tokens.append(token)
            formula_tokens.append(formula if formula else token)
            entity_tokens.append(None if formula else token)
            formula_weights.append(weight)
            execute(len(formula_tokens) - 1)
        #print 'generate_add: formula =', str_formula_tokens

        # Take additional passes to resolve formulas that depend on existence
        # of following entities (e.g., '[two] at Facebook').
        while True:
            changed = False
            for i in range(len(entity_tokens)):
                if entity_tokens[i] is not None:
                    continue
                execute(i)
                changed = True
            if not changed:
                break

        if any(token is None for token in entity_tokens):
            print 'WARNING: failed execution of %s' % formula_tokens

        # Now generate the raw token from the entity from the lexical mapping
        # Example: 'university of pennsylvania' => 'upenn'
        # For now, just write things out explicitly.  Later, incorporate lexical mapping
        raw_tokens = []
        for token in entity_tokens:
            if token is None:
                raw_tokens.append('???')
            elif is_entity(token):
                raw_tokens.extend(' '.join(token[0]))  # 'university of pennsylvania'
            else:
                raw_tokens.append(token)

        # Update the state
        formula_token_candidates = [ \
            [(formula, weight)] if is_entity(entity) else entity \
            for entity, formula, weight in zip(entity_tokens, formula_tokens, formula_weights) \
        ]
        message = Message(who, raw_tokens, entity_tokens, formula_token_candidates)
        state = state.extend(message)
        self.states = [state]

        # Return the utterance
        return raw_tokens, end_turn

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
            if not is_entity(token):
                formula_token_candidates.append(token)
            else:
                candidates = []  # List of formula tokens that could be at position i
                formula_token_candidates.append(candidates)
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
                if self.stats:
                    self.stats['num_formulas_per_token'].append(len(formulas))
                    self.stats['num_consistent_formulas_per_token'].append(len(candidates))
                if len(candidates) == 0:
                    print 'WARNING: no way to generate %s' % (token,)
                    self.executor.kb.dump()
                    state.dump()
                    return None

        return formula_token_candidates
