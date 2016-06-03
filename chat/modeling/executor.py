import os, sys
import string
import random
from collections import defaultdict
from kb import KB

def of(rel):
    return rel + 'Of'
def has(rel):
    return 'Has' + rel
def is_entity(token):
    return not isinstance(token, basestring)

def sort_by_freq(l):
    # Return the distinct elements of l sorted by descending frequency
    # ['a', 'b', 'b'] => ['b', 'a']
    counts = defaultdict(int)
    for x in l:
        counts[x] += 1
    return sorted(counts.keys(), key=lambda x : counts[x], reverse=True)

class Executor(object):
    '''
    Executes logical forms on a world state (scenario, state, ...)
    '''
    def __init__(self, scenario, agent, args):
        self.kb = KB(scenario, agent)
        self.agent = agent
        self.args = args

    def generate_formulas(self, state, who, tokens, i):
        def apply_relation_of(formulas):
            for f in formulas:
                for r in self.kb.relations:
                    yield (of(r), f)
        def apply_has_relation(formulas):
            for f in formulas:
                for r in self.kb.relations:
                    yield (has(r), f)
        def identity(f): return f
        def select_type(f, t): return ('SelectType', f, t)
        def count(f): return ('Count', f)
        def get(f, i): return ('Get', f, str(i))
        def get_first(f): return get(f, 0)
        def get_last(f): return get(f, -1)
        def intersect(f, g): return ('And', f, g)
        def diff(f, g): return ('Diff', f, g)
        def iterate(combiner, formulas1, formulas2):
            for f1 in formulas1:
                for f2 in formulas2:
                    yield combiner(f1, f2)

        # Don't abstract away anything
        if self.args.formulas_mode == 'verbatim':
            yield ('Value', tokens[i])
            return

        # Handle cases where partner says something that we don't have in our
        # KB or is a previous mention.  Note: generate the type by peeking at
        # the current entity.
        if self.agent != who:
            yield ('MagicType', tokens[i][1])  # MagicType(company)

        if self.args.formulas_mode == 'recurse':
            # More recursive and braindead way of generating logical forms.
            # Leads to many more hypotheses.
            combiners = [intersect, diff]
            selectors = [identity, get_first, get_last, count]

            # Conventions:
            # a: friends, b: mentions
            # Number is depth of recursion
            a1 = ['A', 'FriendOfA']  # e.g., FriendOfA
            b1 = [select(select_type(x, t)) for x in ['MentionOfA', 'MentionOfB', 'NextMention'] for t in self.kb.types for select in [identity, get_last]]  # e.g., SelectType(MentionOfA,company)

            # Grow depth on each of a and b
            a2 = list(apply_relation_of(a1)) + list(apply_has_relation(a1))  # e.g., CompanyOf(FriendOfA)
            b2 = list(apply_relation_of(b1)) + list(apply_has_relation(b1))  # e.g., CompanyOf(MentionOfA)

            # Combine a and b
            a1b1 = [combine(x1, x2) for x1 in a1 for x2 in b1 for combine in combiners] # e.g., And(FriendOfA,MentionOfA)
            a1b2 = [combine(x1, x2) for x1 in a1 for x2 in b2 for combine in combiners] # e.g., And(FriendOfA,HasCompany(MentionOfA))
            a2b1 = [combine(x1, x2) for x1 in a2 for x2 in b1 for combine in combiners] # e.g., And(CompanyOf(FriendOfA),MentionOfA)

            # Only return things that are typed
            for f in [select(x) for x in a1 + b1 + a2 + b2 + a1b1 + a1b2 + a2b1 for select in selectors]:
                yield f
            return

        # Generate my relations
        for f in apply_relation_of('A'):  # e.g., CompanyOf(A)
            yield f
            if self.args.formulas_mode == 'full':
                yield get_first(f)

        # Generate relations of my friends
        for f in apply_relation_of(['FriendOfA']):  # e.g., CompanyOf(FriendOfA)[0]
            yield f
            if self.args.formulas_mode == 'full':
                yield get_first(f)

        # Generate friends
        for f in ['FriendOfA']:  # e.g., FriendOfA
            yield f
            if self.args.formulas_mode == 'full':
                yield get_first(f)

        # Generate things that the partner said
        for t in self.kb.types:
            yield select_type('MentionOfB', t)

        # Generate numbers in basic mode, or else there's no way they can't get generated.
        if self.args.formulas_mode == 'basic':
            yield ('Value', ('1', 'number'))
            yield ('Value', ('2', 'number'))
            yield ('Value', ('3', 'number'))

        if self.args.formulas_mode == 'full':
            # Generate friends with properties mentioned in the utterances
            # e.g., "[two] facebook" Count(And(FriendOfA,HasCompany(NextMention)))
            for f in apply_has_relation([get_last('MentionOfA'), 'MentionOfA', get_last('MentionOfB'), 'MentionOfB', 'NextMention']):
                for combine in [intersect, diff]:
                    for select in [count, get_first]:
                        yield select(combine('FriendOfA', f))

            # Generate common attributes to last thing mentioned: e.g., cooking likes [evening]
            for f in apply_relation_of(apply_has_relation([get_last('MentionOfA')])):
                yield f
            # Generate common attributes to last two things mentioned: e.g., cooking lies evening and [outdoors]
            for f in apply_relation_of(iterate(intersect, apply_has_relation([get_last('MentionOfA')]), apply_has_relation([get('MentionOfA', -2)]))):
                yield f

            # Generate friends not mentioned before
            yield diff('FriendOfA', 'MentionOfA')

            # Last things mentioned
            for select in [identity, get_last]:
                for f in ['MentionOfA', 'MentionOfB']:
                    #yield select(f)  # last mentioned (too untyped)
                    for g in apply_relation_of(['FriendOfA']):  # e.g., last company mentioned
                        yield select(intersect(f, g))

    def execute(self, state, who, tokens, i, formula):
        '''
        Note: return None if this logical form is invalid.
        Things that are invalid: no-ops (e.g. getting a single element)
        '''
        table = self.kb.table
        # Base cases
        if isinstance(formula, basestring):
            if formula == 'A':
                return [table[0]['Name']]  # agent
            if formula == 'FriendOfA':
                return [row['Name'] for row in table[1:]]
            if formula == 'MentionOfA':
                # Include both things in previous utterances (in state) and in same utterances
                return state.mentions[self.agent] + ([x for x in tokens[:i-1] if is_entity(x)] if i-1 >= 0 else [])
            if formula == 'MentionOfB':
                return state.mentions[1-self.agent]
            if formula == 'NextMention':
                mentions = [x for x in tokens[i+1:] if is_entity(x)]
                if len(mentions) == 0: return None
                return [mentions[0]]
            try:
                int(formula)
                return [(formula, 'number')]
            except:
                pass
            raise Exception('Invalid base formula: %s' % formula)

        func, args = formula[0], formula[1:]
        if func == 'MagicType':
            if i >= len(tokens):  # Current position not available (at generation test time)
                return None
            return [tokens[i]] + [None] * 100  # Just return the token with junk to lower the prob
        if func == 'Value':
            return [args[0]]  # Don't recursively execute arguments
        if func == 'SelectType':
            # Set of items to select on
            s = self.execute(state, who, tokens, i, args[0])
            if s is None:
                return None
            # Type
            t = args[1]
            # Return all elements that match the type
            return [x for x in s if x is not None and x[1] == t]

        orig_args = args
        args = [self.execute(state, who, tokens, i, arg) for arg in args]
        if any(arg is None for arg in args):  # None signifies failure
            return None

        # Joins (note: this is done with respect to the table, which is A + friends)
        if func.endswith('Of'):
            rel = func[:-2]
            return sort_by_freq([row[rel] for row in table if row['Name'] in args[0]])
        if func.startswith('Has'):
            rel = func[3:]
            return sort_by_freq([row['Name'] for row in table if row[rel] in args[0]])

        # Set intersection / difference
        if func == 'And':
            result = [x for x in args[0] if x in args[1]]
            return result if result not in args else None
        if func == 'Diff':
            result = [x for x in args[0] if x not in args[1]]
            return result if result not in args else None

        # Return singleton lists
        if func == 'Count':
            return [(str(len(args[0])), 'number')]
        if func == 'Get':
            i = int(args[1][0][0])
            try:
                result = [args[0][i]]
                return result if result not in args else None
            except IndexError:
                #print i, len(args[0])
                return None
