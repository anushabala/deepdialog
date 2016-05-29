import os, sys
import string
import random
from collections import defaultdict

def of(rel): return rel + 'Of'
def has(rel): return 'Has' + rel
def is_entity(token): return not isinstance(token, basestring)

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
        self.scenario = scenario
        self.agent = agent
        self.args = args

        # Convert JSON format into a nicer table format so we can do joins
        self.table = []  # List of dictionaries
        def normalize(s): return s.lower()
        def convert(person_info):
            return {
                'Name': (normalize(person_info['name']), 'person'),
                'Company': (normalize(person_info['company']['name']), 'company'),
                'School': (normalize(person_info['school']['name']), 'school'),
                'Major': (normalize(person_info['school']['major']), 'major')
            }

        agent_info = self.scenario['agents'][agent]  # Information specific to agent
        self.table.append(convert(agent_info['info']))
        for friend in agent_info['friends']:
            self.table.append(convert(friend))
        self.dump_kb()

    def dump_kb(self):
        print '############ KB %s, agent=%s:' % (self.scenario['uuid'], self.agent,)
        for row in self.table:
            print ' ', row

    def generate_formulas(self, state, who, tokens, i):
        relations = ['School', 'Major', 'Company']
        def apply_relation_of(formulas):
            for f in formulas:
                for r in relations:
                    yield (of(r), f)
        def apply_has_relation(formulas):
            for f in formulas:
                for r in relations:
                    yield (has(r), f)
        def count(f): return ('Count', f)
        def get(f, i): return ('Get', f, str(i))
        def get_first(f): return get(f, 0)
        def get_last(f): return get(f, -1)
        def intersect(f, g): return ('And', f, g)
        def diff(f, g): return ('Diff', f, g)

        # Only for the other person - last resort - just say something out of nowhere.
        if self.agent != who:
            yield ('MagicType', tokens[i][1])  # MagicType(company)

        # Generate my relations
        for f in apply_relation_of('A'):  # e.g., CompanyOf(A)
            if self.args.formulas_mode == 'full':
                yield get_first(f)
            yield f

        # Generate relations of my friends
        for f in apply_relation_of(['FriendOfA']):  # e.g., CompanyOf(FriendOfA)[0]
            if self.args.formulas_mode == 'full':
                yield get_first(f)
            yield f

        # Generate friends
        for f in ['FriendOfA']:  # e.g., FriendOfA
            if self.args.formulas_mode == 'full':
                yield get_first(f)
            yield f

        if self.args.formulas_mode == 'full':
            # Generate friends with properties mentioned in the utterances
            for f in apply_has_relation([get_last('MentionOfB'), 'MentionOfA', get_last('MentionOfB'), 'MentionOfB', 'NextMention']):  # e.g., Count(And(FriendOfA,HasCompany(NextMention)))
                for combine in [intersect, diff]:
                    for select in [count, get_first]:
                        yield select(combine('FriendOfA', f))

            # Generate friends not mentioned before
            yield diff('FriendOfA', 'MentionOfA')

            # Last things mentioned
            yield get_last('MentionOfA')
            yield get_last('MentionOfB')
            yield 'MentionOfB'

    def execute(self, state, who, tokens, i, formula):
        # Base cases
        if isinstance(formula, basestring):
            if formula == 'A':
                return [self.table[0]['Name']]  # agent
            if formula == 'FriendOfA':
                return [row['Name'] for row in self.table[1:]]
            if formula == 'MentionOfA':
                # Include both things in previous utterances (in state) and in same utterances
                return state.mentions[self.agent] + [x for x in tokens[:i-1] if is_entity(x)]
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
            return [tokens[i]] + [None] * 100  # Just return the token with junk to lower the prob
        #if func == 'Value':
            #return [args[0]]  # Don't recursively execute arguments
        args = [self.execute(state, who, tokens, i, arg) for arg in args]
        if any(arg is None for arg in args):  # None signifies failure
            return None

        # Joins (note: this is done with respect to the table, which is A + friends)
        if func.endswith('Of'):
            rel = func[:-2]
            return sort_by_freq([row[rel] for row in self.table if row['Name'] in args[0]])
        if func.startswith('Has'):
            rel = func[3:]
            return sort_by_freq([row['Name'] for row in self.table if row[rel] in args[0]])

        # Set intersection / difference
        if func == 'And':
            return [x for x in args[0] if x in args[1]]
        if func == 'Diff':
            if len(args[1]) == 0:  # Presupposition failure: subtracting off nothing
                return None
            return [x for x in args[0] if x not in args[1]]

        # Return singleton lists
        if func == 'Count':
            return [(str(len(args[0])), 'number')]
        if func == 'Get':
            i = int(args[1][0][0])
            try:
                return [args[0][i]]
            except IndexError:
                #print i, len(args[0])
                return None
