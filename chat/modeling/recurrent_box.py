import math
from chat.lib.cpt import ConditionalProbabilityTable
from chat.lib import sample_utils
from chat.modeling import tokens as mytokens
from chat.modeling import data_utils

class RecurrentBox(object):
    '''
    This is an abstract interface that the RNN models should implement.
    The DialogueTracker uses the RNN model.
    '''

    def generate(self):
        '''
        Return a n-best list of (token, weight) pairs.
        '''
        raise NotImplementedError

    def observe(self, token, write):
        '''
        Update internal state upon observing token
        write: whether we wrote this or are reading it
        '''
        raise NotImplementedError


class NgramBox(object):
    '''
    A simple of example of a RecurrentBox that's a simple bigram model:
        p(token | prev_token)
    '''
    def __init__(self, cpt):
        self.cpt = cpt
        self.prev_tokens = []

    def generate(self):
        result = None
        # Find the largest n that matches (could use better interpolation)
        for n in range(1, len(self.prev_tokens)+2):
            history = tuple(self.prev_tokens[(len(self.prev_tokens)-(n-1)):])
            new_result = self.cpt.get(history)
            #print self.prev_tokens, n, history, 'hh'
            if new_result is None:
                break
            result = new_result
        #print n
        #print list(result.keys())[:10]
        return result.items()

    def observe(self, token, write):
        #print 'OBSERVE', token
        self.prev_tokens.append(token)
        if len(self.prev_tokens) >= 10:
            self.prev_tokens.pop(0)

def learn_ngram_model(n, data):
    print 'learn_ngram_model on %d examples' % len(data)
    cpt = ConditionalProbabilityTable()
    for ex in data:
        states = ex['states']
        for trial in range(5):  # Sample several trajectories
            messages, log_weight = data_utils.sample_trajectory(states) 
            seqs = data_utils.messages_to_sequences(ex['agent'], messages)
            tokens = [token for seq in seqs for token in seq]
            for i in range(len(tokens)):
                for j in range(max(0, i - n + 1), i + 1):
                    cpt[tuple(tokens[j:i])][tokens[i]] += 1
    cpt.normalize()
    #cpt.dump()
    return cpt
