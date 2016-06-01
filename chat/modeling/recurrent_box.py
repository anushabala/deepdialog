import math
from chat.lib.cpt import ConditionalProbabilityTable
from chat.lib import sample_utils
from chat.modeling import tokens as mytokens
from chat.modeling import data_utils

def RecurrentBox(object):
    '''
    This is an abstract interface that the RNN models should implement.
    The DialogueTracker uses the RNN model.
    '''

    def generate(self):
        '''
        Return a n-best list of (token, weight) pairs.
        '''
        raise NotImplementedError

    def observe(self, token):
        '''
        Update internal state upon observing token.
        '''
        raise NotImplementedError


class BigramBox(object):
    '''
    A simple of example of a RecurrentBox that's a simple bigram model:
        p(token | prev_token)
    '''
    def __init__(self, cpt):
        self.cpt = cpt
        self.prev_token = None

    def generate(self):
        return self.cpt[self.prev_token].items()

    def observe(self, token):
        self.prev_token = token

def learn_bigram_model(data):
    print 'learn_bigram_model on %d examples' % len(data)
    cpt = ConditionalProbabilityTable()
    for ex in data:
        states = ex['states']
        for i in range(5):  # Sample several trajectories
            messages, log_weight = data_utils.sample_trajectory(states) 
            seqs = data_utils.messages_to_sequences(ex['agent'], messages)
            prev_token = None
            for seq in seqs:
                for token in seq:
                    cpt[prev_token][token] += 1
                    prev_token = token 
    cpt.normalize()
    #cpt.dump()
    return cpt
