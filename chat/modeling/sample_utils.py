import random
import numpy

def normalize_weights(weights):
    '''
    [3, 2] => [0.6, 0.4]
    '''
    if len(weights) == 0:
        return []
    s = sum(weights)
    if s == 0:
        print 'WARNING: zero normalization'
        return weights
    return [1.0 * w / s for w in weights]

def sample_candidates(candidates):
    '''
    [('a', 0.2), ('b', 0.8)] => 'a' or 'b'
    '''
    weights = [weight for token, weight in candidates]
    sums = numpy.array(weights).cumsum()
    i = sums.searchsorted(random.random() * sums[-1])
    return candidates[i][0]

def sorted_candidates(candidates):
    return sorted(candidates, key=lambda (token, weight) : weight, reverse=True)
