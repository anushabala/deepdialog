from collections import defaultdict
from chat.lib import sample_utils

class ConditionalProbabilityTable(object):
    '''
    A conditional probability table is a mapping from string to string to probability (double).
    '''
    def __init__(self):
        self.data = defaultdict(lambda : defaultdict(float))

    def __getitem__(self, k1):
        return self.data[k1]

    def get(self, k1):
        return self.data.get(k1)

    def update(self, source):
        for k1, source_m in source.items():
            target_m = self.data[k1]
            for k2, v in source_m.items():
                target_m[k2] += v

    def normalize(self):
        for k1, m in self.data.items():
            for k2, v in sample_utils.normalize_candidates(m.items()):
                m[k2] = v

    def dump(self):
        for k1, m in self.data.items():
            for k2, v in sample_utils.sorted_candidates(m.items()):
                print '%s\t%s\t%s' % (k1, k2, v)
