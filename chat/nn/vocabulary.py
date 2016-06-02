"""A vocabulary for a neural model."""
import numpy
import os
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T


class Vocabulary:
    """A vocabulary of words, and their embeddings.

    By convention, the end-of-sentence token '</s>' is 0, and
    the unknown word token 'UNK' is 1.

    Concrete subclasses should implement:
      - self.get_theano_embedding(index)
      - self.get_theano_params(self)
      - Create self.word_list in __init__
      - Create self.word_to_index in __init__
    """
    END_OF_SENTENCE = '</s>'
    END_OF_SENTENCE_INDEX = 0
    UNKNOWN = 'UNK'
    UNKNOWN_INDEX = 1
    NUM_SPECIAL_SYMBOLS = 2
    
    def get_theano_embedding(self, index):
        """Get theano embedding for given word index."""
        raise NotImplementedError

    def get_theano_params(self):
        """Get theano parameters to back-propagate through."""
        raise NotImplementedError

    def get_theano_all(self):
        """By default, same as self.get_theano_params()."""
        return self.get_theano_params()

    def get_index(self, word):
        if word is None:
            return -1
        if word in self.word_to_index:
            return self.word_to_index[word]
        return self.word_to_index[self.UNKNOWN]

    def get_word(self, index):
        if index == -1:
            return None
        return self.word_list[index]

    def words_to_indices(self, words):
        return [self.get_index(w) for w in words]

    def indices_to_words(self, indices):
        return [self.get_word(i) for i in indices]

    def size(self):
        return len(self.word_list)

class RawVocabulary(Vocabulary):
    """A vocabulary that's initialized randomly."""

    def __init__(self, word_list, emb_size, float_type=numpy.float64, init_embedding=True):
        """Create the vocabulary.

        Args:
          word_list: List of words that occurred in the training data.
          emb_size: dimension of word embeddings
        """
        self.word_list = [self.UNKNOWN] + word_list
        self.word_to_index = dict((x[1], x[0]) for x in enumerate(self.word_list))
        self.emb_size = emb_size
        self.float_type = float_type

        # Embedding matrix
        if init_embedding:
            self.emb_mat = theano.shared(
                name='vocab_emb_mat',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.size(), emb_size)).astype(theano.config.floatX))

    def get_theano_embedding(self, i):
        return self.emb_mat[i]

    def get_theano_params(self):
        return [self.emb_mat]

    def __getstate__(self):
        return self.word_list[2:], self.emb_size, self.float_type, numpy.asarray(self.emb_mat.get_value())

    def __setstate__(self, state):
        self.emb_mat = theano.shared(
            name='vocab_emb_mat',
            value=state.astype(theano.config.floatX)
        )


class GloveVocabulary(Vocabulary):
    """A vocabulary initialized with GloVe vectors.

    Can choose whether or not to back-propagate through the vectors.
    Will always back-prop through </s> and UNK vectors.
    """
    GLOVE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data/glove')

    def __init__(self, data_words, emb_size, hold_fixed=True, float_type=numpy.float64):
        """Read in GloVe vectors, create the vocabulary.

        Args:
          data_words: List of words that occurred in the data.
          emb_size: dimension of word embeddings.  Expects a corresponding file
            to be found in GLOVE_DIR.
          hold_fixed: Whether to hold word vectors fixed during training.
            Does not apply to </s> or UNK vectors
            (these are randomly initialized and never fixed).
        """
        data_word_set = set(data_words)
        self.word_list = [self.END_OF_SENTENCE, self.UNKNOWN]
        self.emb_size = emb_size
        self.hold_fixed = hold_fixed
        self.float_type = float_type

        # Keep separate embeddings for </s> and UNK ("special" words)
        self.eos_vec = theano.shared(
            name='vocab_eos',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, self.emb_size).astype(theano.config.floatX))
        self.unk_vec = theano.shared(
            name='vocab_unk',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, self.emb_size).astype(theano.config.floatX))

        # Check if GloVe vectors of this dimension exist
        glove_file = os.path.join(self.GLOVE_DIR, 'glove.6B.%dd.txt' % emb_size)
        if not os.path.isfile(glove_file):
            raise ValueError('Expected file %s, not found.' % glove_file)

        # Read GloVe vectors
        print >> sys.stderr, 'Reading GloVe vectors...'
        # Pad with 2 rows of 0's because theano ifelse is not lazy
        # when gradients get involved.
        raw_mat = [[0.0] * self.emb_size] * self.NUM_SPECIAL_SYMBOLS
        with open(glove_file) as f:
            for line in f:
                toks = line.split(' ')
                word = toks[0]
                if word not in data_word_set: continue
                vec = [float(x) for x in toks[1:]]
                self.word_list.append(word)
                raw_mat.append(vec)
        self.word_to_index = dict((x[1], x[0]) for x in enumerate(self.word_list))
        self.glove_mat = theano.shared(
            name='vocab_glove_mat',
            value=numpy.array(raw_mat, dtype=self.float_type).astype(theano.config.floatX))
        print >> sys.stderr, 'Finished reading GloVe vectors.'

    def get_theano_embedding(self, i):
        return ifelse(T.lt(i, self.NUM_SPECIAL_SYMBOLS),
                      ifelse(T.eq(i, self.END_OF_SENTENCE_INDEX),
                             self.eos_vec, self.unk_vec),
                      self.glove_mat[i])

    def get_theano_params(self):
        params = [self.eos_vec, self.unk_vec]
        if not self.hold_fixed:
            params.append(self.glove_mat)
        return params

    def get_theano_all(self):
        return [self.eos_vec, self.unk_vec, self.glove_mat]
