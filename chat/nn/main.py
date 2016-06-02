"""Run tests on toy data for IRW models."""
import argparse
import collections
import itertools
import json
import math
import numpy
import random
import sys
import theano
import os

# IRW imports
from encoderdecoder import EncoderDecoderModel
from encdecspec import VanillaEncDecSpec, GRUEncDecSpec, LSTMEncDecSpec
import spec as specutil
from chat.lib import logstats
from chat.modeling import data_utils
from chat.modeling import dialogue_tracker
from chat.modeling import tokens as mytokens

# Imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vocabulary import GloveVocabulary, RawVocabulary, Vocabulary

CONTINUOUS_MODELS = collections.OrderedDict([
    ('rnn', VanillaEncDecSpec),
    ('gru', GRUEncDecSpec),
    ('lstm', LSTMEncDecSpec),
])

MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', RawVocabulary),
    ('glove_fixed', lambda s, e: GloveVocabulary(s, e, hold_fixed=True)),
    ('glove_not_fixed', lambda s, e: GloveVocabulary(s, e, hold_fixed=False))
])

# Global options
OPTIONS = None


def _parse_args():
    global OPTIONS
    parser = argparse.ArgumentParser(
        description='Test neural alignment model on toy data.')
    parser.add_argument('--hidden-size', '-d', type=int,
                        help='Dimension of hidden units')
    parser.add_argument('--input-embedding-dim', '-i', type=int,
                        help='Dimension of input vectors.')
    parser.add_argument('--output-embedding-dim', '-o', type=int,
                        help='Dimension of output word vectors.')
    parser.add_argument('--num-epochs', '-t', type=int, default=0,
                        help='Number of epochs to train (default is no training).')
    parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                        help='Learning rate (default = 0.1).')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Size of mini-batch (default is SGD).')
    parser.add_argument('--continuous-spec', '-c',
                        help='type of continuous model (options: [%s])' % (
                            ', '.join(CONTINUOUS_MODELS)), default='lstm')
    parser.add_argument('--model', '-m',
                        help='type of overall model (options: [%s])' % (
                            ', '.join(MODELS)), default='encoderdecoder')
    parser.add_argument('--input-vocab-type',
                        help='type of input vocabulary (options: [%s])' % (
                            ', '.join(VOCAB_TYPES)), default='raw')
    parser.add_argument('--output-vocab-type',
                        help='type of output vocabulary (options: [%s])' % (
                            ', '.join(VOCAB_TYPES)), default='raw')
    parser.add_argument('--no-eos-on-output', action='store_true',
                        help='Do not add end of sentence token to output sequence.')
    parser.add_argument('--reverse-input', action='store_true',
                        help='Reverse the input sentence (intended for encoder-decoder).')
    parser.add_argument('--float32', action='store_true',
                        help='Use 32-bit floats (default is 64-bit/double precision).')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of candidatesto sample for each training example.')

    # Input paths
    parser.add_argument('--data-prefix', help='If train and dev not specified, then construct paths using this.')
    parser.add_argument('--train-data', help='Path to training data.')
    parser.add_argument('--train-max-examples', type=int, help='Maximum number of training examples.')
    parser.add_argument('--dev-data', help='Path to dev data.')
    parser.add_argument('--dev-max-examples', type=int, help='Maximum number of dev examples.')
    parser.add_argument('--load-params', help='Path to load parameters, will ignore other passed arguments.')

    # Output paths (mostly)
    parser.add_argument('--out-dir', help='If specify, save all files to this directory.')
    parser.add_argument('--save-params', help='Path to save parameters.')
    parser.add_argument('--stats-file', help='Path to save statistics (JSON format).')
    parser.add_argument('--train-eval-file', help='Path to save train eval results (JSON format).')
    parser.add_argument('--dev-eval-file', help='Path to save dev eval results (JSON format).')

    # Theano-specific options
    parser.add_argument('--theano-fast-compile', action='store_true',
                        help='Run Theano in fast compile mode.')
    parser.add_argument('--theano-profile', action='store_true',
                        help='Turn on profiling in Theano.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU.')

    dialogue_tracker.add_arguments(parser)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    OPTIONS = parser.parse_args()

    # Some basic error checking
    if OPTIONS.continuous_spec not in CONTINUOUS_MODELS:
        print >> sys.stderr, 'Error: continuous_spec must be in %s' % (
            ', '.join(CONTINUOUS_MODELS))
        sys.exit(1)
    if OPTIONS.model not in MODELS:
        print >> sys.stderr, 'Error: model must be in %s' % (
            ', '.join(MODELS))
        sys.exit(1)
    if OPTIONS.input_vocab_type not in VOCAB_TYPES:
        print >> sys.stderr, 'Error: input_vocab_type must be in %s' % (
            ', '.join(VOCAB_TYPES))
        sys.exit(1)
    if OPTIONS.output_vocab_type not in VOCAB_TYPES:
        print >> sys.stderr, 'Error: output_vocab_type must be in %s' % (
            ', '.join(VOCAB_TYPES))
        sys.exit(1)


def configure_theano():
    if OPTIONS.theano_fast_compile:
        theano.config.mode = 'FAST_COMPILE'
    else:
        theano.config.mode = 'FAST_RUN'
        theano.config.linker = 'cvm'
    if OPTIONS.theano_profile:
        theano.config.profile = True
    if OPTIONS.float32 or OPTIONS.gpu:
        theano.config.floatX = 'float32'


def load_dataset(name, path, max_examples):
    print 'load_dataset(%s, %s)' % (name, path)
    # Just load the whole JSON file, don't do any processing (everything done at training time).
    data = json.load(open(path, 'r'))
    if max_examples is not None:
        data = data[:max_examples]
    logstats.add('data', name, 'num_examples', len(data))
    return data


def get_word_list(raw_data, is_agent=True):
    '''
    Extract the list of words that are used by the agent (or partner).
    '''
    tokens = set()
    # Add special tokens
    for token in mytokens.TOKENS:
        tokens.add(token)
    for ex in raw_data:
        agent = ex['agent'] 
        who = agent if is_agent else 1 - agent
        for state in ex['states']:
            for message in state['messages']:
                if message['who'] != who:
                    continue
                for candidates in message['formula_token_candidates']:
                    if not isinstance(candidates, list):
                        tokens.add(candidates)
                    else:
                        for token, weight in candidates:
                            tokens.add(token)
    print 'get_word_list: is_agent=%s has %s words' % (is_agent, len(tokens))
    return sorted(list(tokens))


def get_input_vocabulary(examples):
    word_list = get_word_list(examples, is_agent=False)
    constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
    if OPTIONS.float32:
        return constructor(word_list, OPTIONS.input_embedding_dim,
                           float_type=numpy.float32)
    else:
        return constructor(word_list, OPTIONS.input_embedding_dim)


def get_output_vocabulary(examples):
    word_list = get_word_list(examples, is_agent=True)
    constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
    if OPTIONS.float32:
        return constructor(word_list, OPTIONS.output_embedding_dim,
                           float_type=numpy.float32)
    else:
        return constructor(word_list, OPTIONS.output_embedding_dim)


def update_model(model, dataset):
    """Update model for new dataset if fixed word vectors were used."""
    need_new_model = False
    if OPTIONS.input_vocab_type == 'glove_fixed':
        in_vocabulary = get_input_vocabulary(dataset)
        need_new_model = True
    else:
        in_vocabulary = model.in_vocabulary

    if OPTIONS.output_vocab_type == 'glove_fixed':
        out_vocabulary = get_output_vocabulary(dataset)
        need_new_model = True
    else:
        out_vocabulary = model.out_vocabulary

    if need_new_model:
        spec = model.spec
        spec.set_in_vocabulary(in_vocabulary)
        spec.set_out_vocabulary(out_vocabulary)
        model = get_model(spec)  # Create a new model!
    return model


def get_continuous_spec(in_vocabulary, out_vocabulary):
    constructor = CONTINUOUS_MODELS[OPTIONS.continuous_spec]
    return constructor(in_vocabulary, out_vocabulary, OPTIONS.hidden_size)


def get_model(spec):
    constructor = MODELS[OPTIONS.model]

    if OPTIONS.float32:
        model = constructor(OPTIONS, spec, float_type=numpy.float32)
    else:
        model = constructor(OPTIONS, spec)
    return model


def open_if_specified(file_name, mode):
    if file_name:
        return open(file_name, mode)
    else:
        return None


def close_if_specified(fh):
    if fh:
        fh.close()


def run():
    configure_theano()

    # Set default output flags
    if OPTIONS.out_dir:
        if not os.path.exists(OPTIONS.out_dir):
            os.mkdir(OPTIONS.out_dir)
        if not OPTIONS.save_params: OPTIONS.save_params = os.path.join(OPTIONS.out_dir, 'params')
        if not OPTIONS.stats_file: OPTIONS.stats_file = os.path.join(OPTIONS.out_dir, 'stats.json')
        if not OPTIONS.train_eval_file: OPTIONS.train_eval_file = os.path.join(OPTIONS.out_dir, 'train_eval.json')
        if not OPTIONS.dev_eval_file: OPTIONS.dev_eval_file = os.path.join(OPTIONS.out_dir, 'dev_eval.json')
    if OPTIONS.data_prefix:
        if not OPTIONS.train_data: OPTIONS.train_data = OPTIONS.data_prefix + 'train.json'
        if not OPTIONS.dev_data: OPTIONS.dev_data = OPTIONS.data_prefix + 'dev.json'

    logstats.init(OPTIONS.stats_file)

    # Read data
    if OPTIONS.train_data:
        train_examples = load_dataset('train', OPTIONS.train_data, OPTIONS.train_max_examples)
    else:
        train_examples = []
    if OPTIONS.dev_data:
        dev_examples = load_dataset('dev', OPTIONS.dev_data, OPTIONS.dev_max_examples)
    else:
        dev_examples = []

    # Create vocab
    if OPTIONS.load_params:
        print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_params
        spec = specutil.load(OPTIONS.load_params)
        in_vocabulary = spec.in_vocabulary
        out_vocabulary = spec.out_vocabulary
    elif OPTIONS.train_data:
        print >> sys.stderr, 'Initializing parameters...'
        in_vocabulary = get_input_vocabulary(train_examples)
        out_vocabulary = get_output_vocabulary(train_examples)
        spec = get_continuous_spec(in_vocabulary, out_vocabulary)
    else:
        raise Exception('Must either provide parameters to load or training data.')

    # Set up models
    model = get_model(spec)
    dev_model = update_model(model, dev_examples)

    # Train!
    model.train_loop(train_examples, dev_examples)

    if OPTIONS.save_params:
        print >> sys.stderr, 'Saving parameters...'
        spec.save(OPTIONS.save_params)


if __name__ == '__main__':
    _parse_args()
    run()
