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
import utils
from encoderdecoder import EncoderDecoderModel
from encdecspec import VanillaEncDecSpec, GRUEncDecSpec, LSTMEncDecSpec
import spec as specutil
import logstats
import sample_candidates

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
    ('raw', lambda s, e: RawVocabulary.extract_from_sentences(s, e)),
    ('glove_fixed', lambda s, e: GloveVocabulary.extract_from_sentences(s, e, hold_fixed=True)),
    ('glove_not_fixed', lambda s, e: GloveVocabulary.extract_from_sentences(s, e, hold_fixed=False))
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
    parser.add_argument('--num_epochs', '-t', type=int, default=0,
                        help='Number of epochs to train (default is no training).')
    parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                        help='Learning rate (default = 0.1).')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Size of mini-batch (default is SGD).')
    parser.add_argument('--continuous-spec', '-c',
                        help='type of continuous model (options: [%s])' % (
                            ', '.join(CONTINUOUS_MODELS)))
    parser.add_argument('--model', '-m',
                        help='type of overall model (options: [%s])' % (
                            ', '.join(MODELS)))
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
    parser.add_argument('--beam-size', '-k', type=int, default=0,
                        help='Use beam search with given beam size (default is greedy).')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of candidatesto sample for each training example.')

    # Input paths
    parser.add_argument('--data-prefix', help='If train and dev not specified, then construct paths using this.')
    parser.add_argument('--train-data', help='Path to training data.')
    parser.add_argument('--dev-data', help='Path to dev data.')
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


def load_dataset(name, path):
    print 'load_dataset(%s, %s)' % (name, path)
    # Dataset format: Let X be the top-level JSON object.
    # X is a list of examples.
    # X[i]['states'] is a list of states (possible parses of the raw tokens into entities).
    # X[i]['states'][j].prob: sample this state with this probability.
    # X[i]['states'][j].messages: list of messages (one person talking)
    # X[i]['states'][j].messages[k]['formula_token_candidates'][p]: is a list of candidates for the p-th position
    # X[i]['states'][j].messages[k]['formula_token_candidates'][p][l]: the l-th candidate, which is a token-prob pair
    # with open(path) as f:
    #   for line in f:
    #     columns = line.rstrip('\n').split('\t')
    #     left_seqs = [s.strip() for s in columns[0].split("|")]
    #     right_seqs = [s.strip() for s in columns[1].split("|")]
    #     agent_idx = int(columns[2])
    #     scenario_id = columns[3]
    #     metadata.append((agent_idx, scenario_id))
    #     assert(len(left_seqs) == len(right_seqs))
    #     data.append(list(zip(left_seqs, right_seqs)))
    #

    # just load the whole JSON file, don't do any processing (everything done at training time)
    data = json.load(open(path, 'r'))
    logstats.add('data', name, 'num_examples', len(data))

    return data


def get_sentences_from_raw_data(raw_data, input_data=True):
    sentences = []
    for ex in raw_data:
        this_agent = ex["agent"]
        for state in ex["states"]:
            for message in state["messages"]:
                if input_data and message["who"] == this_agent:
                    continue
                if not input_data and message["who"] != this_agent:
                    continue
                sentence = []
                for token_candidates in message["formula_token_candidates"]:
                    if len(token_candidates) == 1:
                        sentence.append(token_candidates[0][0])
                    else:
                        sentence.extend([t[0] for t in token_candidates])
                print sentence
                sentences.append(" ".join(sentence))
    return sentences


def get_input_vocabulary(raw_data):
    sentences = get_sentences_from_raw_data(raw_data, True)
    # sentences = [x[0] for l in dataset for x in l]
    constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
    if OPTIONS.float32:
        return constructor(sentences, OPTIONS.input_embedding_dim,
                           float_type=numpy.float32)
    else:
        return constructor(sentences, OPTIONS.input_embedding_dim)


def get_output_vocabulary(raw_data):
    sentences = get_sentences_from_raw_data(raw_data, False)
    constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
    if OPTIONS.float32:
        return constructor(sentences, OPTIONS.output_embedding_dim,
                           float_type=numpy.float32)
    else:
        return constructor(sentences, OPTIONS.output_embedding_dim)


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


def preprocess_data(in_vocabulary, out_vocabulary, raw):
    print 'preprocess_data(): %s examples' % len(raw)
    eos_on_output = not OPTIONS.no_eos_on_output
    data = []
    for ex in raw:
        kwargs = {}
        x_inds, y_inds = utils.sentence_pairs_to_indices(in_vocabulary, out_vocabulary, ex, eos_on_output)
        # print "x_inds: {}".format(x_inds)
        # print "y_inds: {}".format(y_inds)

        if OPTIONS.reverse_input:
            raise Exception("This option is not currently supported.")
            x_inds = x_inds[::-1]
        data.append((x_inds, y_inds, kwargs))

    print 'input vocab size = %s, output vocab size = %s' % (in_vocabulary.size(), out_vocabulary.size())

    return data


def preprocess_data_for_eval(in_vocabulary, out_vocabulary, raw):
    eos_on_output = not OPTIONS.no_eos_on_output
    data = []
    for ex in raw:
        kwargs = {}
        pairs = utils.sentence_pairs_to_indices_for_eval(in_vocabulary, out_vocabulary, ex, eos_on_output)

        if OPTIONS.reverse_input:
            raise Exception("This option is not currently supported.")

        data.append((pairs, kwargs))
    return data


def get_continuous_spec(in_vocabulary, out_vocabulary):
    constructor = CONTINUOUS_MODELS[OPTIONS.continuous_spec]
    return constructor(in_vocabulary, out_vocabulary, OPTIONS.hidden_size)


def get_model(spec):
    constructor = MODELS[OPTIONS.model]

    if OPTIONS.float32:
        model = constructor(spec, float_type=numpy.float32)
    else:
        model = constructor(spec)
    return model


def print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                           x_len_list, y_len_list):
    # Overall metrics
    num_examples = len(is_correct_list)
    num_correct = sum(is_correct_list)
    num_tokens_correct = sum(tokens_correct_list)
    num_tokens = sum(y_len_list)
    seq_accuracy = float(num_correct) / num_examples
    token_accuracy = float(num_tokens_correct) / num_tokens

    # Per-length metrics
    num_correct_per_len = collections.defaultdict(int)
    tokens_correct_per_len = collections.defaultdict(int)
    num_per_len = collections.defaultdict(int)
    tokens_per_len = collections.defaultdict(int)
    for is_correct, tokens_correct, x_len, y_len in itertools.izip(
            is_correct_list, tokens_correct_list, x_len_list, y_len_list):
        num_correct_per_len[x_len] += is_correct
        tokens_correct_per_len[x_len] += tokens_correct
        num_per_len[x_len] += 1
        tokens_per_len[x_len] += y_len

    # Print sequence-level accuracy
    logstats.add(name, 'sentence', {
        'correct': num_correct,
        'total': num_examples,
        'accuracy': seq_accuracy,
    })
    print 'Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy)
    # for i in sorted(num_correct_per_len):
    #  cur_num_correct = num_correct_per_len[i]
    #  cur_num_examples = num_per_len[i]
    #  cur_accuracy = float(cur_num_correct) / cur_num_examples
    #  print '  input length = %d: %d/%d = %g correct' % (
    #      i - 1, cur_num_correct, cur_num_examples, cur_accuracy)

    # Print token-level accuracy
    logstats.add(name, 'token', {
        'correct': num_tokens_correct,
        'total': num_tokens,
        'accuracy': token_accuracy,
    })
    print 'Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy)
    # for i in sorted(tokens_correct_per_len):
    #  cur_num_tokens_correct = tokens_correct_per_len[i]
    #  cur_num_tokens = tokens_per_len[i]
    #  cur_accuracy = float(cur_num_tokens_correct) / cur_num_tokens
    #  print '  input length = %d: %d/%d = %g correct' % (
    #      i - 1, cur_num_tokens_correct, cur_num_tokens, cur_accuracy)


def decode(model, x_inds):
    if OPTIONS.beam_size == 0:
        return model.decode_greedy(x_inds, max_len=(2 * len(x_inds) + 50))
    else:
        return model.decode_beam(x_inds, beam_size=OPTIONS.beam_size)


def evaluate(name, model, in_vocabulary, out_vocabulary, raw_data, fout):
    """Evaluate the model.


    Supports dataset mapping x to multiple y.  If so, it treats
    any of those y as acceptable answers.
    """
    print '### evaluate(%s)' % name
    # is_correct_list = []
    # tokens_correct_list = []
    # x_len_list = []
    # y_len_list = []
    #
    # all_x_list = []
    # all_metadata_list = []
    # all_x_set = set()
    # x_to_inds = {}
    # x_to_all_y = collections.defaultdict(list)
    # x_to_kwargs = {}

    if fout:
        eval_info = []

    total_nll = 0.0
    for ex in raw_data:
        x_raw, y_raw = sample_candidates.get_raw_token_sequence(ex["states"][0], ex["agent"])
        ex_eval = {"x_raw":x_raw, "y_raw":y_raw, "candidates":[]}
        already_sampled_x = []
        already_sampled_y = []
        ex_objective = 0.0
        max_tries = 5
        ex_probs = []
        for i in xrange(0, OPTIONS.num_samples):
            x, y = sample_candidates.sample_dialogue(ex)
            candidate_eval = {"candidate_x":x, "candidate_y":y}
            tries = 0
            while x in already_sampled_x and y in already_sampled_y and tries <= max_tries:
                # todo find a better way to find out if there are no more possible candidates -
                # probably not important right now for sample size = 1
                x, y = sample_candidates.sample_dialogue(ex)

            if x in already_sampled_x and y in already_sampled_y:
                # give up
                break

            # print "input seq: ", x
            # print "output seq", y

            assert len(x) == len(y)
            already_sampled_x.append(x)
            already_sampled_y.append(y)
            pairs = zip(x, y)

            x_inds, y_inds = utils.sentence_pairs_to_indices(in_vocabulary,
                                                             out_vocabulary,
                                                             pairs,
                                                             eos_on_output=True)

            p_y_seq, _ = model.get_objective_and_gradients(x_inds, y_inds)
            ex_probs.append(p_y_seq)

            ex_eval["candidates"].append(candidate_eval)

        ex_objective += numpy.sum(ex_probs)
        ex_objective = -numpy.log(ex_objective)

        ex_eval["nll"] = ex_objective
        total_nll += ex_objective

    if fout:
        json.dump(eval_info, fout, indent=4, sort_keys=True)

    return total_nll


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
        if not OPTIONS.save_params: OPTIONS.save_params = os.path.join(OPTIONS.out_dir, 'params')
        if not OPTIONS.stats_file: OPTIONS.stats_file = os.path.join(OPTIONS.out_dir, 'stats.json')
        if not OPTIONS.train_eval_file: OPTIONS.train_eval_file = os.path.join(OPTIONS.out_dir, 'train_eval.json')
        if not OPTIONS.dev_eval_file: OPTIONS.dev_eval_file = os.path.join(OPTIONS.out_dir, 'dev_eval.json')
    if OPTIONS.data_prefix:
        if not OPTIONS.train_data: OPTIONS.train_data = OPTIONS.data_prefix + '.train'
        if not OPTIONS.dev_data: OPTIONS.dev_data = OPTIONS.data_prefix + '.dev'

    logstats.init(OPTIONS.stats_file)

    # Read data
    if OPTIONS.train_data:
        train_raw = load_dataset('train', OPTIONS.train_data)
        # train_sentences = get_sentences_from_raw_data(train_raw)
    if OPTIONS.dev_data:
        dev_raw = load_dataset('dev', OPTIONS.dev_data)
        # dev_sentences = get_sentences_from_raw_data(dev_raw)

    # Create vocab
    if OPTIONS.load_params:
        print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_params
        spec = specutil.load(OPTIONS.load_params)
        in_vocabulary = spec.in_vocabulary
        out_vocabulary = spec.out_vocabulary
    elif OPTIONS.train_data:
        print >> sys.stderr, 'Initializing parameters...'
        in_vocabulary = get_input_vocabulary(train_raw)
        out_vocabulary = get_output_vocabulary(train_raw)
        spec = get_continuous_spec(in_vocabulary, out_vocabulary)
    else:
        raise Exception('Must either provide parameters to load or training data.')

    def evaluate_after_epoch(it):
        # Evaluate on both training and dev data
        if OPTIONS.train_data:
            eval_fh = open_if_specified(OPTIONS.train_eval_file, "w")
            train_nll = evaluate('train', model, in_vocabulary, out_vocabulary, train_raw, eval_fh)
            close_if_specified(eval_fh)
        if OPTIONS.dev_data:
            eval_fh = open_if_specified(OPTIONS.dev_eval_file, "w")
            dev_nll = evaluate('dev', dev_model, in_vocabulary, out_vocabulary, dev_raw, eval_fh)
            logstats.add("dev", "iteration", it, "total_nll", dev_nll)
            close_if_specified(eval_fh)

    # Set up models
    model = get_model(spec)
    model.add_listener(evaluate_after_epoch)
    dev_model = update_model(model, dev_raw)

    # Preprocess data
    # if OPTIONS.train_data:
    #   train_data = preprocess_data(in_vocabulary, out_vocabulary, train_raw)
    # train_data_for_eval = preprocess_data_for_eval(in_vocabulary, out_vocabulary, train_raw)
    # if OPTIONS.dev_data:
    #   dev_data = preprocess_data(dev_model.in_vocabulary,
    #                              dev_model.out_vocabulary, dev_raw)
    #   dev_data_for_eval = preprocess_data_for_eval(dev_model.in_vocabulary,
    #                                                dev_model.out_vocabulary, dev_raw)


    # Train!
    if OPTIONS.train_data:
        model.train(train_raw, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate,
                    batch_size=OPTIONS.batch_size, verbose=True, num_samples=OPTIONS.num_samples)

    if OPTIONS.save_params:
        print >> sys.stderr, 'Saving parameters...'
        spec.save(OPTIONS.save_params)


def main():
    _parse_args()
    run()


if __name__ == '__main__':
    main()
