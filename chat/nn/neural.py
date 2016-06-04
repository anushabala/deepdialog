"""A generic continuous neural sequence-to-sequence model."""
import collections
import numpy
import os
import random
import sys
import math
import time
import theano
from theano.ifelse import ifelse
from theano import tensor as T
from heapq import nlargest
from collections import defaultdict
from chat.lib import logstats, sample_utils
from chat.modeling import data_utils, dialogue_tracker, recurrent_box
from chat.modeling.lexicon import Lexicon, load_scenarios
from chat.modeling.recurrent_box import RecurrentBox

CLIP_THRESH = 3.0  # Clip gradient if norm is larger than this

NUM_CANDIDATES = 10

class NeuralBox(RecurrentBox):
    def __init__(self, model):
        self.model = model
        # Hidden state
        self.h_t = self.model.spec.get_init_state().eval()

    def generate(self):
        # Distribution over next tokens
        write_dist = self.model._decoder_write(self.h_t)
        # Take the largest ones
        indices = nlargest(NUM_CANDIDATES, range(len(write_dist)), key=lambda i: write_dist[i])
        candidates = [(self.model.out_vocabulary.get_word(i), write_dist[i]) for i in indices]
        #print 'GENERATE', candidates
        return candidates

    def observe(self, token, write):
        #print 'OBSERVE', token, write
        if write:
            # Observe something we just wrote
            y_t = self.model.out_vocabulary.get_index(token)
            self.h_t = self.model._decoder_step(y_t, self.h_t)
        else:
            # Observe something we read
            x_t = self.model.in_vocabulary.get_index(token)
            self.h_t = self.model._encode(numpy.array([x_t]), self.h_t)

class NeuralModel(object):
    """A generic continuous neural sequence-to-sequence model.

    Implementing classes must implement the following functions:
      - self.setup(): set up the model.
      - self.get_objective_and_gradients(x, y): Get objective and gradients.
      - self.decode_greedy(x, max_len=100): Do a greedy decoding of x, predict y.

    Convention used by this class:
      nh: dimension of hidden layer
      nw: number of words in the vocabulary
      de: dimension of word embeddings
    """

    def __init__(self, args, spec, float_type=numpy.float64, listeners=[]):
        """Initialize.

        Args:
          spec: Spec object.
          float_type: Floating point type (default 64-bit/double precision)
        """
        self.args = args
        self.spec = spec
        self.in_vocabulary = spec.in_vocabulary
        self.out_vocabulary = spec.out_vocabulary
        self.float_type = float_type
        self.params = spec.get_params()
        self.all_shared = spec.get_all_shared()
        self.listeners = listeners

        # Create lexicon (for DialogueTracker)
        self.scenarios = load_scenarios(args.scenarios)
        self.lexicon = Lexicon(self.scenarios)

        print "NeuralModel(): starting setup"
        self.setup()
        print "NeuralModel(): setup complete"

    def setup(self, test_only=False):
        """Do all necessary setup (e.g. compile theano functions)."""
        raise NotImplementedError

    def add_listener(self, listener):
        self.listeners.append(listener)

    def get_objective_and_gradients(self, x, y):
        """Get objective and gradients.

        Returns: tuple (objective, gradients) where
          objective: the current objective value
          gradients: map from parameter to gradient
        """
        raise NotImplementedError

    def on_train_epoch(self, it):
        """Optional method to do things every epoch."""
        # for p in self.params:
        #   print '%s: %s' % (p.name, p.get_value())
        for listener in self.listeners:
            listener(it)
        
    def train_loop(self, train_examples, dev_examples):
        print 'NeuralModel.train_loop()'
        best_dev_objective = 1e100
        for it in range(self.args.num_epochs):
            logstats.add('iteration', it)
            train_summary_map = self.do_iter('train', it, train_examples)
            dev_summary_map = self.do_iter('dev', it, dev_examples)
            self.on_train_epoch(it)
            dev_objective = dev_summary_map['objective']['mean']
            # Save parameters
            if self.args.save_params:
                self.spec.save(self.args.save_params + '.%d' % it)
            # Keep track of the best model so far
            if dev_objective < best_dev_objective:
                best_dev_objective = dev_objective
                logstats.add('best_iteration', it)
                print 'Best dev accuracy so far at iteration %s: %s' % (it, dev_objective)
                if self.args.save_params:
                    self.spec.save(self.args.save_params + '.best')

    def do_iter(self, mode, it, examples):
        if len(examples) == 0:
            return
        is_train = (mode == 'train')
        summary_map = defaultdict(dict)
        if is_train:
            examples = list(examples)
            random.shuffle(examples)
        eval_period = self.args.train_eval_period if is_train else self.args.dev_eval_period

        # Go over all the examples in batches
        for i in range(0, len(examples), self.args.batch_size):
            # Get a batch
            batch_examples = examples[i:(i + self.args.batch_size)]
            # Do stuff
            batch_summary_map = self._do_batch(batch_examples, eval_period, do_update=is_train)
            # Update/print out stats
            logstats.update_summary_map(summary_map, batch_summary_map)
            logstats.add(mode, summary_map)
            print 'NeuralModel.do_iter(%s): iter %d, %d/%d examples, %s' % (\
                mode, it, i, len(examples),
                logstats.summary_map_to_str(summary_map)
            )
        print 'NeuralModel.do_iter(%s): iter %d DONE, %s' % (\
            mode, it,
            logstats.summary_map_to_str(summary_map)
        )
        return summary_map

    def _do_batch(self, examples, eval_period, do_update=True):
        """
        Run training given a batch of training examples.
        Returns objective function and bleu.
        eval_period: call evaluate every this many examples.
        If do_update is False, compute objective but don't do the gradient step.
        """
        t0 = time.time()
        summary_map = defaultdict(dict)
        gradients = {}
        for ex in examples:  # Go over all examples in the mini-batch
            ex_scores = []
            ex_gradients = []
            ex_log_weights = []

            # Evaluate on end-to-end metric
            if hash(ex['scenario_id']) % eval_period == 0:
                scenario = self.scenarios[ex['scenario_id']]
                ex_summary_map = dialogue_tracker.evaluate_example(
                    scenario, self.lexicon, self.args, ex, lambda : NeuralBox(self)
                )
                logstats.update_summary_map(summary_map, ex_summary_map)

            # Sample num_samples trajectories.
            #samples = set()
            for i in range(self.args.num_samples):
                # Sample a trajectory q(x, y | observations) \propto p(observations | x, y)
                # log_weight = \log p(observations | x, y)
                messages, log_weight = data_utils.sample_trajectory(ex['states'])
                sequences = data_utils.messages_to_sequences(ex['agent'], messages)
                assert len(sequences) % 2 == 0
                #samples.add(str(sequences))
                print 'UPDATE TOWARDS', i, sequences

                # Convert to indices:
                #   sequences = [partner, agent, ..., partner, agent]
                #   x_inds = [partner, -1,    ..., partner, -1]
                #   y_inds = [-1,      agent, ..., -1,       agent]
                x_tokens = [token if i % 2 == 0 else None for i, seq in enumerate(sequences) for token in seq] # Partner
                y_tokens = [token if i % 2 == 1 else None for i, seq in enumerate(sequences) for token in seq] # Agent
                x_inds = self.spec.in_vocabulary.words_to_indices(x_tokens)
                y_inds = self.spec.out_vocabulary.words_to_indices(y_tokens)

                # Compute gradients
                sample_objective, sample_gradients = self.get_objective_and_gradients(x_inds, y_inds)
                ex_scores.append(-sample_objective)  # log(p(y | x))
                ex_gradients.append(sample_gradients)
                ex_log_weights.append(log_weight)  # p(observations | x, y)

            # Get a distribution over the samples r(y) \propto p(y | x)
            ex_probs = sample_utils.exp_normalize_weights(ex_scores)
            #print 'PROBS', ex_probs
            #print '%d/%d unique' % (len(samples), self.args.num_samples)

            # Aggregate gradients
            if do_update:
                for i in range(self.args.num_samples):
                    for key in self.params:
                        # Scale gradient by size of mini-batch
                        g = ex_gradients[i][key] * ex_probs[i] / len(examples)
                        if key in gradients:
                            gradients[key] += g
                        else:
                            gradients[key] = g

            # Compute lower bound on the log-likelihood:
            # r(y) \log [p(y | x) p(observations | x, y)] - \log r(y)
            ex_likelihood = 0.0
            ex_entropy = 0.0
            for i in range(self.args.num_samples):
                ex_likelihood += ex_probs[i] * (ex_scores[i] + ex_log_weights[i])
                ex_entropy += ex_probs[i] * -math.log(ex_probs[i])
            ex_objective = -(ex_likelihood + ex_entropy)
            logstats.update_summary(summary_map['likelihood'], ex_likelihood)
            logstats.update_summary(summary_map['entropy'], ex_entropy)
            logstats.update_summary(summary_map['objective'], ex_objective)

        if do_update:
            for p in self.params:
                self._perform_sgd_step(p, gradients[p])

        t1 = time.time()
        logstats.update_summary(summary_map['time'], t1 - t0)

        return summary_map

    def _perform_sgd_step(self, param, gradient):
        """Do a gradient descent step."""
        # print param.name
        # print param.get_value()
        # print gradient
        old_value = param.get_value()
        grad_norm = numpy.sqrt(numpy.sum(gradient ** 2))
        if grad_norm >= CLIP_THRESH:
            gradient = gradient * CLIP_THRESH / grad_norm
            new_norm = numpy.sqrt(numpy.sum(gradient ** 2))
            # print 'Clipped norm of %s from %g to %g' % (param, grad_norm, new_norm)
        new_value = old_value - self.args.learning_rate * gradient
        param.set_value(new_value)
