"""A generic continuous neural sequence-to-sequence model."""
import collections
import numpy
import os
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import time
import utils
from chat.lib import logstats
from sample_candidates import sample_dialogue

CLIP_THRESH = 3.0  # Clip gradient if norm is larger than this


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

    def __init__(self, spec, float_type=numpy.float64, listeners=[]):
        """Initialize.

        Args:
          spec: Spec object.
          float_type: Floating point type (default 64-bit/double precision)
        """
        self.spec = spec
        self.in_vocabulary = spec.in_vocabulary
        self.out_vocabulary = spec.out_vocabulary
        self.float_type = float_type
        self.params = spec.get_params()
        #print "NeuralModel(): got all params"
        self.all_shared = spec.get_all_shared()
        #print "NeuralModel(): got all theano shared variables"
        self.listeners = listeners

        #print "NeuralModel(): starting setup"
        self.setup()
        #print "NeuralModel(): setup complete"

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

    def decode_greedy(self, x, max_len=100):
        """Decode x to predict y, greedily."""
        raise NotImplementedError

    def on_train_epoch(self, it):
        """Optional method to do things every epoch."""
        # for p in self.params:
        #   print '%s: %s' % (p.name, p.get_value())
        for listener in self.listeners:
            listener(it)

    def train(self, dataset, eta=0.1, T=10, verbose=False, batch_size=1, num_samples=1):
        #print 'NeuralModel.train()'
        #n batch_size = size for mini batch.  Defaults to SGD.
        for it in range(T):
            logstats.add('train', 'iteration', it)
            t0 = time.time()
            total_nll = 0
            random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                do_update = i + batch_size <= len(dataset)
                cur_examples = dataset[i:(i + batch_size)]
                nll = self._train_batch(cur_examples, eta, do_update=do_update, num_samples=num_samples)
                total_nll += nll
                if verbose:
                    print 'NeuralModel.train(): iter %d, %d/%d examples, nll = %g' % (it, i, len(dataset), nll)
            t1 = time.time()
            logstats.add('train', 'nll', total_nll)
            logstats.add('train', 'time', t1 - t0)
            print 'NeuralModel.train(): iter %d: total nll = %g (%g seconds)' % (
                it, total_nll, t1 - t0)
            self.on_train_epoch(it)

    def _train_batch(self, examples, eta, do_update=True, num_samples=1):
        """Run training given a batch of training examples.

        Returns negative log-likelihood.
        If do_update is False, compute nll but don't do the gradient step.
        """
        objective = 0
        gradients = {}
        for ex in examples:
            # x_inds, y_inds
            # print 'xi: %s' % x_inds
            # print 'yi: %s' % y_inds
            # print 'x: %s' % self.in_vocabulary.indices_to_sentence(x_inds)
            # print 'y: %s' % self.out_vocabulary.indices_to_sentence(y_inds)
            max_tries = 5
            ex_gradients = []
            ex_objective = []
            ex_candidate_probs = [] # stores all p_theta(z_i)
            ex_cond_probs_unnorm = [] # stores p(x|z_i) unnormalized

            # TODO: push all sampling logic into sample_candidates.sample_dialogues(ex, num_samples)
            # todo should we be sampling with replacement? removed that code for now (anushabala 06/01)
            for i in xrange(0, num_samples):
                x, y, cond_prob, norm_cond_prob, unnorm_cond_prob = sample_dialogue(ex)
                candidate_prob = numpy.sum(numpy.log(unnorm_cond_prob))
                ex_cond_probs_unnorm.append(candidate_prob)
                #print "input seq: ", x
                #print "output seq", y

                assert len(x) == len(y)
                pairs = zip(x, y)

                x_inds, y_inds = utils.sentence_pairs_to_indices(self.spec.in_vocabulary,
                                                                 self.spec.out_vocabulary,
                                                                 pairs,
                                                                 eos_on_output=True)

                #print "x_inds: ", x_inds
                #print "y_inds", y_inds
                p_y_seq, cur_gradients = self.get_objective_and_gradients(x_inds, y_inds)

                #print "Raw probabilities per token:", p_y_seq
                #print "Probability of candidate sequence:", p_y_seq
                # ex_objective += p_y_seq

                ex_candidate_probs.append(numpy.sum(numpy.log(p_y_seq)))

                ex_gradients.append(cur_gradients)

            ex_candidate_probs_norm = ex_candidate_probs/numpy.sum(ex_candidate_probs)
            gradients_and_probs = zip(xrange(0, num_samples), ex_gradients)
            for p in self.params:
                #print p
                for idx, candidate_gradients in gradients_and_probs:
                    # weight gradient by probability of candidate
                    #print "candidate sequence probability:", candidate_prob
                    #print "prob norm factor:", norm_factor
                    #print "candidate sequence probability:", candidate_prob
                    candidate_prob = ex_candidate_probs_norm[idx]
                    if p in gradients:
                        gradients[p] += candidate_prob * candidate_gradients[p] / len(examples)
                    else:
                        gradients[p] = candidate_prob * candidate_gradients[p] / len(examples)
            for idx in xrange(0, num_samples):
                ex_objective += ex_candidate_probs_norm[idx] * (ex_candidate_probs[idx] + ex_cond_probs_unnorm[idx] - numpy.log(ex_candidate_probs_norm[idx]))
            # loss w.r.t. one example is log of sum of losses of candidates
            # add to total objective
            objective += ex_objective
            #print gradients.keys()
        if do_update:
            for p in self.params:
                #print p, type(gradients[p]), gradients[p]
                self._perform_sgd_step(p, gradients[p], eta)
        return objective

    """
    def decode_greedy(self, x, max_len=100):
      r, w = self._get_map(x, max_len)
      r_list = list(r)
      w_list = list(w)
      try:
        eos_ind = w_list.index(Vocabulary.END_OF_SENTENCE_INDEX)
      except ValueError:
        eos_ind = len(w_list) - 1
      r_out = r_list[:(eos_ind+1)]
      w_out = w_list[:(eos_ind+1)]
      return r_out, w_out

    def decode_beam(self, x, max_len=100, beam_size=5):
      print 'decode_beam'
      BeamState = collections.namedtuple(
          'BeamState', ['r_seq', 'w_seq', 'h_prev', 'next_read', 'log_p'])
      best_finished_state = None
      max_log_p = float('-Inf')
      beam = []
      # Start with a read
      beam.append([BeamState([x[0]], [-1], self.spec.h0.get_value(), 1, 0)])
      for i in range(1, max_len):
        candidates = []
        for state in beam[i-1]:
          if state.w_seq[-1] == Vocabulary.END_OF_SENTENCE_INDEX:
            if state.log_p > max_log_p:
              max_log_p = state.log_p
              best_finished_state = state
            continue
          if state.log_p < max_log_p: continue  # Prune here
          h_t, p_r, p_dist_w = self._step_forward(
              state.r_seq[-1], state.w_seq[-1], state.h_prev)
          if state.next_read < len(x):
            read_state = BeamState(
                state.r_seq + [x[state.next_read]], state.w_seq + [-1], h_t,
                state.next_read + 1, state.log_p + numpy.log(p_r))
            candidates.append(read_state)
          else:
            p_r = 0  # Force write
          if p_r < 1:
            write_candidates = sorted(enumerate(p_dist_w), key=lambda x: x[1],
                                      reverse=True)[:beam_size]
            for index, prob in write_candidates:
              new_state = BeamState(
                  state.r_seq + [-1], state.w_seq + [index], h_t, state.next_read,
                  state.log_p + numpy.log(1 - p_r) + numpy.log(prob))
              candidates.append(new_state)
        beam.append(sorted(
            candidates, key=lambda x: x.log_p, reverse=True)[:beam_size])

      return (best_finished_state.r_seq, best_finished_state.w_seq)
    """

    def get_gradient_seq(self, y_seq):
        """Utility to compute gradient with respect to a sequence."""

        def grad_fn(j, y, *params):
            return T.grad(y[j], self.params, disconnected_inputs='warn')

        results, _ = theano.scan(fn=grad_fn,
                                 sequences=T.arange(y_seq.shape[0]),
                                 non_sequences=[y_seq] + self.params,
                                 strict=True)
        # results[i][j] is gradient of y[j] w.r.t. self.params[i]
        return results

    def _perform_sgd_step(self, param, gradient, eta):
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
        new_value = old_value - eta * gradient
        param.set_value(new_value)
