"""An LSTM layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer


class LSTMLayer(RNNLayer):
    """An LSTM layer.

    Parameter names follow convention in Richard Socher's CS224D slides.
    """

    def create_vars(self, create_init_state, create_output_layer):
        # Initial state
        # The hidden state must store both c_t, the memory cell,
        # and h_t, what we normally call the hidden state
        if create_init_state:
            self.h0 = theano.shared(
                name='h0',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, 2 * self.nh).astype(theano.config.floatX))
            init_state_params = [self.h0]
        else:
            init_state_params = []

        # Recurrent layer
        self.wi = theano.shared(
            name='wi',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
        self.ui = theano.shared(
            name='ui',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
        self.wf = theano.shared(
            name='wf',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
        self.uf = theano.shared(
            name='uf',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
        self.wo = theano.shared(
            name='wo',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
        self.uo = theano.shared(
            name='uo',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
        self.wc = theano.shared(
            name='wc',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
        self.uc = theano.shared(
            name='uc',
            value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
        recurrence_params = [
            self.wi, self.ui, self.wf, self.uf,
            self.wo, self.uo, self.wc, self.uc,
        ]

        # Output layer
        if create_output_layer:
            self.w_out = theano.shared(
                name='w_out',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (2 * self.nh, self.nw)).astype(theano.config.floatX))
            output_params = [self.w_out]
        else:
            output_params = []

        # Params
        self.params = init_state_params + recurrence_params + output_params

    def unpack(self, hidden_state):
        c_t = hidden_state[0:self.nh]
        h_t = hidden_state[self.nh:]
        return (c_t, h_t)

    def pack(self, c_t, h_t):
        return T.concatenate([c_t, h_t])

    def get_init_state(self):
        return self.h0

    def __getstate__(self):
        print "calling get_params from LSTM layer to pickle"
        if hasattr(self, 'h0'):
            has_init = True
            h_0 = numpy.asarray(self.h0.get_value())
        else:
            has_init = False
            h_0 = None
        wi = numpy.asarray(self.wi.get_value())
        ui = numpy.asarray(self.ui.get_value())
        wf = numpy.asarray(self.wf.get_value())
        uf = numpy.asarray(self.uf.get_value())
        wo = numpy.asarray(self.wo.get_value())
        uo = numpy.asarray(self.uo.get_value())
        wc = numpy.asarray(self.wc.get_value())
        uc = numpy.asarray(self.uc.get_value())

        if hasattr(self, 'w_out'):
            has_out = True
            w_out = numpy.asarray(self.w_out.get_value())
        else:
            w_out = None
            has_out = False

        return has_init, h_0, wi, ui, wf, uf, wo, uo, wc, uc, has_out, w_out

    def __setstate__(self, state):
        print "calling get_params from LSTM layer to unpickle"
        has_init, h_0, wi, ui, wf, uf, wo, uo, wc, uc, has_out, w_out = state
        if has_init:
            self.h0 = theano.shared(
                name='h0',
                value=h_0.astype(theano.config.floatX))
            init_state_params = [self.h0]
        else:
            init_state_params = []
        self.wi = theano.shared(
            name='wi',
            value=wi.astype(theano.config.floatX))
        self.ui = theano.shared(
            name='ui',
            value=ui.astype(theano.config.floatX))
        self.wf = theano.shared(
            name='wf',
            value=wf.astype(theano.config.floatX))
        self.uf = theano.shared(
            name='uf',
            value=uf.astype(theano.config.floatX))
        self.wo = theano.shared(
            name='wo',
            value=wo.astype(theano.config.floatX))
        self.uo = theano.shared(
            name='uo',
            value=uo.astype(theano.config.floatX))
        self.wc = theano.shared(
            name='wc',
            value=wc.astype(theano.config.floatX))
        self.uc = theano.shared(
            name='uc',
            value=uc.astype(theano.config.floatX))

        recurrence_params = [
            self.wi, self.ui, self.wf, self.uf,
            self.wo, self.uo, self.wc, self.uc,
        ]

        if has_out:
            self.w_out = theano.shared(
                name='w_out',
                value=w_out.astype(theano.config.floatX))
            output_params = [self.w_out]
        else:
            output_params = []

        self.params = init_state_params + recurrence_params + output_params

    def step(self, x_t, c_h_prev):
        input_t = self.f_embedding(x_t)
        c_prev, h_prev = self.unpack(c_h_prev)
        i_t = T.nnet.sigmoid(T.dot(input_t, self.wi) + T.dot(h_prev, self.ui))
        f_t = T.nnet.sigmoid(T.dot(input_t, self.wf) + T.dot(h_prev, self.uf))
        o_t = T.nnet.sigmoid(T.dot(input_t, self.wo) + T.dot(h_prev, self.uo))
        c_tilde_t = T.tanh(T.dot(input_t, self.wc) + T.dot(h_prev, self.uc))
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * T.tanh(c_t)
        return self.pack(c_t, h_t)

    def write(self, c_h_t):
        # TODO: try writing only based on h_t, not c_t
        return T.nnet.softmax(T.dot(c_h_t, self.w_out))[0]