"""A basic encoder-decoder model."""
import itertools
import numpy
import theano
from theano import tensor as T
from theano.ifelse import ifelse

from neural import NeuralModel
from vocabulary import Vocabulary

class EncoderDecoderModel(NeuralModel):
  """An encoder-decoder RNN model."""
  def setup(self):
    self.setup_encoder()
    self.setup_decoder_step()
    self.setup_decoder_write()
    self.setup_backprop()

  def setup_encoder(self):
    """Run the encoder.  Used at test time."""
    x = T.lvector('x_for_enc')
    h_init = T.vector('h_init_for_enc')
    def recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)
    results, _ = theano.scan(recurrence,
                             sequences=[x],
                             outputs_info=[h_init],
                             non_sequences=self.spec.get_all_shared())
    h_last = results[-1]
    self._encode = theano.function(inputs=[x, h_init], outputs=h_last)

  def setup_decoder_step(self):
    """Advance the decoder by one step.  Used at test time."""
    y_t = T.lscalar('y_t_for_dec')
    h_prev = T.vector('h_prev_for_dec')
    h_t = self.spec.f_dec(y_t, h_prev)
    self._decoder_step = theano.function(inputs=[y_t, h_prev], outputs=h_t)

  def setup_decoder_write(self):
    """Get the write distribution of the decoder.  Used at test time."""
    h_prev = T.vector('h_prev_for_write')
    write_dist = self.spec.f_write(h_prev)
    self._decoder_write = theano.function(inputs=[h_prev], outputs=write_dist)

  # def setup_backprop(self):
  #   x = T.lvector('x_for_backprop')
  #   y = T.lvector('y_for_backprop')
    
  #   def encoder_recurrence(x_t, h_prev, *params):
  #     return self.spec.f_enc(x_t, h_prev)

  #   def decoder_recurrence(y_t, h_prev, *params):
  #     write_dist = self.spec.f_write(h_prev)
  #     p_y_t = write_dist[y_t]
  #     h_t = self.spec.f_dec(y_t, h_prev)
  #     return (h_t, p_y_t)

  #   def enc_dec_recurrence(x_t, y_t, h_prev, *params):
  #     cond = T.eq(y_t,-1)
  #     enc_result = (encoder_recurrence(x_t, h_prev, params), T.as_tensor_variable(-1.0))
  #     dec_result = decoder_recurrence(y_t, h_prev, params)
  #     return ifelse(cond, enc_result, dec_result)

  #   enc_dec_results, _ = theano.scan(fn=enc_dec_recurrence,
  #                                    sequences=[x,y],
  #                                    outputs_info=[self.spec.get_init_state(),None],
  #                                    non_sequences=self.spec.get_all_shared())

  #   p_y_seq = enc_dec_results[1]
  #   p_y_seq = p_y_seq[(p_y_seq>=0).nonzero()]
  #   log_p_y = T.sum(T.log(p_y_seq))
  #   gradients = T.grad(log_p_y, self.params, disconnected_inputs='ignore')
  #   self._backprop = theano.function(
  #     inputs=[x, y], 
  #     outputs=[p_y_seq, log_p_y] + [-g for g in gradients],
  #     mode=theano.Mode(linker='vm'))

  def setup_backprop(self):
    x = T.lvector('x_for_backprop')
    y = T.lvector('y_for_backprop')
    
    def encoder_recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)

    def decoder_recurrence(y_t, h_prev, *params):
      write_dist = self.spec.f_write(h_prev)
      p_y_t = write_dist[y_t]
      h_t = self.spec.f_dec(y_t, h_prev)
      return (h_t, p_y_t)

    def enc_dec_recurrence(x_t, y_t, h_prev, *params):
      cond = T.eq(y_t,-1)
      enc_h = encoder_recurrence(x_t, h_prev, params)
      dec_h, dec_p_y = decoder_recurrence(y_t, h_prev, params)
      return (ifelse(cond, enc_h, dec_h), ifelse(cond, -dec_p_y, dec_p_y))

    enc_dec_results, _ = theano.scan(fn=enc_dec_recurrence,
                                     sequences=[x,y],
                                     outputs_info=[self.spec.get_init_state(),None],
                                     non_sequences=self.spec.get_all_shared())

    p_y_seq = enc_dec_results[1]
    p_y_seq = p_y_seq[(p_y_seq>=0).nonzero()]
    log_p_y = T.sum(T.log(p_y_seq))
    gradients = T.grad(log_p_y, self.params, disconnected_inputs='ignore')
    self._backprop = theano.function(
      inputs=[x, y], 
      outputs=[p_y_seq, log_p_y] + [-g for g in gradients],
      mode=theano.Mode(linker='vm'))

  def get_objective_and_gradients(self, x, y, **kwargs):
    info = self._backprop(x, y)
    p_y_seq = info[0]
    log_p_y = info[1]
    gradients_list = info[2:]
    objective = -log_p_y
    gradients = dict(itertools.izip(self.params, gradients_list))
    # print 'P(y_i): %s' % p_y_seq
    return (objective, gradients)

  # mkayser: x is now a list of lists of indices!
  def decode_greedy(self, x, max_len=100):
    print "Experimental: decode: {}".format(x)
    #return []

    y_seq = []
    p_y_seq = []  # Should be handy for error analysis

    h_t = self.spec.get_init_state().eval()

    for xi in x:
      h_t = self._encode(xi, h_t)
      for w_index in range(max_len):
        write_dist = self._decoder_write(h_t)
        y_t = numpy.argmax(write_dist)
        p_y_t = write_dist[y_t]
        y_seq.append(y_t)
        p_y_seq.append(p_y_t)
        h_t = self._decoder_step(y_t, h_t)
        if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
          break
    return y_seq


    # h_t = self._encode(x)
    # y_seq = []
    # p_y_seq = []  # Should be handy for error analysis
    # while True:
    #   write_dist = self._decoder_write(h_t)
    #   y_t = numpy.argmax(write_dist)
    #   p_y_t = write_dist[y_t]
    #   y_seq.append(y_t)
    #   p_y_seq.append(p_y_t)
    #   if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
    #     break
    #   h_t = self._decoder_step(y_t, h_t)
    # return y_seq
