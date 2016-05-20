"""Specifies a particular instance of an encoder-decoder model."""
from vocabulary import RawVocabulary
from gru import GRULayer
from lstm import LSTMLayer
from spec import Spec
from vanillarnn import VanillaRNNLayer


class EncoderDecoderSpec(Spec):
    """Abstract class for a specification of an encoder-decoder model.

    Concrete subclasses must implement the following method:
    - self.create_layer(vocab, hidden_size): Create an RNN layer.
    """

    def create_vars(self):
        self.encoder = self.create_layer(self.in_vocabulary, self.hidden_size, True)
        self.decoder = self.create_layer(self.out_vocabulary, self.hidden_size, False)

    def get_local_params(self):
        return self.encoder.params + self.decoder.params

    def create_layer(self, vocab, hidden_size, is_encoder, create_vars=True):
        raise NotImplementedError

    def get_init_state(self):
        return self.encoder.get_init_state()

    def f_enc(self, x_t, h_prev):
        """Returns the next hidden state for encoder."""
        return self.encoder.step(x_t, h_prev)

    def f_dec(self, y_t, h_prev):
        """Returns the next hidden state for decoder."""
        return self.decoder.step(y_t, h_prev)

    def f_write(self, h_t):
        """Gives the softmax output distribution."""
        return self.decoder.write(h_t)

    def __getstate__(self):
        dec_state = self.decoder.__getstate__()
        enc_state = self.encoder.__getstate__()
        invoc_state = self.in_vocabulary.__getstate__()
        outvoc_state = self.out_vocabulary.__getstate__()
        return dec_state, enc_state, invoc_state, outvoc_state

    def __setstate__(self, state):
        dec_state, enc_state, invoc_state, outvoc_state = state
        # todo generify this
        inword_list, inemb_size, infloat_type, in_emb_mat = invoc_state
        self.in_vocabulary = RawVocabulary(inword_list, inemb_size, infloat_type)
        self.in_vocabulary.__setstate__(in_emb_mat)

        outword_list, outemb_size, outfloat_type, out_emb_mat = outvoc_state
        self.out_vocabulary = RawVocabulary(outword_list, outemb_size, outfloat_type)
        self.out_vocabulary.__setstate__(out_emb_mat)

        self.encoder = self.create_layer(self.in_vocabulary, self.hidden_size, True, False)
        self.encoder.__setstate__(enc_state)
        self.decoder = self.create_layer(self.out_vocabulary, self.hidden_size, False, False)
        self.decoder.__setstate__(dec_state)


class VanillaEncDecSpec(EncoderDecoderSpec):
    """Encoder-decoder model with vanilla RNN recurrent units."""

    def create_layer(self, vocab, hidden_size, is_encoder, create_vars=True):
        return VanillaRNNLayer(vocab, hidden_size,
                               create_init_state=is_encoder,
                               create_output_layer=not is_encoder)


class GRUEncDecSpec(EncoderDecoderSpec):
    """Encoder-decoder model with GRU recurrent units."""

    def create_layer(self, vocab, hidden_size, is_encoder, create_vars=True):
        return GRULayer(vocab, hidden_size,
                        create_init_state=is_encoder,
                        create_output_layer=not is_encoder)


class LSTMEncDecSpec(EncoderDecoderSpec):
    """Encoder-decoder model with LSTM recurrent units."""

    def create_layer(self, vocab, hidden_size, is_encoder, create_vars=True):
        return LSTMLayer(vocab, hidden_size,
                         create_init_state=is_encoder,
                         create_output_layer=not is_encoder,
                         create_vars=create_vars)
