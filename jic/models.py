"""Modelling code, e.g. flax modules."""

import dataclasses

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import last

from jic import datasets

# TODO
# 1. Switch to new flax RNN API


@dataclasses.dataclass
class RecognitionLatticeConfig:
    encoder_embedding_size: int
    context_size: int
    vocab_size: int
    hidden_size: int
    rnn_size: int
    rnn_embedding_size: int
    locally_normalize: bool

    def build(self) -> last.RecognitionLattice:
        context = last.contexts.FullNGram(
            # In LAST's convention, blank doesn't count towards vocab_size.
            vocab_size=self.vocab_size - 1,
            context_size=self.context_size,
        )
        alignment = last.alignments.FrameDependent()

        def weight_fn_cacher_factory(context: last.contexts.FullNGram):
            return last.weight_fns.SharedRNNCacher(
                vocab_size=context.vocab_size,
                context_size=context.context_size,
                rnn_size=self.rnn_size,
                rnn_embedding_size=self.rnn_embedding_size,
            )

        def weight_fn_factory(context: last.contexts.FullNGram):
            weight_fn = last.weight_fns.JointWeightFn(
                vocab_size=context.vocab_size, hidden_size=self.hidden_size
            )
            if self.locally_normalize:
                weight_fn = last.weight_fns.LocallyNormalizedWeightFn(weight_fn)
            return weight_fn

        lattice = last.RecognitionLattice(
            context=context,
            alignment=alignment,
            weight_fn_cacher_factory=weight_fn_cacher_factory,
            weight_fn_factory=weight_fn_factory,
        )
        return lattice


class LSTMEncoder(nn.Module):
    """A stack of unidirectional LSTMs."""

    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, xs):
        """Encodes the inputs.

    Args:
      xs: [batch_size, max_num_frames, feature_size] input sequences.

    Returns:
      [batch_size, max_num_frames, hidden_size] output sequences.
    """
        # Flax allows randomness in initializing the hidden state. However LSTM's
        # initial hidden state is deterministic, so we just need a dummy RNG here.
        dummy_rng = jax.random.PRNGKey(0)
        init_carry = nn.OptimizedLSTMCell.initialize_carry(
            rng=dummy_rng, batch_dims=xs.shape[:1], size=self.hidden_size
        )
        # A stack of num_layers LSTMs.
        for _ in range(self.num_layers):
            # nn.scan() when passed a class returns another class like object. Thus
            # nn.scan(nn.OptimizedLSTMCell, ...)() constructs such a new instance of
            # such a "class". Then, we invoke the __call__() method of this object
            # by passing (init_carry, xs).
            #
            # The __call__() method of the nn.scan()-transformed "class" loops through
            # the input whereas the original class takes one step.
            _, xs = nn.scan(
                nn.OptimizedLSTMCell,
                variable_broadcast='params',
                split_rngs={'params': False},
                in_axes=1,
                out_axes=1,
            )()(init_carry, xs)
        return xs


def sequence_mask(lengths, max_length):
    """Computes a boolean mask over sequence positions for each given length.

  Example:
  ```
  sequence_mask([1, 2], 3)
  [[True, False, False],
   [True, True, False]]
  ```

  Args:
    lengths: The length of each sequence. <int>[batch_size]
    max_length: The width of the boolean mask. Must be >= max(lengths).

  Returns:
    A mask with shape: <bool>[batch_size, max_length] indicating which
    positions are valid for each sequence.
  """
    return jnp.arange(max_length)[None] < lengths[:, None]


@jax.vmap
def flip_sequences(inputs, lengths):
    """Flips a sequence of inputs along the time dimension.

  This function can be used to prepare inputs for the reverse direction of a
  bidirectional LSTM. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.

  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```

  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

  Returns:
    An ndarray with the flipped inputs.
  """
    # Note: since this function is vmapped, the code below is effectively for
    # a single example.
    max_length = inputs.shape[0]
    return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class SimpleBiLSTM(nn.Module):
    """A simple bi-directional LSTM."""
    hidden_size: int

    def setup(self):
        self.forward_lstm = LSTMEncoder(self.hidden_size, 1)
        self.backward_lstm = LSTMEncoder(self.hidden_size, 1)

    def __call__(self, embedded_inputs, lengths):
        # Forward LSTM.
        forward_outputs = self.forward_lstm(embedded_inputs)

        # Backward LSTM.
        reversed_inputs = flip_sequences(embedded_inputs, lengths)
        backward_outputs = self.backward_lstm(reversed_inputs)
        backward_outputs = flip_sequences(backward_outputs, lengths)

        # Concatenate the forwardand backward representations.
        outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
        return outputs


class KeysOnlyMlpAttention(nn.Module):
    """Computes MLP-based attention scores based on keys alone, without a query.

  Attention scores are computed by feeding the keys through an MLP. This
  results in a single scalar per key, and for each sequence the attention
  scores are normalized using a softmax so that they sum to 1. Invalid key
  positions are ignored as indicated by the mask. This is also called
  "Bahdanau attention" and was originally proposed in:
  ```
  Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
  Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
  ```

  Attributes:
    hidden_size: The hidden size of the MLP that computes the attention score.
  """
    hidden_size: int

    @nn.compact
    def __call__(self, keys, mask):
        """Applies model to the input keys and mask.

    Args:
      keys: The inputs for which to compute an attention score. Shape:
        <float32>[batch_size, seq_length, embeddings_size].
      mask: A mask that determines which values in `keys` are valid. Only
        values for which the mask is True will get non-zero attention scores.
        <bool>[batch_size, seq_length].

    Returns:
      The normalized attention scores. <float32>[batch_size, seq_length].
    """
        hidden = nn.Dense(self.hidden_size, name='keys', use_bias=False)(keys)
        energy = nn.tanh(hidden)
        scores = nn.Dense(1, name='energy', use_bias=False)(energy)
        scores = scores.squeeze(
            -1
        )  # New shape: <float32>[batch_size, seq_len].
        scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
        scores = nn.softmax(scores, axis=-1)
        print(scores)

        # Captures the scores if 'intermediates' is mutable, otherwise does nothing.
        self.sow('intermediates', 'attention', scores)

        return scores


class IntentClassifier(nn.Module):
    # Size for encoder LSTMs, the output context LSTM, the joint weight function.
    hidden_size: int = 768
    dropout_rate = 0.1
    dense_size = 1024

    num_scenarios: int = len(datasets.SCENARIOS)
    num_actions: int = len(datasets.ACTIONS)
    num_intents: int = len(datasets.INTENTS)

    def setup(self):
        self.Downstream_Encoder = SimpleBiLSTM(self.hidden_size)
        self.second_downstream_Encoder = SimpleBiLSTM(self.hidden_size)
        self.intents_head = nn.Dense(self.num_intents)
        self.keys_only_mlp_attention = KeysOnlyMlpAttention(
            2 * self.hidden_size
        )
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)
        self.regrouping_head = nn.Dense(self.dense_size)
        self.post_attention = nn.Dense(2 * self.hidden_size)

    def __call__(self, full_lattice, num_frames, *, is_test):
        regrouped_lattice = self.regrouping_head(full_lattice)
        encoded_lattice = self.Downstream_Encoder(regrouped_lattice, num_frames)
        encoded_lattice = self.dropout_layer(
            encoded_lattice, deterministic=is_test
        )
        encoded_lattice = self.second_downstream_Encoder(
            encoded_lattice, num_frames
        )
        mask = sequence_mask(num_frames, encoded_lattice.shape[1])
        attention = self.keys_only_mlp_attention(encoded_lattice, mask)
        # Summarize the inputs by taking their weighted sum using attention scores.
        context = jnp.expand_dims(attention, 1) @ encoded_lattice
        context = context.squeeze(
            1
        )  # <float32>[batch_size, encoded_inputs_size]
        context = self.dropout_layer(context, deterministic=is_test)
        context = self.post_attention(context)

        intents = self.intents_head(context)
        return intents


class Model(nn.Module):
    lattice: last.RecognitionLattice
    classifier: IntentClassifier

    def __call__(self, frames, num_frames, *, is_test):
        # TODO: alternative methods of computing full lattice arc weights.
        cache = self.lattice.build_cache()
        blank, lexical = self.lattice.weight_fn(cache, frames)
        full_lattice = jnp.concatenate(
            [jnp.expand_dims(blank, axis=-1), lexical], axis=-1
        )
        # Turn log-probs into probs.
        full_lattice = jnp.exp(full_lattice)
        full_lattice = einops.rearrange(full_lattice, 'B T Q V -> B T (Q V)')
        return self.classifier(full_lattice, num_frames, is_test=is_test)


def count_number_params(params):
    return sum(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: x.size, params)
        )
    )


def compute_accuracies(intents, batch, loss):
    intents_results = intents == batch["intent"]
    return {"intents": jnp.mean(intents_results), "loss": loss}
