"""Modelling code, e.g. flax modules."""

import dataclasses

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import last

from jic import datasets


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


class BiLSTM(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, frames, num_frames):
        return nn.Bidirectional(
            nn.RNN(nn.OptimizedLSTMCell(self.hidden_size, name='forward')),
            nn.RNN(nn.OptimizedLSTMCell(self.hidden_size, name='backward'))
        )(frames, seq_lengths=num_frames)


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

    @nn.compact
    def __call__(self, full_lattice, num_frames, *, is_test):
        regrouped_lattice = nn.Dense(self.dense_size,
                                     name='regrouping_head')(full_lattice)
        # Encoder stack
        encoded_lattice = BiLSTM(self.hidden_size, name='downstream_encoder_0'
                                )(regrouped_lattice, num_frames)
        encoded_lattice = nn.Dropout(self.dropout_rate, name='encoder_dropout'
                                    )(encoded_lattice, deterministic=is_test)
        encoded_lattice = BiLSTM(self.hidden_size, name='downstream_encoder_1'
                                )(encoded_lattice, num_frames)
        # Attention pooling
        mask = (
            jnp.arange(encoded_lattice.shape[-2])
            < jnp.expand_dims(num_frames, axis=-1)
        )
        attention = KeysOnlyMlpAttention(
            2 * self.hidden_size, name='keys_only_mlp_attention'
        )(encoded_lattice, mask)
        context = jnp.einsum('BT,BTH->BH', attention, encoded_lattice)
        context = nn.Dropout(self.dropout_rate, name='context_dropout'
                            )(context, deterministic=is_test)
        context = nn.Dense(2 * self.hidden_size, name='post_attention')(context)
        # Classification
        intents = nn.Dense(self.num_intents, name='intents_head')(context)
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
