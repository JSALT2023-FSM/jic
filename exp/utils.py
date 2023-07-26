import pandas as pd
import jax
import jax.numpy as jnp

def get_scenario_action(semantics) : 
    scenario = semantics.split("'")[3]
    action = semantics.split("'")[7]
    intent = scenario+"--"+action
    return scenario, action, intent


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


