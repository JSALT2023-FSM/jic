import jax
import jax.numpy as jnp

from jic import models

LATTICE_CONFIG = models.RecognitionLatticeConfig(
    encoder_embedding_size=512,
    context_size=2,
    vocab_size=32,
    hidden_size=640,
    rnn_size=640,
    rnn_embedding_size=128,
    locally_normalize=True
)


def test_num_params():
    model = models.Model(
        lattice=LATTICE_CONFIG.build(), classifier=models.IntentClassifier()
    )
    frames = jnp.zeros([2, 16, LATTICE_CONFIG.encoder_embedding_size])
    num_frames = jnp.array([12, 16])
    param_shapes = jax.eval_shape(
        lambda: model.
        init(jax.random.PRNGKey(0), frames, num_frames, is_test=True)
    )
    assert sorted(param_shapes['params'].keys()) == ['classifier', 'lattice']
    assert models.count_number_params(
        param_shapes['params']['lattice']
    ) == 2731168
    assert models.count_number_params(
        param_shapes['params']['classifier']
    ) == 62545477
