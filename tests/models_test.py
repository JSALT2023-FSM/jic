import jax
import jax.numpy as jnp
import numpy.testing as npt

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
    ) == 2731168, str(param_shapes['params']['lattice'])
    assert models.count_number_params(
        param_shapes['params']['classifier']
    ) == 43130880, str(param_shapes['params']['classifier'])


def test_build_lattice():
    model = models.Model(
        lattice=LATTICE_CONFIG.build(), classifier=models.IntentClassifier()
    )
    data_rng, params_rng = jax.random.split(jax.random.PRNGKey(0))
    frames = jax.random.normal(
        data_rng, [2, 16, LATTICE_CONFIG.encoder_embedding_size]
    )
    num_frames = jnp.array([12, 16])
    params = model.init(params_rng, frames, num_frames, is_test=True)
    # Lattice as arc probs.
    full_lattice = model.apply(
        params, frames, num_frames, method=model.build_lattice
    )
    npt.assert_allclose(
        full_lattice.sum(axis=-1), jnp.ones([2, 16, 993]), rtol=1e-5
    )
    # Lattice as arc marginals.
    model.use_arc_marginals = True
    full_lattice = model.apply(
        params, frames, num_frames, method=model.build_lattice
    )
    npt.assert_allclose(
        full_lattice.sum(axis=(-1, -2)), jnp.ones([2, 16]), rtol=1e-5
    )


def test_padded_inputs():
    model = models.Model(
        lattice=LATTICE_CONFIG.build(), classifier=models.IntentClassifier()
    )
    data_rng, params_rng = jax.random.split(jax.random.PRNGKey(0))
    frames = jax.random.normal(
        data_rng, [2, 16, LATTICE_CONFIG.encoder_embedding_size]
    )
    num_frames = jnp.array([12, 16])
    actual, params = model.init_with_output(
        params_rng, frames, num_frames, is_test=True
    )
    expected = []
    for i in range(len(frames)):
        expected.append(
            model.apply(
                params,
                frames[i:i + 1, :num_frames[i]],
                num_frames[i:i + 1],
                is_test=True,
            )
        )
    expected = jnp.concatenate(expected, axis=0)
    npt.assert_allclose(actual, expected, rtol=1e-4)
