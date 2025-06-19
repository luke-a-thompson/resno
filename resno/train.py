import jax
import jax.numpy as jnp
from resno.resno_model import RESNO
import equinox as eqx
import optax


def _compute_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - targets) ** 2)


@eqx.filter_value_and_grad
def _compute_loss(
    model: RESNO,
    curr_logsig: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    y_pred = jax.vmap(model)(curr_logsig)
    return _compute_mse(y_pred, y)


def train_step(
    model: RESNO,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    curr_logsig: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[jnp.ndarray, RESNO, optax.OptState]:
    loss, grads = _compute_loss(model, curr_logsig, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return loss, new_model, new_opt_state


def main(epochs: int, C_AE: float = 1.0, C_e: float = 1) -> None:
    key = jax.random.PRNGKey(seed=0)
    path_key, compressor_key, decompressor_key = jax.random.split(key, 3)

    # Create a batch of 1 path: (batch_size, sequence_length, num_channels)

    # get_log_signature will return (batch_size, num_features)
    low_depth_logsig: jax.Array = get_log_signature(batched_path, depth=2, log_signature_type="lyndon")
    high_depth_logsig: jax.Array = get_log_signature(batched_path, depth=3, log_signature_type="lyndon")
