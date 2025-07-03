import jax
import jax.numpy as jnp
from resno.resno_model import RESNO
from resno.dataloaders import get_rde_pipelines
import equinox as eqx
import optax
from resno.config import ExperimentConfig
from nvidia.dali.plugin.jax.iterator import DALIGenericIterator
from tqdm import tqdm
from quicksig import get_signature_dim, get_log_signature_dim
from typing import Callable
from data.rdes import save_rde_and_driver_paths


@jax.jit
def _all_step_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    # predictions: (batch_size, num_steps * dim) - model output (flattened)
    # targets: (batch_size, num_steps, dim) - solution path
    # Reshape predictions to match targets
    targets = targets.reshape(targets.shape[0], -1)

    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions (shape: {predictions.shape}) and targets (shape: {targets.shape}) must have the same shape.")
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"Predictions (shape: {predictions.shape}) and targets (shape: {targets.shape}) must have 2 dimensions.")
    return jnp.mean((predictions - targets) ** 2)


@jax.jit
def _final_step_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    # predictions: (batch_size, num_steps * dim) - model output (flattened)
    # targets: (batch_size, num_steps, dim) - solution path
    # Reshape predictions and extract final step
    batch_size = predictions.shape[0]
    num_steps = targets.shape[1]
    dim = targets.shape[2]
    predictions_reshaped = predictions.reshape(batch_size, num_steps, dim)
    predictions_final = predictions_reshaped[:, -1, :]  # (batch_size, dim)
    targets_final = targets[:, -1, :]  # (batch_size, dim)
    return jnp.mean((predictions_final - targets_final) ** 2)


@eqx.filter_jit
@eqx.filter_value_and_grad
def _compute_loss(
    model: RESNO,
    driver: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable[[jax.Array, jax.Array], jnp.ndarray],
) -> jnp.ndarray:
    y_pred = jax.vmap(model)(driver.reshape(driver.shape[0], -1))  # Shape: (batch_size, out_path_dim)
    loss = loss_fn(y_pred, y)
    return loss


@eqx.filter_jit
def train_step(
    model: RESNO,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],
    loss_fn: Callable[[jax.Array, jax.Array], jnp.ndarray],
) -> tuple[jnp.ndarray, RESNO, optax.OptState]:

    if batch["driving_path"].ndim != 3:
        raise ValueError(f"Driving path must have 3 dimensions (batch_size, num_steps, dim). Got {batch['driving_path'].ndim} with shape {batch['driving_path'].shape}")
    if batch["solution_path"].ndim != 3:
        raise ValueError(f"Solution path must have 3 dimensions (batch_size, num_steps, dim). Got {batch['solution_path'].ndim} with shape {batch['solution_path'].shape}")
    if batch["driving_path"].shape != batch["solution_path"].shape:
        raise ValueError(f"Driving path and solution path must be the same size. Got {batch['driving_path'].shape} and {batch['solution_path'].shape}")

    loss, grads = _compute_loss(model, batch["driving_path"], batch["solution_path"], loss_fn)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)

    return loss, new_model, new_opt_state


@eqx.filter_jit
def eval_step(
    model: RESNO,
    batch: dict[str, jax.Array],
    loss_fn: Callable[[jax.Array, jax.Array], jnp.ndarray],
) -> jnp.ndarray:
    y_pred = jax.vmap(model)(batch["driving_path"].reshape(batch["driving_path"].shape[0], -1))  # Shape: (batch_size, out_path_dim)
    return loss_fn(y_pred, batch["solution_path"])


def main() -> None:
    config = ExperimentConfig.from_toml("configs/config.toml")

    # If the input is a rough path, we must size the model input to the signature dimension
    if config.model_config.use_rough_paths:
        if config.dataset_config.signature_type == "signature":
            model_in_dim = get_signature_dim(config.dataset_config.signature_depth, config.dataset_config.dim)
        else:
            model_in_dim = get_log_signature_dim(config.dataset_config.signature_depth, config.dataset_config.dim)
    # If the input is a smooth path, we must size the model input to the number of steps * dimension (neural operator style)
    # Note: The data has num_steps + 1 points (from 0 to 1 inclusive)
    else:
        model_in_dim = (config.dataset_config.num_steps + 1) * config.dataset_config.dim

    model = RESNO(
        in_sig_dim=model_in_dim,
        hidden_sig_dim=config.dataset_config.dim,
        out_path_dim=(config.dataset_config.num_steps + 1) * config.dataset_config.dim,  # Output full time series
        num_resno_blocks=config.model_config.num_resno_blocks,
        key=jax.random.PRNGKey(config.data_config.seed),
    )

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Generate and save the drivers and RDEs
    rde_specific_params = config.rde_config.__dict__  # Convert the RDE config to a dictionary

    save_rde_and_driver_paths(
        jax.random.PRNGKey(config.data_config.seed),
        config.dataset_config.driver,
        config.dataset_config.rde,
        config.dataset_config.hurst,
        config.dataset_config.signature_type,
        config.dataset_config.signature_depth,
        config.dataset_config.num_steps,
        config.dataset_config.num_paths,
        dim=config.dataset_config.dim,
        train_percentage=config.dataset_config.train_percentage,
        val_percentage=config.dataset_config.val_percentage,
        test_percentage=config.dataset_config.test_percentage,
        **rde_specific_params,
    )

    # Load the drviers and RDEs generated above
    train_iterator, val_iterator, test_iterator = get_rde_pipelines(
        config.data_config.batch_size,
        config.model_config.use_rough_paths,
        config.dataset_config.rde,
        config.data_config.seed,
    )

    # Choose which loss function to use
    loss_fn = _all_step_mse  # or _final_step_mse depending on what you want to predict

    epoch_pbar = tqdm(range(config.data_config.epochs), desc="Epochs", leave=False, position=0)

    for epoch in epoch_pbar:

        # Training loop
        train_losses = []
        batch_pbar = tqdm(train_iterator, desc="Training batches", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_pbar):
            train_loss, model, opt_state = train_step(model, opt_state, optimizer, batch, loss_fn)
            train_losses.append(train_loss)
            batch_pbar.set_postfix({"train_loss": f"{train_loss:.4f}"})

        # Validation loop
        val_losses = []
        batch_pbar = tqdm(val_iterator, desc="Validation batches", leave=False, position=1)
        for batch_idx, batch in enumerate(batch_pbar):
            val_loss = eval_step(model, batch, loss_fn)
            val_losses.append(val_loss)
            batch_pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})

        # Test loop
        avg_train_loss = jnp.mean(jnp.array(train_losses))
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        epoch_pbar.set_postfix({"train_loss": f"{avg_train_loss:.4f}", "val_loss": f"{avg_val_loss:.4f}"})

    test_losses = []
    test_batch_pbar = tqdm(test_iterator, desc="Test batches", leave=False, position=1)
    for batch_idx, batch in enumerate(test_batch_pbar):
        test_loss = eval_step(model, batch, loss_fn)
        test_losses.append(test_loss)
        test_batch_pbar.set_description(f"Test batch {batch_idx} loss: {test_loss:.6f}")
    avg_test_loss = jnp.mean(jnp.array(test_losses))
    print(f"Test loss: {avg_test_loss:.6f}")


if __name__ == "__main__":
    main()
