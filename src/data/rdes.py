"""
This file is used to generate the driving signals and the solutions to the RDEs.

We use the following parlance:
* RDE: The rough differential equation (RDE).
* Driving signal: The signal that is used to drive the RDE.
* Solution: The solution to the RDE.
* Rough path: The log signature tuple (X. 𝕏) that drives the RDE.

"""

from typing import Literal
from diffrax import (
    diffeqsolve,
    ODETerm,
    ControlTerm,
    LinearInterpolation,
    SaveAt,
    Dopri5,
    MultiTerm,
    Solution,
)
from functools import partial
import os
import jax.numpy as jnp
import jax
from jax import Array, random
from data.driver_and_solution_info import RDE, Driver, rde_locations
from data.driving_signals import fbm_davies_harte
from quicksig import get_signature, get_log_signature


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def fractional_ou_process(
    driving_path: jnp.ndarray,
    num_steps: int,
    lam: float,
    sigma: float,
    y0: float,
) -> Solution:
    """
    Solves a fractional Ornstein-Uhlenbeck process given a pre-generated fBM path.

    The SDE is dy(t) = -lam * y(t) * dt + sigma * dB_H(t).

    Args:
        lam: The reversion rate lambda.
        sigma: The volatility.
        num_steps: The number of time steps or equivalently the mesh size to solve the SDE.
        fbm_path: A pre-generated path of fractional Brownian motion, corresponding to `ts`.
                  It must have the same length as `ts` and start at 0.
        y0: The initial value of the process.

    Returns:
        A tuple of (ts, ys) containing the time points and the solution values.
    """
    if driving_path.ndim != 2:
        raise ValueError(f"Driving path must have shape (num_steps + 1, m), got {driving_path.shape}")
    if driving_path.shape[0] != num_steps + 1:
        raise ValueError(f"Driving path must have length {num_steps + 1}, got {driving_path.shape}")

    d = driving_path.shape[1]  # state dimension
    y0_array = jnp.full((d,), y0)  # broadcast scalar to shape (d,)

    timespan = jnp.linspace(0, 1, num_steps + 1)
    control = LinearInterpolation(ts=timespan, ys=driving_path)

    drift = lambda t, y, args: -lam * y
    diffusion = lambda t, y, args: sigma * jnp.ones((d, d))  # or (d, m) if m != d

    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, control))

    solver = Dopri5()
    t0 = timespan[0]
    t1 = timespan[-1]
    discretization = timespan[1] - timespan[0] if len(timespan) > 1 else 0.1
    saveat = SaveAt(ts=timespan)

    sol = diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=discretization,
        y0=y0_array,
        saveat=saveat,
    )

    return sol


def save_train_val_test_split(
    driving_paths: jax.Array,
    driving_rough_paths: jax.Array,
    solutions: jax.Array,
    solutions_rough_paths: jax.Array,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    driver: Driver,
    rde: RDE,
) -> None:
    """
    Split the data into train/val/test sets and save them to appropriate directories.

    Args:
        driving_paths: Array of driving paths with shape (num_paths, num_steps, dim)
        driving_rough_paths: Array of driving rough paths with shape (num_paths, rough_path_dim)
        solutions: Array of RDE solutions with shape (num_paths, num_steps, dim)
        solutions_rough_paths: Array of solution rough paths with shape (num_paths, rough_path_dim)
        train_percentage: Percentage of data for training (0.0 to 1.0)
        val_percentage: Percentage of data for validation (0.0 to 1.0)
        test_percentage: Percentage of data for testing (0.0 to 1.0)
        driver: The driver type (e.g., Driver.fBM)
        rde: The RDE type (e.g., RDE.fOU)
    """
    if train_percentage + val_percentage + test_percentage != 1.0:
        raise ValueError(
            f"Train, val, and test percentages must sum to 1.0. Got {train_percentage} + {val_percentage} + {test_percentage} = {train_percentage + val_percentage + test_percentage}"
        )

    num_paths = driving_paths.shape[0]

    # Calculate split indices
    train_end = int(num_paths * train_percentage)
    val_end = train_end + int(num_paths * val_percentage)

    # Split the data
    train_driving_paths = driving_paths[:train_end]
    val_driving_paths = driving_paths[train_end:val_end]
    test_driving_paths = driving_paths[val_end:]

    train_driving_rough_paths = driving_rough_paths[:train_end]
    val_driving_rough_paths = driving_rough_paths[train_end:val_end]
    test_driving_rough_paths = driving_rough_paths[val_end:]

    train_solutions = solutions[:train_end]
    val_solutions = solutions[train_end:val_end]
    test_solutions = solutions[val_end:]

    train_solutions_rough_paths = solutions_rough_paths[:train_end]
    val_solutions_rough_paths = solutions_rough_paths[train_end:val_end]
    test_solutions_rough_paths = solutions_rough_paths[val_end:]

    # Create directories for each split
    for split in ["train", "val", "test"]:
        os.makedirs(f"{rde_locations[rde]}/{split}/paths", exist_ok=True)
        os.makedirs(f"{rde_locations[rde]}/{split}/rough_paths", exist_ok=True)

    # Save training data as individual files
    for i in range(train_driving_paths.shape[0]):
        path_id = str(i).zfill(3)
        jnp.save(f"{rde_locations[rde]}/train/paths/X_driver_{path_id}.npy", train_driving_paths[i])
        jnp.save(f"{rde_locations[rde]}/train/rough_paths/X_rough_driver_{path_id}.npy", train_driving_rough_paths[i])
        jnp.save(f"{rde_locations[rde]}/train/paths/y_solution_{path_id}.npy", train_solutions[i])
        jnp.save(f"{rde_locations[rde]}/train/rough_paths/y_rough_solution_{path_id}.npy", train_solutions_rough_paths[i])

    # Save validation data as individual files
    for i in range(val_driving_paths.shape[0]):
        path_id = str(i).zfill(3)
        jnp.save(f"{rde_locations[rde]}/val/paths/X_driver_{path_id}.npy", val_driving_paths[i])
        jnp.save(f"{rde_locations[rde]}/val/rough_paths/X_rough_driver_{path_id}.npy", val_driving_rough_paths[i])
        jnp.save(f"{rde_locations[rde]}/val/paths/y_solution_{path_id}.npy", val_solutions[i])
        jnp.save(f"{rde_locations[rde]}/val/rough_paths/y_rough_solution_{path_id}.npy", val_solutions_rough_paths[i])

    # Save test data as individual files
    for i in range(test_driving_paths.shape[0]):
        path_id = str(i).zfill(3)
        jnp.save(f"{rde_locations[rde]}/test/paths/X_driver_{path_id}.npy", test_driving_paths[i])
        jnp.save(f"{rde_locations[rde]}/test/rough_paths/X_rough_driver_{path_id}.npy", test_driving_rough_paths[i])
        jnp.save(f"{rde_locations[rde]}/test/paths/y_solution_{path_id}.npy", test_solutions[i])
        jnp.save(f"{rde_locations[rde]}/test/rough_paths/y_rough_solution_{path_id}.npy", test_solutions_rough_paths[i])

    print(
        f"""
Saved train/val/test split ({train_percentage*100:.0f}%/{val_percentage*100:.0f}%/{test_percentage*100:.0f}%) for {driver} and {rde}:
    Dataset shapes (paths, rough_paths):
    {driver}:
        Train: Driver shape: {train_driving_paths.shape}, Rough driver shape: {train_driving_rough_paths.shape}
        Val:   Driver shape: {val_driving_paths.shape}, Rough driver shape: {val_driving_rough_paths.shape}
        Test:  Driver shape: {test_driving_paths.shape}, Rough driver shape: {test_driving_rough_paths.shape}
    {rde}:
        Train: Solution shape: {train_solutions.shape}, Rough solution shape: {train_solutions_rough_paths.shape}
        Val:   Solution shape: {val_solutions.shape}, Rough solution shape: {val_solutions_rough_paths.shape}
        Test:  Solution shape: {test_solutions.shape}, Rough solution shape: {test_solutions_rough_paths.shape}
    Data saved to {rde_locations[rde]}/
"""
    )


def save_rde_and_driver_paths(
    key: Array,
    driver: Driver,
    rde: RDE,
    hurst: float,
    signature_type: Literal["signature", "log_signature"],
    signature_depth: int,
    num_steps: int,
    num_paths: int,
    dim: int = 1,
    train_percentage: float = 0.8,
    val_percentage: float = 0.1,
    test_percentage: float = 0.1,
    **rde_specific_params: int | float,
) -> None:

    # Create base directory for the RDE
    os.makedirs(rde_locations[rde], exist_ok=True)

    keys = random.split(key, num_paths)  # This is how we do num_paths paths

    match driver:
        case Driver.fBM:
            vmap_davies_harte = jax.vmap(fbm_davies_harte, in_axes=(0, None, None, None, None, None))
            driving_paths: jax.Array = vmap_davies_harte(keys, 0, 1, num_steps, hurst, dim)

            if signature_type == "signature":
                driving_rough_paths = jax.vmap(get_signature, in_axes=(0, None, None))(driving_paths, signature_depth, False)
            elif signature_type == "log_signature":
                driving_rough_paths = jax.vmap(get_log_signature, in_axes=(0, None, None, None))(driving_paths, signature_depth, "lyndon", False)
            else:
                raise ValueError(f"Signature type {signature_type} not supported")
        case _:
            raise ValueError(f"Driver {driver} not supported")

    match rde:
        case RDE.fOU:
            # Extract the RDE-specific parameters
            lam = rde_specific_params["lam"]
            sigma = rde_specific_params["sigma"]
            y0 = rde_specific_params["y0"]

            # Create a partial function with the RDE-specific parameters fixed
            partial_fractional_ou_process = lambda driving_path, num_steps: fractional_ou_process(driving_path, num_steps, lam, sigma, y0)

            vmap_fractional_ou_process = jax.vmap(partial_fractional_ou_process, in_axes=(0, None))
            solutions = vmap_fractional_ou_process(driving_paths, num_steps)
            if signature_type == "signature":
                solutions_rough_paths = jax.vmap(get_signature, in_axes=(0, None, None))(solutions.ys, signature_depth, False)
            elif signature_type == "log_signature":
                solutions_rough_paths = jax.vmap(get_log_signature, in_axes=(0, None, None, None))(solutions.ys, signature_depth, "lyndon", False)
            else:
                raise ValueError(f"Signature type {signature_type} not supported")
        case _:
            raise ValueError(f"RDE {rde} not supported")

    save_train_val_test_split(
        driving_paths,
        driving_rough_paths,
        solutions.ys,
        solutions_rough_paths,
        train_percentage,
        val_percentage,
        test_percentage,
        driver,
        rde,
    )


if __name__ == "__main__":
    save_rde_and_driver_paths(
        random.PRNGKey(42),
        Driver.fBM,
        RDE.fOU,
        0.4,
        "log_signature",
        3,
        1000,
        100,
        2,
        0.8,
        0.1,
        0.1,
        lam=1.0,
        sigma=0.5,
        y0=1.0,
    )
