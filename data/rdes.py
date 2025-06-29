"""
This file is used to generate the driving signals and the solutions to the RDEs.

We use the following parlance:
* RDE: The rough differential equation (RDE).
* Driving signal: The signal that is used to drive the RDE.
* Solution: The solution to the RDE.
* Rough path: The log signature tuple (X. 𝕏) that drives the RDE.

"""

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
from driver_and_solution_info import RDE, Driver, rde_solution_locations, driver_path_locations, driver_rough_path_locations
from driving_signals import fbm_davies_harte
from quicksig import get_log_signature


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def fractional_ou_process(
    driving_path: jnp.ndarray,
    num_steps: int,
    lam: float,
    sigma: float,
    y0: float | jnp.ndarray,
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
    # In diffrax, the driving noise path is provided as a `Control`.
    # `LinearInterpolation` is a simple way to create a control from a path.
    assert driving_path.shape == (num_steps + 1,), f"fbm_path must have length {num_steps + 1}, got {driving_path.shape}"

    timespan = jnp.linspace(0, 1, num_steps + 1)  # THIS CURRENTLY DOES NO INTERPOLATION
    control = LinearInterpolation(ts=timespan, ys=driving_path)
    assert False, (timespan.shape, control.ts.shape)

    drift = lambda t, y, args: -lam * y
    diffusion = lambda t, y, args: sigma * jnp.ones_like(y)

    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, control))

    # Choose a solver
    solver = Dopri5()

    # Get time-interval and discretization from the provided time points
    t0 = timespan[0]
    t1 = timespan[-1]
    discretization = timespan[1] - timespan[0] if len(timespan) > 1 else 0.1

    # Specify that we want the solution at the same points as our noise path
    saveat = SaveAt(ts=timespan)

    # Solve the equation
    sol = diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=discretization,
        y0=jnp.asarray(y0),
        saveat=saveat,
    )

    return sol


def save_rde_and_driver_paths(
    key: Array,
    driver: Driver,
    rde: RDE,
    hurst: float,
    num_steps: int,
    num_paths: int,
    lam: float,
    sigma: float,
    y0: float | jnp.ndarray,
) -> None:

    os.makedirs(driver_path_locations[driver], exist_ok=True)
    os.makedirs(rde_solution_locations[rde], exist_ok=True)

    keys = random.split(key, num_paths)  # This is how we do num_paths paths

    match driver:
        case Driver.fBM:
            vmap_davies_harte = jax.vmap(fbm_davies_harte, in_axes=(0, None, None, None, None))
            driving_paths: jax.Array = vmap_davies_harte(keys, 0, 1, num_steps, hurst)
            rough_paths = jax.vmap(get_log_signature, in_axes=(0, None, None, None))(driving_paths, 3, "lyndon", False)
        case _:
            raise ValueError(f"Driver {driver} not supported")
    jnp.save(f"{driver_path_locations[driver]}/fbm_path_h{hurst}.npy", driving_paths)
    jnp.save(f"{driver_rough_path_locations[driver]}/fbm_rough_path_h{hurst}.npy", rough_paths)

    match rde:
        case RDE.fOU:
            vmap_fractional_ou_process = jax.vmap(fractional_ou_process, in_axes=(0, None, None, None, None))
            solutions = vmap_fractional_ou_process(driving_paths, num_steps, lam, sigma, y0)
        case _:
            raise ValueError(f"RDE {rde} not supported")
    jnp.save(f"{rde_solution_locations[rde]}/fOU_solution_h{hurst}.npy", solutions.ys)

    print(f"Saved {num_paths} {driver} paths and {num_paths} {rde} solutions to {driver_path_locations[driver]} and {rde_solution_locations[rde]}")


if __name__ == "__main__":
    save_rde_and_driver_paths(
        random.PRNGKey(42),
        Driver.fBM,
        RDE.fOU,
        0.4,
        1000,
        100,
        1.0,
        0.5,
        1.0,
    )
