from functools import partial

import jax.numpy as jnp
from jax import random, Array
import jax
from driver_and_solution_info import Driver, driver_path_locations


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def fbm_davies_harte(key: Array, start_time: float, end_time: float, num_steps: int, hurst: float):
    """
    Generates sample paths of fractional Brownian Motion using the Davies Harte method with JAX.

    @author: Luke Thompson, PhD Student, University of Sydney
    @author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

    args:
        key:    JAX random key
        start_time: start time
        end_time: end time
        num_steps:      number of time steps within timeframe
        hurst:      Hurst parameter
    """
    gamma = lambda k, H: 0.5 * (jnp.abs(k - 1) ** (2 * H) - 2 * jnp.abs(k) ** (2 * H) + jnp.abs(k + 1) ** (2 * H))

    k_vals = jnp.arange(0, num_steps)
    g = gamma(k_vals, hurst)
    r = jnp.concatenate([g, jnp.array([0.0]), jnp.flip(g)[:-1]])

    # Step 1 (eigenvalues)
    j = jnp.arange(0, 2 * num_steps)
    k = 2 * num_steps - 1
    lk = jnp.fft.fft(r * jnp.exp(2 * jnp.pi * 1j * k * j * (1 / (2 * num_steps))))[::-1].real

    # Step 2 (get random variables)
    key, subkey = random.split(key)
    Vj = jnp.zeros((2 * num_steps, 2))

    key, subkey = random.split(key)
    Vj = Vj.at[0, 0].set(random.normal(subkey))

    key, subkey = random.split(key)
    Vj = Vj.at[num_steps, 0].set(random.normal(subkey))

    key, subkey = random.split(key)
    rvs = random.normal(subkey, shape=(num_steps - 1, 2))

    indices1 = jnp.arange(1, num_steps)
    indices2 = jnp.arange(2 * num_steps - 1, num_steps, -1)

    Vj = Vj.at[indices1, :].set(rvs)
    Vj = Vj.at[indices2, :].set(rvs)

    # Step 3 (compute Z)
    wk = jnp.zeros(2 * num_steps, dtype=jnp.complex64)
    wk = wk.at[0].set(jnp.sqrt(lk[0] / (2 * num_steps)) * Vj[0, 0])
    wk = wk.at[1:num_steps].set(jnp.sqrt(lk[1:num_steps] / (4 * num_steps)) * (Vj[1:num_steps, 0] + 1j * Vj[1:num_steps, 1]))
    wk = wk.at[num_steps].set(jnp.sqrt(lk[num_steps] / (2 * num_steps)) * Vj[num_steps, 0])
    wk = wk.at[num_steps + 1 : 2 * num_steps].set(jnp.sqrt(lk[num_steps + 1 : 2 * num_steps] / (4 * num_steps)) * (jnp.flip(Vj[1:num_steps, 0]) - 1j * jnp.flip(Vj[1:num_steps, 1])))

    Z = jnp.fft.ifft(wk)
    fGn = Z[0:num_steps].real
    fBm = jnp.cumsum(fGn) * (num_steps ** (-hurst))
    fBm = (end_time - start_time) ** hurst * (fBm)
    path = jnp.concatenate([jnp.array([0.0]), fBm])
    path = path.reshape(-1, 1)
    return path


if __name__ == "__main__":
    print(fbm_davies_harte(random.PRNGKey(42), 0, 1, 20, 0.4).shape)
