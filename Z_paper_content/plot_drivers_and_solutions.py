import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from data.driver_and_solution_info import driver_path_locs, rde_solution_locs, RDE, Driver


def plot_driving_signal(driver: Driver):
    paths = []
    for file in os.listdir(driver_path_locs[driver]):
        if file.endswith(".npy") and file != "fbm_path_h0.4.npy":  # Skip the main file, only use individual files
            path = jnp.load(f"{driver_path_locs[driver]}/{file}")
            paths.append(path)

    if not paths:
        print(f"No individual path files found in {driver_path_locs[driver]}")
        return

    # Convert to numpy array for plotting
    paths_array = np.array(paths)
    time_points = np.linspace(0, 1, paths_array.shape[1])

    fig, ax = plt.subplots()
    ax.plot(time_points, paths_array.T, linewidth=0.5, color="darkgrey")
    ax.set_xlabel("Time")
    ax.set_ylabel(driver.value)
    os.makedirs("Z_paper_content/figures", exist_ok=True)
    plt.savefig(f"Z_paper_content/figures/{driver.value}.svg", bbox_inches="tight")
    plt.close()
    print(f"Plotted {len(paths)} {driver.value} paths")


def plot_rde_solution(rde: RDE):
    paths = []
    for file in os.listdir(rde_solution_locs[rde]):
        if file.endswith(".npy"):
            # Load the RDE solutions - these are Solution objects from diffrax
            solutions = jnp.load(f"{rde_solution_locs[rde]}/{file}", allow_pickle=True)
            # Extract the ys (solution values) from each Solution object
            for sol in solutions:
                if hasattr(sol, "ys") and sol.ys is not None:
                    paths.append(sol.ys)

    if not paths:
        print(f"No solution files found in {rde_solution_locs[rde]}")
        return

    # Convert to numpy array for plotting
    paths_array = np.array(paths)
    time_points = np.linspace(0, 1, paths_array.shape[1])

    fig, ax = plt.subplots()
    ax.plot(time_points, paths_array.T, linewidth=0.5, color="darkgrey")
    ax.set_xlabel("Time")
    ax.set_ylabel(rde.value)
    os.makedirs("Z_paper_content/figures", exist_ok=True)
    plt.savefig(f"Z_paper_content/figures/{rde.value}.svg", bbox_inches="tight")
    plt.close()
    print(f"Plotted {len(paths)} {rde.value} solutions")


if __name__ == "__main__":
    plot_driving_signal(Driver.fBM)
    plot_rde_solution(RDE.fOU)
