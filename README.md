# Training Process
For the purposes of this section we discuss SDEs and RDEs of the form:

dYₜ = μ(Yₜ, t)dt + σ(Yₜ, t)dXₜ

Where:
* μ(Yₜ, t)dt is the **drift**
* σ(Yₜ, t) is the **diffusion**
* dXₜ is the **driver**

## RDE Training Process
1. First run `data/driving_signals.py` to generate the **driver** for the RDEs which save to `data/drivers/DRIVER_paths/`.
1. Then run `data/rdes.py` (which loads the driver from above) which solves the given RDE and saves the solution to `data/solutions/RDE_solutions/`

To train MODEL_NAME, we then input some driving path from `data/drivers/fbm_paths`, and train it to predict the RDE solution in `data/solutions/RDE_solutions/` given that driver.

Notes:
* In line with QuickSig, all time series must be shape [timesteps, dim], even if its a 1D process. (e.g., 1000 timesteps of fractional Brownian motion would be of shape [1000, 1]).