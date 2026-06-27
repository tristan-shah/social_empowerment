# Multi-Agent Empowerment and the Emergence of Complex Behavior in Groups

Code for the paper **"Multi-Agent Empowerment and Emergence of Complex Behavior in Groups"**
(Shah, Nemenman, Polani, Tiomkin).

The package shows that maximizing empowerment alone produces non-trivial group-level
behavior in two qualitatively different environments:

1. **Linked pendulums** — two pendulums coupled by a tendon. Empowerment maximization
   produces **dominance hierarchies** when agents have unequal strength (one pendulum
   swings up, the other is suppressed) and **cooperation** when their strengths are
   comparable (both swing up).
2. **Vicsek flock** — a large controllable flock. Egoistic empowerment prevents the usual
   convergence to a single shared heading and instead drives the population into **two
   opposing directional bands**.

## How it works

For each agent `n`, the linearized sensitivity of the future state to actions is split into
a *direct* channel (the agent's effect on its own future) and *interference* channels (the
effect of every other agent, treated as noise). Each agent then solves a single-user
Gaussian-channel water-filling problem against the combined interference + observation
noise, and these updates are iterated to a fixed point (a Nash equilibrium of the
non-cooperative game). Once each agent's empowerment is computed, control actions are
chosen proportional to the gradient of empowerment with respect to the agent's actions.

Two control policies are studied:

- **Egoistic** — each agent acts to maximize *its own* empowerment.
- **Altruistic / leader** — an agent acts to maximize the empowerment of *another* agent.

The full implementation lives in `src/soc_emp/empowerment.py`:

- `compute_F_from_A_B` — builds the sensitivity (channel) matrix from per-step linearized
  dynamics `A`, `B`.
- `split_channel_matrix` — splits the channel matrix into each agent's direct channel and
  the interference channels from all others.
- `waterfilling_implicit` / `waterfilling_solver` — single-user water-filling (with a
  differentiable implicit form via `jax.lax.custom_root`).
- `waterfilling_operator` / `iterative_waterfilling` — the core multi-user iterative
  water-filling fixed-point iteration (Algorithm 1 in the paper).
- `compute_empowerment` / `compute_multiagent_empowerment` — single- and multi-agent
  empowerment, with JAX-differentiable gradients for control.

Everything is written in [JAX](https://github.com/jax-ml/jax) and is JIT-compiled,
`vmap`-batched, and `pmap`-parallelized across GPUs. Rigid-body dynamics use
[MuJoCo MJX](https://mujoco.readthedocs.io/), which provides differentiable simulation
needed for the linearization step.

## Repository layout

```
src/soc_emp/
  dynamics.py            # MuJoCo MJX Dynamics wrapper: step, linearize, render
  empowerment.py         # interference channel + iterative water-filling + control
  utils.py               # state helpers, angle wrapping, misc utilities
  envs/flock/
    flock.py             # deterministic soft-coupling flock model
    vicsek.py            # stochastic Vicsek active-matter model (used in the paper)
    utils.py, plot.py    # state encoding, order parameter, rendering

scripts/
  linked_pendulum/       # two coupled pendulums experiment
    sweep_power.py       # sweep over (left, right) agent power -> outcome heatmap
    plot_heatmap.py      # render the outcome heatmaps
    plot_outcomes.py     # render example trajectories for each outcome
    plot_ave.py          # altruistic ("ave") control plots
    sweep_power.sh       # SLURM launcher
  flock/vicsek/          # Vicsek flock experiment (paper version)
    main.py              # run a single flock simulation for a chosen behavior
    iwf.py               # standalone iterative water-filling demo
    run.sh               # SLURM launcher
    plot_*.py, render.py, efficiency.py  # figures and analysis
  flock/basic/           # deterministic flock variant (exploratory)

xml/custom/
  linked_pendulums.xml   # MuJoCo model: two tendon-coupled pendulums

results/                 # saved trajectories, metrics, figures, and videos
```

## Installation

Requires Python with a CUDA-capable GPU (the experiments are GPU-parallelized; CPU works
for small runs). The pinned dependencies (JAX 0.6.2 + CUDA 12, MuJoCo 3.3.3) are in
`requirements.txt`.

```bash
# create and activate an environment (conda or venv)
pip install -r requirements.txt
pip install -e .          # installs the soc_emp package from src/
```

## Running the experiments

### Linked pendulums

Sweep over a grid of left/right agent power budgets and classify each outcome (neither,
left, right, or both pendulums upright):

```bash
python scripts/linked_pendulum/sweep_power.py \
    --steps 2000 --horizon 130 --alpha 0.01 \
    --stiffness 3.0 --damping 0.1 \
    --state_type angle --control_type egoistic \
    --resolution 200 --max_power 4.0
```

Key arguments:

- `--control_type {egoistic, ave}` — egoistic self-empowerment, or `ave` (altruistic, all
  agents raise agent 0's empowerment).
- `--horizon` — empowerment planning horizon (in steps).
- `--stiffness` / `--damping` — tendon coupling between the two pendulums.
- `--resolution` — grid resolution of the power sweep.
- `--device_batch_size` — per-GPU batch size; the sweep is `pmap`-parallelized over devices.

Results (trajectories, per-cell outcomes, and an `outcome_heatmap.png`) are written under
`results/linked_pendulum/control_type=.../`. See `sweep_power.sh` for the SLURM submission
used to produce the paper figures.

### Vicsek flock

Run a single flock simulation under a chosen empowerment-driven behavior:

```bash
python scripts/flock/vicsek/main.py \
    --steps 2000 --num_agents 100 \
    --J 0.1 --D 0.0 --radius 0.5 \
    --horizon 5 --power_density 2.0 --alpha 0.01 \
    --observation_noise 1.0 \
    --behavior egoistic
```

`--behavior` selects how empowerment gradients drive control:

- `egoistic` — each agent maximizes its own empowerment (produces the opposing-band split).
- `leader` — agents act to empower a single leader.
- `feedback` — leader empowers the flock while the flock empowers the leader.
- `collective` — all agents maximize the summed group empowerment.
- `vanilla` — standard single-agent empowerment baseline.
- `passive` — no control (plain Vicsek dynamics, for comparison).

Vicsek parameters: `--J` alignment strength, `--D` noise intensity, `--num_agents`,
`--grid_size`, `--radius` (neighbor falloff), `--speed`. Each run saves the trajectory,
empowerment and order-parameter histories, figures, and a rendered `vid.mp4` under
`results/Vicsek/`. See `run.sh` for the SLURM launcher.

## Citation

```bibtex
@article{shah_multiagent_empowerment,
  title   = {Multi-Agent Empowerment and Emergence of Complex Behavior in Groups},
  author  = {Shah, Tristan and Nemenman, Ilya and Polani, Daniel and Tiomkin, Stas},
  year    = {2025}
}
```
