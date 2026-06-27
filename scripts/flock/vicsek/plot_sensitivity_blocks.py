"""Sensitivity block norm vs. pairwise agent distance in the Vicsek flock.

For each ordered pair (i, j) with i != j we compute the Frobenius norm of the
linearised sensitivity block

    F^{ij}  =  [ partial angle_i / partial u_j^{1:T} ]

which measures how strongly agent j's probing actions over the planning
horizon propagate to agent i's heading angle.  The Gaussian neighbour kernel
in the Vicsek model couples nearby agents strongly and distant agents weakly,
so we expect  ||F^{ij}||_F  to decay with the minimum-image distance d(i,j).

Usage
-----
    python scripts/flock/vicsek/plot_sensitivity_blocks.py
    python scripts/flock/vicsek/plot_sensitivity_blocks.py --num_agents 20 --seeds 0 1 2 3 4

Output: sensitivity_vs_distance.pdf  (saved in the current directory)
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from soc_emp.dynamics import make_unroll
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix
from soc_emp.envs.flock.utils import (
    build_flock_state_matrix,
    decode_state,
    minimum_image_diff,
)
from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step

# ── Defaults (matches main.py) ───────────────────────────────────────────────
_DEFAULTS = dict(
    num_agents = 100,
    grid_size  = 5.0,
    radius     = 0.5,
    speed      = 1.0,
    J          = 0.1,
    D          = 0.0,
    horizon    = 5,
    seeds      = [0, 1, 2],
    n_bins     = 4,
    out_file   = "sensitivity_vs_distance.pdf",
)
# ─────────────────────────────────────────────────────────────────────────────


def compute_block_norms(
    linearize: callable,
    unroll:    callable,
    state_matrix: jax.Array,
    xt:        jax.Array,
    horizon:   int,
    key:       jax.Array,
) -> jax.Array:
    """Return the (N, N) matrix of Frobenius norms ||F^{ij}||_F.

    F^{ij} is the linearised sensitivity of agent i's angle to agent j's
    action sequence over the planning horizon.  The diagonal is identically
    zero (each agent's self-sensitivity is excluded here; it lives in
    F_agent, not F_noise).

    Parameters
    ----------
    linearize    : vmapped jacfwd of the step function over time.
    unroll       : stochastic unroll function.
    state_matrix : (N, 1) integer index array selecting each agent's angle.
    xt           : (state_dim,) current state.
    horizon      : planning horizon T.
    key          : JAX random key used to draw the noise trajectory.
    """
    num_agents = state_matrix.shape[0]
    U    = jnp.zeros((horizon, num_agents))
    keys = jax.random.split(key, horizon)

    X    = unroll(xt, U, keys)
    A, B = linearize(X[:-1], U, keys)

    # F[t, state_dim, ctrl_dim]: cumulative state→action sensitivity
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))           # (state_dim, T, ctrl_dim)

    # Split: F_noise[i, j] = sensitivity of full state to agent j's actions
    #        (shape: N x N x state_dim x message_dim, diagonal zeroed)
    _, F_noise = split_channel_matrix(F, num_agents)

    # Restrict state axis to agent i's own angle index
    # state_matrix[:, None, :, None] broadcasts to (N, N, 1, message_dim)
    F_noise = jnp.take_along_axis(
        F_noise,
        state_matrix[:, None, :, None],
        axis=2,
    )  # (N, N, 1, message_dim)

    # Frobenius norm over the last two axes for each (i, j) pair
    norms = jnp.linalg.norm(
        F_noise.reshape(num_agents, num_agents, -1),
        axis=2,
    )  # (N, N)
    return norms


def pairwise_distances(xt: jax.Array, num_agents: int, grid_size: float) -> jax.Array:
    """Minimum-image pairwise Euclidean distances, shape (N, N)."""
    x, y, _ = decode_state(xt, num_agents)
    pos      = jnp.stack([x, y], axis=1)          # (N, 2)
    diff     = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
    diff_mi  = minimum_image_diff(diff, grid_size)
    return jnp.linalg.norm(diff_mi, axis=2)        # (N, N)


def binned_statistics(d: np.ndarray, n: np.ndarray, num_bins: int):
    """Mean and std of n within equal-width distance bins."""
    edges   = np.linspace(0.0, d.max(), num_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means, stds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi)
        vals = n[mask]
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std()  if len(vals) else np.nan)
    return centres, np.array(means), np.array(stds)


def main(args):
    flock        = Vicsek(args.num_agents, args.grid_size, args.radius, args.speed, args.J, args.D)
    reset        = make_reset(flock)
    step         = make_step(flock)
    state_matrix = build_flock_state_matrix(args.num_agents, "angle")  # (N, 1)

    # Build helpers once so JIT is reused across seeds
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums=(0, 1))))
    unroll    = make_unroll(step, stochastic=True)

    off_diag = ~np.eye(args.num_agents, dtype=bool)   # mask for i != j

    # ── Collect (distance, norm) pairs for each random state ─────────────────
    all_d, all_n = [], []
    per_seed     = []   # store separately for per-seed scatter colours

    for seed in args.seeds:
        key = jax.random.key(seed)
        xt  = reset(key)

        norms = compute_block_norms(linearize, unroll, state_matrix, xt, args.horizon, key)
        dists = pairwise_distances(xt, args.num_agents, args.grid_size)

        d = np.array(dists[off_diag])
        n = np.array(norms[off_diag])

        per_seed.append((d, n))
        all_d.append(d)
        all_n.append(n)

    all_d = np.concatenate(all_d)
    all_n = np.concatenate(all_n)

    # ── Statistics ───────────────────────────────────────────────────────────
    corr = float(np.corrcoef(all_d, all_n)[0, 1])

    # Table uses args.n_bins (default 4); plot uses finer bins for a smooth curve
    PLOT_BINS = 20
    table_centres, table_means, table_stds = binned_statistics(all_d, all_n, args.n_bins)
    plot_centres,  plot_means,  plot_stds  = binned_statistics(all_d, all_n, PLOT_BINS)

    edges = np.linspace(0.0, all_d.max(), args.n_bins + 1)
    counts = [int(((all_d >= lo) & (all_d < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]

    SHOW_STD   = False  # set True to restore std column
    SHOW_COUNT = True  # set True to restore count column

    print(f"\nPearson r (distance vs. block norm) = {corr:.4f}\n")
    print(f"Mean ||F^ij||_F by distance bin (Pearson r = {corr:.4f}):\n")

    header = f"| {'Distance bin':<20} | {'Mean ||F^ij||':>16}"
    sep    = f"|{'-'*22}|{'-'*18}"
    if SHOW_STD:
        header += f" | {'Std ||F^ij||':>16}"
        sep    += f"|{'-'*18}"
    if SHOW_COUNT:
        header += f" | {'Count':>6}"
        sep    += f"|{'-'*8}"
    print(header + " |")
    print(sep    + "|")

    for lo, hi, mu, sd, cnt in zip(edges[:-1], edges[1:], table_means, table_stds, counts):
        bin_str = f"[{lo:.3f}, {hi:.3f})"
        row = f"| {bin_str:<20} | {mu:>16.4e}" if not np.isnan(mu) else f"| {bin_str:<20} | {'---':>16}"
        if SHOW_STD:
            row += f" | {sd:>16.4e}" if not np.isnan(sd) else f" | {'---':>16}"
        if SHOW_COUNT:
            row += f" | {cnt:>6}"
        print(row + " |")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))

    colours = cm.tab10(np.linspace(0.0, 0.9, len(args.seeds)))

    for idx, (d, n) in enumerate(per_seed):
        ax.scatter(
            d, n,
            s=3, alpha=0.25,
            color=colours[idx],
            label=f"Seed {args.seeds[idx]}",
            rasterized=True,
        )

    # Binned mean ± 1 std across all pooled states (finer bins for smooth curve)
    ax.plot(plot_centres, plot_means, color="black", linewidth=2.0, label="Binned mean")
    ax.fill_between(
        plot_centres, plot_means - plot_stds, plot_means + plot_stds,
        color="black", alpha=0.15, label=r"$\pm 1\,\sigma$",
    )

    ax.set_xlabel(r"Pairwise distance $d_{ij}$")
    ax.set_ylabel(r"$\|F^{ij}\|_F$")
    ax.set_title(
        "Sensitivity block norm vs. agent distance\n"
        f"(Vicsek, $N={args.num_agents}$, $r={args.radius}$, $T={args.horizon}$)"
    )
    ax.legend(markerscale=3, fontsize=8, framealpha=0.9)
    fig.tight_layout()

    fig.savefig(args.out_file, dpi=300)
    print(f"Saved → {args.out_file}")
    plt.show()


if __name__ == "__main__":
    d = _DEFAULTS
    parser = argparse.ArgumentParser(
        description="Plot sensitivity block norms vs. pairwise distance for the Vicsek flock."
    )
    # Vicsek model
    parser.add_argument("--num_agents", type=int,   default=d["num_agents"])
    parser.add_argument("--grid_size",  type=float, default=d["grid_size"])
    parser.add_argument("--radius",     type=float, default=d["radius"],  help="Neighbour interaction radius")
    parser.add_argument("--speed",      type=float, default=d["speed"])
    parser.add_argument("--J",          type=float, default=d["J"],       help="Alignment strength")
    parser.add_argument("--D",          type=float, default=d["D"],       help="Noise intensity")
    # Sensitivity computation
    parser.add_argument("--horizon",    type=int,   default=d["horizon"], help="Planning horizon T")
    # Sampling
    parser.add_argument("--seeds",      type=int,   default=d["seeds"], nargs="+", help="Random seeds for initial states")
    # Plot
    parser.add_argument("--n_bins",     type=int,   default=d["n_bins"],  help="Number of distance bins for mean curve")
    parser.add_argument("--out_file",   type=str,   default=d["out_file"])

    main(parser.parse_args())
