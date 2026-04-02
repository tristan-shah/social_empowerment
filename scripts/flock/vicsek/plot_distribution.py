import json
from pathlib import Path

import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.envs.flock.vicsek import Vicsek
from soc_emp.envs.flock.utils import decode_state
from soc_emp.envs.flock.plot import render_video


def wrap_angle(theta):
    """Wrap angles to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_mean(theta):
    """Circular mean angle."""
    return np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))


def collect_angles(X, flock, start, offsets, relative=True):
    """
    Collect angles from several nearby timesteps and concatenate them.
    """
    angles = []

    for dt in offsets:
        idx = start + dt
        if idx < 0 or idx >= len(X):
            continue

        _, _, theta = decode_state(X[idx], flock.num_agents)
        theta = np.asarray(theta)

        if relative:
            mean_theta = circular_mean(theta)
            theta = wrap_angle(theta - mean_theta)
        else:
            theta = wrap_angle(theta)

        angles.append(theta)

    if len(angles) == 0:
        return np.array([])

    return np.concatenate(angles)


if __name__ == '__main__':

    root = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000')

    with open(root / 'params.json', 'r') as f:
        params = json.load(f)

    flock = Vicsek(
        params['num_agents'],
        params['grid_size'],
        params['radius'],
        params['speed'],
        params['J'],
        params['D']
    )

    X = jnp.load(root / 'trajectory.npy')
    # render_video(X, flock, path='vid.mp4')

    # -----------------------------
    # Publication-quality settings
    # -----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Four representative time windows
    starts = [20, 1000, 2000, 4000]

    # Backward-looking pooling: [start-10, ..., start]
    offsets = list(range(-10, 1))

    # Common binning across all panels
    bins = np.linspace(-np.pi, np.pi, 17)   # 16 bins

    # Nice tick labels
    xticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    xticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

    # ---------------------------------------
    # First pass: compute global max bar height
    # ---------------------------------------
    all_angles = {}
    global_ymax = 0.0

    for start in starts:
        a = collect_angles(X, flock, start, offsets, relative=True)
        all_angles[start] = a

        if len(a) == 0:
            continue

        weights = np.ones_like(a) / len(a)
        hist, _ = np.histogram(a, bins=bins, weights=weights)
        global_ymax = max(global_ymax, hist.max())

    # Add a little headroom
    global_ymax *= 1.08

    # ---------------------------------------
    # Second pass: plot with shared y-limits
    # ---------------------------------------
    for start in starts:
        a = all_angles[start]

        if len(a) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4.2, 3.6))

        weights = np.ones_like(a) / len(a)

        ax.hist(
            a,
            bins=bins,
            weights=weights,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.9
        )

        # ax.set_title(f'{start * 0.05} (s)')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, global_ymax)

        ax.set_xlabel(r'Relative angle $\Delta \theta$')
        ax.set_ylabel('Probability mass')

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # Cleaner axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Reference line at zero relative angle
        ax.axvline(0, linestyle='--', linewidth=1.0, alpha=0.8)

        fig.tight_layout()
        plt.savefig(f'vicsek_angle_distribution_t{start}.png', bbox_inches='tight', dpi=300)
        plt.show()


# import json
# from pathlib import Path

# import numpy as np
# from jax import numpy as jnp
# import matplotlib.pyplot as plt

# from soc_emp.envs.flock.vicsek import Vicsek
# from soc_emp.envs.flock.utils import decode_state
# from soc_emp.envs.flock.plot import render_video


# def wrap_angle(theta):
#     """Wrap angles to [-pi, pi)."""
#     return (theta + np.pi) % (2 * np.pi) - np.pi


# def circular_mean(theta):
#     """Circular mean angle."""
#     return np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))


# def collect_angles(X, flock, start, offsets, relative=True):
#     """
#     Collect angles from several nearby timesteps and concatenate them.

#     Parameters
#     ----------
#     X : array
#         Trajectory array
#     flock : Vicsek
#         Flock object
#     start : int
#         Base timestep
#     offsets : list[int]
#         Offsets relative to start to pool
#     relative : bool
#         If True, subtract circular mean heading at each timestep

#     Returns
#     -------
#     a : np.ndarray
#         Concatenated angles
#     """
#     angles = []

#     for dt in offsets:
#         _, _, theta = decode_state(X[start + dt], flock.num_agents)
#         theta = np.asarray(theta)

#         if relative:
#             mean_theta = circular_mean(theta)
#             theta = wrap_angle(theta - mean_theta)
#         else:
#             theta = wrap_angle(theta)

#         angles.append(theta)

#     return np.concatenate(angles)


# if __name__ == '__main__':

#     root = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000')

#     with open(root / 'params.json', 'r') as f:
#         params = json.load(f)

#     flock = Vicsek(
#         params['num_agents'],
#         params['grid_size'],
#         params['radius'],
#         params['speed'],
#         params['J'],
#         params['D']
#     )

#     X = jnp.load(root / 'trajectory.npy')
#     # render_video(X, flock, path = 'vid.mp4')

#     # -----------------------------
#     # Publication-quality settings
#     # -----------------------------
#     plt.rcParams.update({
#         "font.size": 12,
#         "axes.labelsize": 13,
#         "axes.titlesize": 13,
#         "xtick.labelsize": 11,
#         "ytick.labelsize": 11,
#         "axes.linewidth": 1.0,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     })


#     # Four representative time windows
#     starts = [20, 1000, 2000, 3990]

#     # Pool nearby timesteps for cleaner statistics
#     offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#     # Common binning across all panels
#     bins = np.linspace(-np.pi, np.pi, 17)   # 16 bins

#     fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)

#     for ax, start in zip(axes, starts):
#         a = collect_angles(X, flock, start, offsets, relative=True)

#         # Probability mass histogram (bars sum to 1)
#         weights = np.ones_like(a) / len(a)

#         ax.hist(
#             a,
#             bins=bins,
#             weights=weights,
#             edgecolor='black',
#             linewidth=0.8,
#             alpha=0.9
#         )

#         ax.set_title(f'{start * 0.05} (s)')
#         ax.set_xlim(-np.pi, np.pi)

#         # Cleaner axes
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         # Reference line at zero relative angle
#         ax.axvline(0, linestyle='--', linewidth=1.0, alpha=0.8)

#     axes[0].set_ylabel('Probability mass')
#     for ax in axes:
#         ax.set_xlabel(r'Relative angle $\Delta \theta$')

#     # Nice tick labels
#     xticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
#     xticklabels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
#     for ax in axes:
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)

#     fig.tight_layout()
#     plt.savefig('vicsek_angle_distributions.png', bbox_inches='tight', dpi = 300)
#     plt.show()
