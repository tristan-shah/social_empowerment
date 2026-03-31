import re
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.dynamics import make_step
from sweep_power import get_linked_pendulum_outcome, make_compute_group_empowerment, build_linked_pendulum_state_matrix, set_tendon_properties

if __name__ == '__main__':

    state_type = 'angle'
    control_type = 'egoistic'
    horizon = 125
    dt = 0.01
    stiffness = 3.0
    damping = 0.1
    alpha = 0.01
    observation_noise = 1.0
    N = 1 ## average the empowerment over the last N states in the trajectory

    root = Path(f'results/linked_pendulum/control_type={control_type}/state_type={state_type}-horizon={horizon}-alpha={alpha}-observation_noise={observation_noise}-stiffness={stiffness}-damping={damping}-steps=2000-min_power=0.1-max_power=4.0')

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    dyn = set_tendon_properties(dyn, stiffness, damping)
    step = make_step(dyn)

    state_matrix = build_linked_pendulum_state_matrix(state_type)
    U = jnp.zeros((horizon, dyn.control_dim))
    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, alpha, observation_noise)
    # inner vmap: over N timesteps for one episode (power is fixed per episode)
    # (N, state_dim), (num_agents,) -> (N, num_agents)
    empowerment_over_time = jax.vmap(compute_group_empowerment, in_axes=(0, None))
    # outer vmap: over batch
    # (batch, N, state_dim), (batch, num_agents) -> (batch, N, num_agents)
    batch_compute_group_empowerment = jax.jit(jax.vmap(empowerment_over_time, in_axes=(0, 0)))

    powers = jnp.load(root / 'powers.npy')
    resolution = len(powers)

    I, J = jnp.meshgrid(jnp.arange(resolution), jnp.arange(resolution), indexing='ij')
    I_flat = I.reshape(-1)
    J_flat = J.reshape(-1) 
    
    outcomes = jnp.zeros((resolution, resolution))
    empowerment_outcome = jnp.zeros((resolution, resolution, 2))



    outcomes = jnp.load(root / 'outcomes.npy')

    # empowerment_outcome: (resolution, resolution, 2) — agent 0 is left, agent 1 is right
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    tick_spacing = max(1, resolution // 10)
    ticks = jnp.unique(jnp.concatenate([jnp.arange(0, resolution, tick_spacing), jnp.array([resolution - 1])]))

    # titles = ['Left Agent Empowerment', 'Right Agent Empowerment']
    # for agent_idx, (ax, title) in enumerate(zip(axes[:2], titles)):
    #     img = ax.imshow(empowerment_outcome[:, :, agent_idx], origin='lower', aspect='auto')
    #     cbar = plt.colorbar(img, ax=ax)
    #     cbar.set_label('Empowerment (nats)')
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels([f'{v:.2f}' for v in powers[ticks]], rotation=90)
    #     ax.set_yticks(ticks)
    #     ax.set_yticklabels([f'{v:.2f}' for v in powers[ticks]])
    #     ax.set_xlabel('Right Agent Power')
    #     ax.set_ylabel('Left Agent Power')
    #     ax.set_title(title)

    colors = ['lightgray', 'blue', 'orange', 'green']
    labels = ['Neither', 'Left', 'Right', 'Both']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    img = axes.imshow(outcomes, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    axes.set_xticks(ticks)
    axes.set_xticklabels([f'{v:.2f}' for v in powers[ticks]], rotation=90)
    axes.set_yticks(ticks)
    axes.set_yticklabels([f'{v:.2f}' for v in powers[ticks]])
    axes.set_xlabel('Right Agent Power')
    axes.set_ylabel('Left Agent Power')
    axes.set_title('Outcome')
    cbar = plt.colorbar(img, ax=axes, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Pendulum Upright (|θ - π| ≤ 1.0 rad)')

    plt.tight_layout()
    fig.savefig(root / f'outcome_heatmap.png', dpi=300)
    plt.close(fig)



    '''
    for plotting full empowerment heatmap comparison
    '''
    # # sort files by batch index extracted from filename                                                                                                            
    # batch_files = sorted(root.glob('X_batch_*'), key=lambda p: int(re.search(r'\d+', p.stem).group()))
    # batch_get_linked_pendulum_outcome = jax.vmap(get_linked_pendulum_outcome)
    # effective_batch_size = None

    # for path in batch_files:
    #     batch_idx = int(re.search(r'\d+', path.stem).group())
    #     X_batch = jnp.load(path.resolve())
    #     pairs = jnp.load(root / f'pairs_batch_{batch_idx}.npy')

    #     last_N_states = X_batch[:, -N:, :]

    #     if effective_batch_size is None:
    #         effective_batch_size = X_batch.shape[0]

    #     start_idx = batch_idx * effective_batch_size
    #     end_idx = min(start_idx + effective_batch_size, resolution * resolution)
    #     actual_size = end_idx - start_idx

    #     last_N_states = last_N_states[:actual_size]
    #     pairs = pairs[:actual_size]

    #     # (batch, N, num_agents) -> mean over N -> (batch, num_agents)
    #     e = batch_compute_group_empowerment(last_N_states, pairs).mean(axis = 1)

    #     batch_I = I_flat[start_idx:end_idx]
    #     batch_J = J_flat[start_idx:end_idx]
    #     empowerment_outcome = empowerment_outcome.at[batch_I, batch_J].set(e)
    #     print(f'batch {batch_idx}: placed {actual_size} empowerments at [{start_idx}:{end_idx}]')

    # outcomes = jnp.load(root / 'outcomes.npy')

    # # empowerment_outcome: (resolution, resolution, 2) — agent 0 is left, agent 1 is right
    # from matplotlib.colors import ListedColormap, BoundaryNorm
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # tick_spacing = max(1, resolution // 10)
    # ticks = jnp.unique(jnp.concatenate([jnp.arange(0, resolution, tick_spacing), jnp.array([resolution - 1])]))

    # titles = ['Left Agent Empowerment', 'Right Agent Empowerment']
    # for agent_idx, (ax, title) in enumerate(zip(axes[:2], titles)):
    #     img = ax.imshow(empowerment_outcome[:, :, agent_idx], origin='lower', aspect='auto')
    #     cbar = plt.colorbar(img, ax=ax)
    #     cbar.set_label('Empowerment (nats)')
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels([f'{v:.2f}' for v in powers[ticks]], rotation=90)
    #     ax.set_yticks(ticks)
    #     ax.set_yticklabels([f'{v:.2f}' for v in powers[ticks]])
    #     ax.set_xlabel('Right Agent Power')
    #     ax.set_ylabel('Left Agent Power')
    #     ax.set_title(title)

    # colors = ['lightgray', 'blue', 'orange', 'green']
    # labels = ['Neither', 'Left', 'Right', 'Both']
    # cmap = ListedColormap(colors)
    # norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    # img = axes[2].imshow(outcomes, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    # axes[2].set_xticks(ticks)
    # axes[2].set_xticklabels([f'{v:.2f}' for v in powers[ticks]], rotation=90)
    # axes[2].set_yticks(ticks)
    # axes[2].set_yticklabels([f'{v:.2f}' for v in powers[ticks]])
    # axes[2].set_xlabel('Right Agent Power')
    # axes[2].set_ylabel('Left Agent Power')
    # axes[2].set_title('Outcome')
    # cbar = plt.colorbar(img, ax=axes[2], ticks=[0, 1, 2, 3])
    # cbar.ax.set_yticklabels(labels)
    # cbar.set_label('Pendulum Upright (|θ - π| ≤ 1.0 rad)')

    # plt.tight_layout()
    # fig.savefig(root / f'N={N}-empowerment_heatmap.png', dpi=300)
    # plt.close(fig)