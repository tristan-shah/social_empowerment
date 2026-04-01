import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Horizons to plot
horizons = [75, 125, 150]
state_type = 'angle'
control_type = 'egoistic'
alpha = 0.01
observation_noise = 1.0
stiffness = 3.0
damping = 0.1

# Outcome heatmap colors
colors = ['lightgray', 'blue', 'orange', 'green']
labels = ['Neither', 'Left', 'Right', 'Both']
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 5))

# Ensure axes is always a list for consistent indexing
if len(horizons) == 1:
    axes = [axes]

# Store last image for colorbar
last_img = None

for ax, horizon in zip(axes, horizons):
    root = Path(f'results/linked_pendulum/control_type={control_type}/'
                f'state_type={state_type}-horizon={horizon}-alpha={alpha}-'
                f'observation_noise={observation_noise}-stiffness={stiffness}-damping={damping}-steps=2000-min_power=0.1-max_power=4.0')

    powers = jnp.load(root / 'powers.npy')
    outcomes = jnp.load(root / 'outcomes.npy')
    resolution = len(powers)
    
    tick_spacing = max(1, resolution // 10)
    ticks = jnp.unique(jnp.concatenate([jnp.arange(0, resolution, tick_spacing), jnp.array([resolution-1])]))
    
    img = ax.imshow(outcomes, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    last_img = img  # keep reference for colorbar
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:.2f}' for v in powers[ticks]], rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{v:.2f}' for v in powers[ticks]])
    ax.set_xlabel('Right Agent Power')
    ax.set_ylabel('Left Agent Power')
    ax.set_title(f'Horizon = {horizon * 0.01} (s)')

# Create a single colorbar outside the last axis
divider = make_axes_locatable(axes[-1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(last_img, cax=cax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(labels)
cbar.set_label('Pendulum Upright (|θ - π| ≤ 1.0 rad)')

plt.tight_layout()
fig.savefig('outcome_heatmaps_horizons.png', dpi = 300)
plt.show()