import jax
from jax import numpy as jnp

import matplotlib.pyplot as plt

if __name__ == '__main__':

    colab_alignment = jnp.stack([
        jnp.load('colab_alignment_0.npy'),
        jnp.load('colab_alignment_1.npy'),
        jnp.load('colab_alignment_2.npy'),
        jnp.load('colab_alignment_3.npy'),
        jnp.load('colab_alignment_4.npy'),
        jnp.load('colab_alignment_5.npy'),
        jnp.load('colab_alignment_6.npy'),
        jnp.load('colab_alignment_7.npy'),
    ])

    domin_alignment = jnp.stack([
        jnp.load('domin_alignment_0.npy'),
        jnp.load('domin_alignment_1.npy'),
        jnp.load('domin_alignment_2.npy'),
        jnp.load('domin_alignment_3.npy'),
        jnp.load('domin_alignment_4.npy'),
        jnp.load('domin_alignment_5.npy'),
        jnp.load('domin_alignment_6.npy'),
        jnp.load('domin_alignment_7.npy'),
    ])

    ## subtract the minimum value of each timeseries
    colab_alignment = colab_alignment - colab_alignment.min(axis = 1, keepdims = True)
    mean_colab_alignment = colab_alignment.mean(axis = 0)
    std_colab_alignment = colab_alignment.std(axis=0) / 2  # Standard deviation for error bars

    ## subtract the minimum value of each timeseries
    domin_alignment = domin_alignment - domin_alignment.min(axis = 1, keepdims = True)
    mean_domin_alignment = domin_alignment.mean(axis = 0)
    std_domin_alignment = domin_alignment.std(axis=0) / 2  # Standard deviation for error bars


    x = jnp.linspace(0.0, 15.0, mean_colab_alignment.shape[0])  # x-values (indices)

    fig, ax = plt.subplots(1, 1)

    ax.set_title('Temporal Strategy Evolution', fontsize = 14)
    ax.set_xlabel('Interaction Time (s)', fontsize = 14)
    ax.set_ylabel('Strategy Gap', fontsize = 14)

    ax.plot(x, mean_colab_alignment, color = 'green', label = 'Collaboration')
    ax.plot(x, mean_domin_alignment, color = 'blue', label = '(Left) Domination')

    ax.tick_params(axis = 'both', labelsize = 12)
    # Add shaded error band
    ax.fill_between(
        x,
        jnp.maximum(mean_colab_alignment - std_colab_alignment, 0),
        mean_colab_alignment + std_colab_alignment,
        color='green',
        alpha=0.2,
    )

    ax.fill_between(
        x,
        jnp.maximum(mean_domin_alignment - std_domin_alignment, 0),
        mean_domin_alignment + std_domin_alignment,
        color='blue',
        alpha=0.2,
    )

    ax.set_xlim(0.0, 15.0)

    n_ticks = 5
    import numpy as np
    positions = np.linspace(0, 15.0, n_ticks)
    labels = np.linspace(0.0, 15.0, n_ticks)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation = 'horizontal')

    ax.legend(fontsize = 12)
    fig.tight_layout()
    fig.savefig('strategy_dist.png', dpi = 300)
    plt.show()