from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.utils import smooth_angle_wrap

if __name__ == '__main__':


    ave_empowerment = jnp.load(f'seed=0-ave-empowerment.npy')
    ave_angle = jnp.abs(smooth_angle_wrap(jnp.load('seed=0-ave-X.npy')[:, 0] - jnp.pi))

    solo_empowerment = jnp.load('seed=0-solo-empowerment.npy')
    solo_angle = jnp.abs(smooth_angle_wrap(jnp.load('seed=0-solo-X.npy')[:, 0] - jnp.pi))

    steps = len(ave_empowerment)
    dt = 0.01

    fig, ax = plt.subplots(1, 1)

    ax.set_ylabel('Empowerment (nats)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, steps)
    ax.plot(ave_empowerment[:, 0], label = 'Assisted')
    ax.plot(solo_empowerment[:, 0], label = 'Unassisted')
    ax.legend(loc = 'upper left')

    n_ticks = 5
    positions = jnp.linspace(0, steps - 1, n_ticks)
    labels = jnp.linspace(0.0, steps * dt, n_ticks)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation = 'horizontal')
    fig.tight_layout()
    fig.savefig('ave.png', dpi = 300)
    plt.show()

    fig, ax = plt.subplots(1, 1)

    ax.set_ylabel('Angle from upright (rad)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, steps)
    ax.plot(ave_angle, label='Assisted')
    ax.plot(solo_angle, label='Unassisted')
    ax.legend(loc='lower left')

    n_ticks = 5
    positions = jnp.linspace(0, steps - 1, n_ticks)
    labels = jnp.linspace(0.0, steps * dt, n_ticks)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation='horizontal')

    fig.tight_layout()
    fig.savefig('angle.png', dpi=300)
    plt.show()