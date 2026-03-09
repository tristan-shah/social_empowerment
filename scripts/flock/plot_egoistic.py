from pathlib import Path
from jax import numpy as jnp

import matplotlib.pyplot as plt

if __name__ == '__main__':


    path = Path('results/flock/')
    egoistic_path = path / 'behavioregoistic_agents20_grid10.0_speed1.0_radius1.0_dt0.05_horizon5_power2.0_alpha0.01_noise1.0_steps500_seed0'
    passive_path = path / 'behaviorpassive_agents20_grid10.0_speed1.0_radius1.0_dt0.05_horizon5_power2.0_alpha0.01_noise1.0_steps500_seed0'

    egoistic_empowerment = jnp.load(egoistic_path / 'empowerment_hist.npy')
    passive_empowerment = jnp.load(passive_path / 'empowerment_hist.npy')

    egoistic_order = jnp.load(egoistic_path / 'order_parameter_hist.npy')
    passive_order = jnp.load(passive_path / 'order_parameter_hist.npy')

    print(egoistic_empowerment)

    fig, ax = plt.subplots(2, 1)


    ax[0].set_title('Empowerment')
    ax[0].set_ylabel('Empowerment (nats)')
    for i in range(egoistic_empowerment.shape[1]):
        ax[0].plot(egoistic_empowerment[:, i], color = 'blue', alpha = 0.5)
        ax[0].plot(passive_empowerment[:, i], color = 'orange', alpha = 0.5)


    ax[1].set_title('Order Parameter')
    ax[1].set_ylabel('Order Parameter')
    ax[1].set_xlabel('Timestep')
    ax[1].plot(egoistic_order, color = 'blue', label = 'Egoistic')
    ax[1].plot(passive_order, color = 'orange', label = 'Passive')

    fig.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi = 300)
    plt.show()