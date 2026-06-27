from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dt = 0.05


    empowerment_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000/order_parameter_hist.npy')
    passive_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=passive-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000/order_parameter_hist.npy')


    empowerment_order = jnp.load(empowerment_path)
    passive_order = jnp.load(passive_path)

    T = len(empowerment_order)
    t = jnp.arange(T) * dt

    fig, ax = plt.subplots()

    ax.plot(t, empowerment_order, color='blue', label='Egoistic')
    ax.plot(t, passive_order, color='orange', label='Baseline')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Order Parameter', fontweight = 'bold')
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    fig.savefig('order.png', dpi = 300)
    plt.show()