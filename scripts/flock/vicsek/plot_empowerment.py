from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dt = 0.05

    # empowerment_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=100-observation_noise=1.0-power_density=2.0-radius=0.5-seed=0-speed=1.0-steps=2000/empowerment_hist.npy')
    empowerment_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000/empowerment_hist.npy')
    # passive_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=passive-grid_size=5.0-horizon=5-num_agents=100-observation_noise=1.0-power_density=2.0-radius=0.5-seed=0-speed=1.0-steps=2000/empowerment_hist.npy')

    empowerment_hist = jnp.load(empowerment_path)
    # passive_hist = jnp.load(passive_path)

    T, num_agents = empowerment_hist.shape
    t = jnp.arange(T) * dt

    print(empowerment_hist.shape)

    # Mean and std across agents at each timestep
    empowerment_mean = empowerment_hist.mean(axis=1)
    empowerment_std = empowerment_hist.std(axis=1)

    # passive_mean = passive_hist.mean(axis=1)
    # passive_std = passive_hist.std(axis=1)

    fig, ax = plt.subplots()

    # Egoistic
    ax.plot(t, empowerment_mean, color='blue', label='Egoistic')
    ax.fill_between(
        t,
        empowerment_mean - empowerment_std,
        empowerment_mean + empowerment_std,
        color='blue',
        alpha=0.2
    )

    # # Passive
    # ax.plot(t, passive_mean, color='orange', label='Passive')
    # ax.fill_between(
    #     t,
    #     passive_mean - passive_std,
    #     passive_mean + passive_std,
    #     color='orange',
    #     alpha=0.2
    # )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Empowerment (nats)')
    ax.set_title('Mean Empowerment ± Standard Deviation')
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    fig.savefig('empowerment.png', dpi = 300)
    plt.show()