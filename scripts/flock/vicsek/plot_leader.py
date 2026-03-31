from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dt = 0.05

    empowerment_path = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=leader-grid_size=5.0-horizon=5-num_agents=100-observation_noise=1.0-power_density=2.0-radius=0.5-seed=0-speed=1.0-steps=2000/empowerment_hist.npy')

    empowerment_hist = jnp.load(empowerment_path)

    T, num_agents = empowerment_hist.shape
    t = jnp.arange(T) * dt


    leader_empowerment = empowerment_hist[:, 0]
    followers_empowerment = empowerment_hist[:, 1:]

    # Mean and std across agents at each timestep
    followers_mean = followers_empowerment.mean(axis=1)
    followers_std = followers_empowerment.std(axis=1)


    

    fig, ax = plt.subplots()
    ax.plot(t, leader_empowerment, color='blue', label='Leader')

    ax.plot(t, followers_mean, color='orange', label='Followers')
    ax.fill_between(
        t,
        followers_mean - followers_std,
        followers_mean + followers_std,
        color='orange',
        alpha=0.2
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Empowerment (nats)')
    ax.set_title('Leader vs Followers Empowerment')
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.tight_layout()
    fig.savefig('leader.png', dpi = 300)
    plt.show()