import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import waterfilling_implicit, split_channel_matrix, waterfilling_operator, batch_diag

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    receive_dim = 10
    num_agents = 30
    horizon = 10
    agent_transmit_dim = 10
    transmit_dim = num_agents * agent_transmit_dim
    power = jnp.ones(num_agents)
    iterations = 50
    alpha = 0.6
    assert 0.0 <= alpha < 1.0

    H = jax.random.normal(key, (receive_dim, horizon, transmit_dim))

    H_agent, H_noise = split_channel_matrix(H, num_agents)

    S = batch_diag(jax.random.uniform(key, (num_agents, horizon * agent_transmit_dim)))
    S_z = jnp.eye(receive_dim) * 1.0

    hist = jnp.zeros((iterations, num_agents))

    for i in range(iterations):
        e, S_ = waterfilling_operator(H_agent, H_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_
        hist = hist.at[i].set(e)
        print(e)

    fig, ax = plt.subplots(1, 1)
    for i in range(num_agents):
        ax.plot(hist[:, i])
    plt.show()