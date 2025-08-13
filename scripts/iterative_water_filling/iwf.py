import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import waterfilling_implicit, split_channel_matrix, waterfilling_operator, batch_diag

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    receive_dim = 10
    num_agents = 3
    horizon = 20
    agent_transmit_dim = 10
    # transmit_dim = num_agents * agent_transmit_dim

    power = jnp.ones(num_agents)
    iterations = 100
    alpha = 0.7
    # alpha = 0.0
    assert 0.0 <= alpha < 1.0

    ## construct a random channel matrix
    H = jax.random.normal(key, (receive_dim, horizon, num_agents * agent_transmit_dim))

    H_agent, H_noise = split_channel_matrix(H, num_agents)

    '''
    construct the covariance matrix of each agent
    the size of the covariance matrix will be determined by horizon and the size of its message.
    '''
    S = batch_diag(jax.random.uniform(key, (num_agents, horizon * agent_transmit_dim)) * 0.2)
    S_z = jnp.eye(receive_dim) * 1.0

    hist = jnp.zeros((iterations, num_agents))

    for i in range(iterations):
        e, S_ = waterfilling_operator(H_agent, H_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_
        hist = hist.at[i].set(e)
        print(e)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Link Channel Capacity')
    for i in range(num_agents):
        ax.plot(hist[:, i])
    plt.show()