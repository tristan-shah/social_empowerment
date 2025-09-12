'''
Script that runs the standard iterative waterfilling algorithm
'''

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import split_channel_matrix, waterfilling_operator, batch_diag

if __name__ == '__main__':
    key = jax.random.PRNGKey(41)

    receive_dim = 2
    num_agents = 50
    horizon = 2
    agent_transmit_dim = 1
    dm = horizon * agent_transmit_dim

    power = jnp.ones(num_agents)
    iterations = 5
    alpha = 0.3
    assert 0.0 <= alpha < 1.0

    ## construct a random channel matrix
    H = jax.random.normal(key, (receive_dim, horizon, num_agents * agent_transmit_dim))

    H_agent, H_noise = split_channel_matrix(H, num_agents)

    print(H_agent.shape, H_noise.shape)

    '''
    construct the covariance matrix of each agent
    the size of the covariance matrix will be determined by horizon and the size of its message.
    '''
    S = batch_diag(jnp.ones((num_agents, dm)))
    S_z = jnp.eye(receive_dim) * 10.0

    hist = jnp.zeros((iterations, num_agents))

    for i in range(iterations):
        e, S_ = waterfilling_operator(H_agent, H_noise, S, S_z, power)
        ## simultanious iterative waterfilling update
        S = alpha * S + (1 - alpha) * S_
        hist = hist.at[i].set(e)
        print(e)

        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        for agent in range(num_agents):
            ax.arrow(0.0, 0.0, S[agent][0, 0], S[agent][1, 0])
            ax.arrow(0.0, 0.0, S[agent][0, 1], S[agent][1, 1])

        # ax.arrow(0.0, 0.0, S[0][0, 0], S[0][1, 0], color = 'blue')
        # ax.arrow(0.0, 0.0, S[0][0, 1], S[0][1, 1], color = 'blue')

        # ax.arrow(0.0, 0.0, S[1][0, 0], S[1][1, 0], color = 'red')
        # ax.arrow(0.0, 0.0, S[1][0, 1], S[1][1, 1], color = 'red')
        plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Link Channel Capacity')
    for i in range(num_agents):
        ax.plot(hist[:, i])
    plt.show()