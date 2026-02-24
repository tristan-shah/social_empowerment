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

    receive_dim = 5
    num_agents = 10
    horizon = 100
    agent_transmit_dim = 5
    dm = horizon * agent_transmit_dim

    power = jnp.ones(num_agents)
    iterations = 200 #100
    alpha = 0.2 #0.7

    ## construct a random channel matrix
    H = jax.random.normal(key, (receive_dim, horizon, num_agents * agent_transmit_dim))

    H_agent, H_noise = split_channel_matrix(H, num_agents)

    print(H_agent.shape, H_noise.shape)

    '''
    construct the covariance matrix of each agent
    the size of the covariance matrix will be determined by horizon and the size of its message.
    '''
    S = batch_diag(jnp.ones((num_agents, dm)))
    S_z = jnp.eye(receive_dim) * 1.0

    hist = jnp.zeros((iterations, num_agents))
    e_prev = jnp.ones(num_agents) * jnp.inf

    for i in range(iterations):

        e, S_ = waterfilling_operator(H_agent, H_noise, S, S_z, power)
        ## simultanious iterative waterfilling update
        S = (1-alpha) * S + alpha * S_
        hist = hist.at[i].set(e)

        delta = e - e_prev
        rel_change = delta/e_prev
        e_prev = e
        print(e)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Link Channel Capacity')
    for i in range(num_agents):
        ax.plot(hist[:, i])
    plt.show()