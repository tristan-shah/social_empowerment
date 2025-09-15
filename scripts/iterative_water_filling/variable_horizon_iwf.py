import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import batch_diag, iterative_waterfilling, waterfilling_operator

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    num_agents = 2
    power = jnp.ones(num_agents)
    alpha = 0.90

    receive_dim = 5 ## shared recieve dimention
    transmit_dim = 10 ## shared transmit dimention (action size)

    ## different horizons
    horizon_0 = 80
    horizon_1 = 60

    message_dim_0 = transmit_dim * horizon_0
    message_dim_1 = transmit_dim * horizon_1

    key, _ = jax.random.split(key)
    H_agent_0 = jax.random.normal(key, (receive_dim, message_dim_0))
    key, _ = jax.random.split(key)
    H_noise_0 = jax.random.normal(key, (receive_dim, message_dim_1))

    key, _ = jax.random.split(key)
    H_agent_1 = jax.random.normal(key, (receive_dim, message_dim_1))
    key, _ = jax.random.split(key)
    H_noise_1 = jax.random.normal(key, (receive_dim, message_dim_0))

    max_message_dim = max(message_dim_0, message_dim_1)

    ## determine the padding for each agent's message
    pad_0 = max_message_dim - message_dim_0
    pad_1 = max_message_dim - message_dim_1

    ## pad the agent channel matrices to be the maximum message length
    H_agent_0 = jnp.pad(H_agent_0, ((0, 0), (0, pad_0)))
    H_agent_1 = jnp.pad(H_agent_1, ((0, 0), (0, pad_1)))

    ## stack the agent channel matrices
    H_agent = jnp.stack([H_agent_0, H_agent_1])

    ## initial agent covariance matrices
    S = batch_diag(jnp.ones((2, max_message_dim)))
    ## noise covariance
    S_z = jnp.eye(receive_dim)

    '''
    padding for noise matrices are reversed because the noise for agent 0
    comes from the message from agent 1 and vice versa
    '''
    H_noise_0 = jnp.pad(H_noise_0, ((0, 0), (0, pad_1)))
    H_noise_1 = jnp.pad(H_noise_1, ((0, 0), (0, pad_0)))

    ## zero out the block diagonal
    H_noise = jnp.stack([
        jnp.stack([H_agent_0 * 0.0,     H_noise_0]),
        jnp.stack([H_noise_1,           H_agent_1 * 0.0])
    ])

    # i, e, S = iterative_waterfilling(H_agent, H_noise, S, S_z, power, alpha)
    # print(i)
    # print(e)
    # print(S.shape)

    iterations = 10

    hist = jnp.zeros((iterations, num_agents))
    e_prev = jnp.ones(num_agents) * jnp.inf

    for i in range(iterations):

        e, S_ = waterfilling_operator(H_agent, H_noise, S, S_z, power)
        ## simultanious iterative waterfilling update
        S = (1-alpha) * S + alpha * S_
        hist = hist.at[i].set(e)
        print(i, e)

    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Link Channel Capacity')
    # for i in range(num_agents):
    #     ax.plot(hist[:, i])
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].set_title('Channel Capacity Over Iterations')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Link Channel Capacity')
    ax[0].plot(hist[:, 0], label = f'Horizon {horizon_0}')
    ax[0].plot(hist[:, 1], label = f'Horizon {horizon_1}')

    ax[1].set_title('Variance of Message')
    ax[1].set_xlabel('Horizon')
    ax[1].set_ylabel('Variance')
    ax[1].plot(jnp.diag(S[0])[:message_dim_0])
    ax[1].plot(jnp.diag(S[1])[:message_dim_1])
    ax[0].legend()
    fig.tight_layout()
    plt.show()