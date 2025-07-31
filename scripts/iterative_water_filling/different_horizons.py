import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(0)
    alpha = 0.01
    horizon = jnp.array([50, 100, 20, 10])
    power = jnp.array([0.75, 0.10, 1.40, 0.70])

    state_dim = 8
    control_dim = 4

    agent_state_adr = [
        jnp.array([0, 1]),  ## agent 0 state adr
        jnp.array([2, 3]),  ## agent 1 state adr
        jnp.array([4, 5]),  ## agent 2 state adr
        jnp.array([6, 7]),  ## agent 3 state adr
    ]

    agent_ctrl_adr = [
        jnp.array([0]),     ## agent 0 control adr
        jnp.array([1]),     ## agent 1 control adr
        jnp.array([2]),     ## agent 2 control adr
        jnp.array([3]),     ## agent 3 control adr
    ]

    num_agents = len(agent_state_adr)

    A = jax.random.normal(key, (horizon.max(), state_dim, state_dim))
    B = jax.random.normal(key, (horizon.max(), state_dim, control_dim))

    F_agent = []
    F_noise = [[] for _ in range(num_agents)]
    S = [] ## covariance matrix for each agent

    '''
    iterate over each agent and compute an F matrix with a different horizon.
    '''
    for i, h in enumerate(horizon):
        ## agent i, horizon h
        
        ## create a zero covariance matrix for each agent
        S.append(jnp.zeros((h, h)))

        ## compute the sensitivity matrix for the given horizon
        F = compute_F_from_A_B(A[:h], B[:h])
        F = jnp.permute_dims(F, (1, 0, 2))
        F_agent.append(F[agent_state_adr[i], :, agent_ctrl_adr[i]])

        for j in range(num_agents):
            if i == j:
                F_noise[i].append(F[agent_state_adr[j], :, agent_ctrl_adr[i]] * 0.0)
            else:
                F_noise[i].append(F[agent_state_adr[j], :, agent_ctrl_adr[i]])


    F_noise = [[F_noise[j][i] for j in range(num_agents)] for i in range(num_agents)]

    for i in range(len(F_noise)):
        for j in range(len(F_noise[i])):
            print(F_noise[i][j].shape, end = ' ')
        print(end = '\n')

    print()

    for i in range(len(S)):
        print(S[i].shape)