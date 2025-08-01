from typing import List

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix, waterfilling_implicit, compute_power

def waterfilling_operator(H_agent: Array, H_noise: List[Array]):
    return


if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(0)
    alpha = 0.01
    horizon = jnp.array([254, 247, 148, 232])
    # power = jnp.array([0.75, 0.10, 1.40, 0.70])
    power = jnp.ones(4)

    state_dim = 9
    control_dim = 5

    agent_state_adr = [
        jnp.array([0, 1]),  ## agent 0 state adr
        jnp.array([2, 3]),  ## agent 1 state adr
        jnp.array([4, 5]),  ## agent 2 state adr
        jnp.array([6, 7, 8]),  ## agent 3 state adr
    ]

    agent_ctrl_adr = [
        jnp.array([0, 1]),     ## agent 0 control adr
        jnp.array([2]),     ## agent 1 control adr
        jnp.array([3]),     ## agent 2 control adr
        jnp.array([4]),     ## agent 3 control adr
    ]

    num_agents = len(power)

    A = jax.random.normal(key, (horizon.max(), state_dim, state_dim)) * 0.2
    B = jax.random.normal(key, (horizon.max(), state_dim, control_dim)) * 0.2

    F_agent = []
    F_noise = [[] for _ in range(num_agents)]
    ## covariance matrix for each agent
    S = []
    ## observation covariance
    S_z = [jnp.eye(len(agent_state_adr[i])) * 0.0 for i in range(len(agent_state_adr))]

    '''
    iterate over each agent and compute an F matrix with a different horizon.
    '''
    for i, h in enumerate(horizon):
        ## agent i, horizon h
        
        ## create a zero covariance matrix for each agent
        # S.append(jnp.zeros((h, h)))
        S.append(jnp.diag(jax.random.uniform(key, (h,))))

        ## compute the sensitivity matrix for the given horizon
        F = compute_F_from_A_B(A[:h], B[:h])
        F = jnp.permute_dims(F, (1, 0, 2))

        print(F.min(), F.max())

        F_agent.append(
            F[jnp.ix_(agent_state_adr[0], jnp.arange(horizon[i]), agent_ctrl_adr[0])]
        )

        for j in range(num_agents):
            cross_idx = jnp.ix_(agent_state_adr[j], jnp.arange(horizon[i]), agent_ctrl_adr[i])

            if i == j:
                F_noise[i].append(F[cross_idx] * 0.0)
            else:
                F_noise[i].append(F[cross_idx])

    ## transpose
    F_noise = [[F_noise[j][i] for j in range(num_agents)] for i in range(num_agents)]

    iterations = 5
    hist = jnp.zeros((iterations, num_agents))

    for iteration in range(iterations):
        ## iterate over agents
        for i in range(len(F_noise)):

            S_noise = S_z[i]
            ## compute the total noise from other agents
            for j in range(len(F_noise[i])):
                S_noise = S_noise + F_noise[i][j] @ S[j] @ F_noise[i][j].T
            
            ## eigen-decomp on noise
            D, Q = jnp.linalg.eigh(S_noise)
            D_inv_sqrt = jnp.diag((D + 1e-12) ** -0.5)

            ## compute new channel matrix (1/\sqrt{D}) @ Q.T @ F_agent
            H = D_inv_sqrt @ Q.T @ F_agent[i]
            ## compute snr levels
            _, E, M = jnp.linalg.svd(H, full_matrices = False)
            eigs = jnp.power(E, 2.0).clip(min = 1e-12)

            ## compute waterline (for each agent) [nu_0, nu_1, ..., nu_k]
            nu = waterfilling_implicit(eigs, power[i])
            p = compute_power(nu, eigs)
            
            ## update covariances
            P = jnp.diag(p)

            '''
            M.T P M appears to be the reverse of what is defined in boyd's paper on IWF but it is correct because 
            SVD returns a transposed orthoganal matrix.
            '''
            ## obtain dense covariance matrices: M.T P M
            S[i] = M.T @ P @ M

            ## channel capacities
            e = 0.5 * jnp.sum(jnp.log(1.0 + p * eigs))
            hist = hist.at[iteration, i].set(e)

    print(hist)

    
    fig, ax = plt.subplots(1, 1)
    for a in range(num_agents):
        ax.plot(hist[:, a])
    plt.show()