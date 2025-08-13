from typing import List

import jax
from jax import Array
from jax import numpy as jnp

from soc_emp.empowerment import waterfilling_implicit, compute_power

def waterfilling_operator(H_agent: Array, H_noise: List[Array], S_noise: List[Array], S_z: Array, power: float):
    return

@jax.jit
def compute_covaraince(H: Array, S: Array):
    return H @ S @ H.T

def pad_covariances(S: List[Array]):
    
    return

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    power = 2.0
    S_z = jnp.eye(2) * 0.1
    
    H_agent = jax.random.normal(key, (2, 3))
    S_agent = jax.random.uniform(key, (3, 3)) * 0.2

    key, _ = jax.random.split(key)
    H_noise_1 = jax.random.normal(key, (2, 4))
    S_noise_1 = jax.random.uniform(key, (4, 4)) * 0.2

    key, _ = jax.random.split(key)
    H_noise_2 = jax.random.normal(key, (2, 5))
    S_noise_2 = jax.random.uniform(key, (5, 5)) * 0.2

    key, _ = jax.random.split(key)
    H_noise_3 = jax.random.normal(key, (2, 6))
    S_noise_3 = jax.random.uniform(key, (6, 6)) * 0.2

    H_noise = [H_noise_1, H_noise_2, H_noise_3]
    S_noise = [S_noise_1, S_noise_2, S_noise_3]

    # agent_cov = H_agent @ S_agent @ H_agent.T
    # noise_cov = H_noise_1 @ S_noise_1 @ H_noise_1.T + H_noise_2 @ S_noise_2 @ H_noise_2.T + H_noise_3 @ S_noise_3 @ H_noise_3.T

    # print(
    #     sum([compute_covaraince(H_noise[i], S_noise[i]) for i in range(len(H_noise))])
    # )


    max_s_dim = max([S_noise[i].shape[0] for i in range(len(S_noise))])

    print(max_s_dim)

    print(S_noise[0].shape)
    print()
    print(jnp.pad(S_noise[0], ((0, 2), (0, 2))))



    # D, Q = jnp.linalg.eigh(noise_cov)
    # D_inv_sqrt = jnp.diag((D + 1e-12) ** -0.5)

    # H = D_inv_sqrt @ Q.T @ H_agent

    # ## compute snr levels
    # _, E, M = jnp.linalg.svd(H, full_matrices = False)
    # eigs = jnp.power(E, 2.0).clip(min = 1e-12)

    # mu = waterfilling_implicit(eigs, power)

    # p = compute_power(mu, eigs)
    # P = jnp.diag(p)
    # S_agent = M.T @ P @ M

    # print(S_agent)