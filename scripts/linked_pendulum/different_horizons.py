import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, waterfilling_implicit, compute_power

def get_all_but_i(a: list, i: int):
    return a[:i] + a[i+1:]

def waterfilling_operator(H: Array, noise_cov: Array, power: float):
    ## eigen-decomp on noise
    D, Q = jnp.linalg.eigh(noise_cov)
    D_inv_sqrt = jnp.diag((D + 1e-12) ** -0.5)
    H = D_inv_sqrt @ Q.T @ H
    ## compute snr levels
    _, E, M = jnp.linalg.svd(H, full_matrices = False)
    eigs = jnp.power(E, 2.0).clip(min = 1e-12)
    ## waterfilling_implicit is a differentiable jax function
    nu = waterfilling_implicit(eigs, power)
    p = compute_power(nu, eigs)
    ## update covariances
    P = jnp.diag(p)
    '''
    M.T P M appears to be the reverse of what is defined in boyd's paper on IWF but it is correct because 
    SVD returns a transposed orthoganal matrix.
    '''
    ## obtain dense covariance matrices: M.T P M
    S = M.T @ P @ M

    e = 0.5 * jnp.sum(jnp.log(1.0 + p * eigs), keepdims = True)
    return e, S

def linked_pendulum_empowerment(dyn: Dynamics, x0: Array, U: Array, horizon: Array, power: Array, alpha: float):

    pad = horizon.max() - horizon

    ## perform simulation
    X = unroll(dyn, x0, U)
    ## compute jacobians
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    ## state address for each agent
    agent_state_adr = [
        jnp.array([0, 2]),
        jnp.array([1, 3])
    ]

    ## control address for each agent
    agent_ctrl_adr = [
        jnp.array([0]),
        jnp.array([1]),
    ]

    F_0 = compute_F_from_A_B(A[:horizon[0]], B[:horizon[0]]) ## (horizon_0 x state x control)
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon_0 x control)

    F_1 = compute_F_from_A_B(A[:horizon[1]], B[:horizon[1]])## (horizon_1 x state x control)
    F_1 = jnp.permute_dims(F_1, (1, 0, 2)) ## (state x horizon_1 x control)

    '''
    Initialize covariance matrices.
    '''
    # S_0 = jnp.diag(jax.random.uniform(key, (horizon[0],)) * 0.1)
    S_0 = jnp.zeros((horizon[0], horizon[0]))

    # S_1 = jnp.diag(jax.random.uniform(key, (horizon[1],)) * 0.1)
    S_1 = jnp.zeros((horizon[1], horizon[1]))

    ## noise covariance
    S_z = jnp.eye(4) + jnp.diag(jnp.ones(4) * 0.1)
    S_z_0 = S_z[0:2, 0:2]
    S_z_1 = S_z[2:4, 2:4]

    ## indices for agent 0's channel matrix
    agent_0_idx = jnp.ix_(
        agent_state_adr[0],
        jnp.arange(0, horizon[0]),
        agent_ctrl_adr[0]
        )
    
    ## indices for agent 1's interference matrices
    noise_0_idx = jnp.ix_(
        agent_state_adr[0],
        jnp.arange(0, horizon[0]),
        agent_ctrl_adr[1]
        )

    F_0_agent = F_0[agent_0_idx].reshape(len(agent_state_adr[0]), -1)
    F_0_noise = F_0[noise_0_idx].reshape(len(agent_state_adr[0]), -1)

    ## indices for agent 1's channel matrix
    agent_1_idx = jnp.ix_(
        agent_state_adr[1],
        jnp.arange(0, horizon[1]),
        agent_ctrl_adr[1]
        )
    
    ## indices for agent 1's interference matrices
    noise_1_idx = jnp.ix_(
        agent_state_adr[1],
        jnp.arange(0, horizon[1]),
        agent_ctrl_adr[0]
        )
    
    F_1_agent = F_1[agent_1_idx].reshape(len(agent_state_adr[1]), -1)
    F_1_noise = F_1[noise_1_idx].reshape(len(agent_state_adr[1]), -1)

    ## pad by the missing time indexes for agent 1's planning horizon
    F_0_noise = jnp.pad(F_0_noise, ((0, 0), (0, pad[0])))
    ## pad by the missing time indexes for agent 0's planning horizon
    F_1_noise = jnp.pad(F_1_noise, ((0, 0), (0, pad[1])))

    iterations = 10
    # e_hist = jnp.zeros((iterations, 2))

    for i in range(iterations):

        S_0_noise = jnp.pad(S_0, ((0, pad[0]), (0, pad[0])))
        S_1_noise = jnp.pad(S_1, ((0, pad[1]), (0, pad[1])))

        noise_0 = F_0_noise @ S_1_noise @ F_0_noise.T + S_z_0
        noise_1 = F_1_noise @ S_0_noise @ F_1_noise.T + S_z_1

        e_0, S_0_ = waterfilling_operator(F_0_agent, noise_0, power[0])
        S_0 = alpha * S_0 + (1 - alpha) * S_0_

        e_1, S_1_ = waterfilling_operator(F_1_agent, noise_1, power[1])
        S_1 = alpha * S_1 + (1 - alpha) * S_1_

        e = jnp.concatenate([e_0, e_1])
        # e_hist = e_hist.at[i].set(e)

    return e

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(42)
    alpha = 0.01
    sim_horizon = 200
    ## horizon length for each agent
    horizon = jnp.array([
        100,
        100
    ])
    pad = horizon.max() - horizon
    assert sim_horizon >= horizon.max()
    power = jnp.array([1.0, 1.0])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {sim_horizon}')

    dx = dyn.state_dim
    du = dyn.control_dim

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((sim_horizon, dyn.control_dim))

    x0 = dyn.init_state()
    x0 = x0.at[0].set(3.14)
    x0 = x0.at[1].set(3.14)
    print(x0)

    linked_pendulum_empowerment = jax.jit(
        linked_pendulum_empowerment,
        static_argnums = 0
    )

    e = linked_pendulum_empowerment(dyn, x0, U, horizon, power, alpha)
    print(e)