import jax
from jax import Array
from jax import numpy as jnp

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix

def build_linked_pendulum_channel_matrix(A: Array, B: Array, horizon: tuple[int, int]):

    ## how much to pad
    pad_0 = A.shape[0] - horizon[0]
    pad_1 = A.shape[0] - horizon[1]

    ## compute the sensitivity of the final state in the horizon to each action
    F_0 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, horizon[0]),
        jax.lax.dynamic_slice_in_dim(B, 0, horizon[0]))
    
    F_1 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, horizon[1]),
        jax.lax.dynamic_slice_in_dim(B, 0, horizon[1]))
    
    ## swap the indices so the state dimention is first
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon[0] x control)
    F_1 = jnp.permute_dims(F_1, (1, 0, 2)) ## (state x horizon[0] x control)

    ## split the channel matrix into the effect of each agents actions on each agents state
    F_0_agent, F_0_noise = split_channel_matrix(F_0, 2)
    F_1_agent, F_1_noise = split_channel_matrix(F_1, 2)

    # print(F_0_agent[0, [0, 2], :] == F_0_agent[0][[0, 2], :])
    # print(F_0_noise[0, :, [0, 2], :] == F_0_noise[0][:, [0, 2], :])

    print(F_1_noise[[1], :, [1, 3], :])
    print(F_1_noise[[1], :, [1, 3], :].shape)

    # ## sensitivity of agent i's action on its own state
    # F_agent = jnp.stack([
    #     jnp.pad(F_0_agent[0, [0, 2], :], ((0, 0), (0, pad_0))),
    #     jnp.pad(F_1_agent[1, [1, 3], :], ((0, 0), (0, pad_1)))
    # ])

    # F_noise = jnp.stack([
    #     jnp.pad(F_0_noise[0][:, [0, 2], :], ((0, 0), (0, 0), (0, pad_0))),
    #     jnp.pad(F_1_noise[1][:, [1, 3], :], ((0, 0), (0, 0), (0, pad_1)))
    # ])

    # return F_agent

# build_linked_pendulum_channel_matrix = jax.jit(build_linked_pendulum_channel_matrix, static_argnums = 2)

def compute_multiagent_empowerment(
        dyn: Dynamics, 
        x0: Array, 
        U: Array,
        horizon: Array,
        power: Array,
        alpha: float,
        observation_noise: float):
    
    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    build_linked_pendulum_channel_matrix(A, B, horizon)

    return

if __name__ == '__main__':

    ## system hyperparameters
    max_horizon = 6
    horizon = (3, 5)
    assert max(horizon) <= max_horizon
    power = jnp.array([1.5, 1.3])
    alpha = 0.50
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((max_horizon, dyn.control_dim))
    
    xt = dyn.init_state()
    xt = xt.at[0].set(3.1)
    xt = xt.at[1].set(3.0)

    # compute_multiagent_empowerment(dyn, xt, U, horizon, power, alpha, observation_noise)

    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    
    build_linked_pendulum_channel_matrix(A, B, horizon)