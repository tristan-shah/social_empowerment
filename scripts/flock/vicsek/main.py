import jax
from jax import Array
from jax import numpy as jnp
from jax.experimental import sparse

import matplotlib.pyplot as plt

from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.plot import render_image, render_video
from soc_emp.envs.flock.utils import build_flock_state_matrix

def make_unroll(step: callable):
    '''
    Unrolls dynamical system with randomness
    '''

    def unroll(x0: Array, U: Array, keys: Array):

        def body_fn(x: Array, inputs: tuple):

            u, key = inputs
            x_next = step(x, u, key)

            return x_next, x_next
        
        _, X = jax.lax.scan(body_fn, init = x0, xs = (U, keys))
        X = jnp.concatenate([x0[None, :], X], axis = 0)

        return X
    
    return jax.jit(unroll)


from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling

def make_compute_group_empowerment(step: callable, state_matrix: Array, U: Array, power_density: Array, alpha: float, observation_noise: float):

    ## build helper functions
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    ## extract relevant shapes
    horizon = U.shape[0]
    power = horizon * power_density ## total probing power depends on horizon
    num_agents = len(power_density)
    total_control_dim = U.shape[1] ## total dimention of control
    agent_control_dim = total_control_dim // num_agents
    message_dim = horizon * agent_control_dim ## this is the length of the message from each agent

    ## this is the initial covariance matrix for each agent.
    S = jax.vmap(jnp.diag)(power[:, None] * jnp.ones((num_agents, message_dim)) / message_dim)
    S_z = jnp.eye(state_matrix.shape[1]) * observation_noise ## this is the observation covariance matrix for each agent

    def compute_group_empowerment(xt: Array, key):

        keys = jax.random.split(key, horizon)

        X = unroll(xt, U, keys)
        A, B = linearize(X[:-1], U, keys)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))
        F_agent, F_noise = split_channel_matrix(F, num_agents)

        ## selects agent's own state from the big sensitivity matrix
        F_agent = jnp.take_along_axis(
            F_agent,
            state_matrix[:, :, None],
            axis = 1
        )

        ## noise in the agents state comes from the actions of other agents
        F_noise = jnp.take_along_axis(
            F_noise,
            state_matrix[:, None, :, None],
            axis = 2
        )

        _, e, _ = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)

        return e

    return jax.jit(compute_group_empowerment)


if __name__ == '__main__':
    seed = 10
    key = jax.random.key(seed)
    num_agents = 30
    grid_size = 5.0 #10.0
    neighbor_radius = 0.5
    speed = 1.0
    J = 0.1
    D = 0.0 #0.1
    steps = 1000
    horizon = 5
    state_type = 'angle'
    state_matrix = build_flock_state_matrix(num_agents, state_type)

    ## empowerment arguments
    power_density = 2.0 * jnp.ones(num_agents)
    alpha = 0.01
    observation_noise = 1.0

    
    flock = Vicsek(num_agents, grid_size, neighbor_radius, speed, J, D)
    reset = make_reset(flock)
    step = make_step(flock)
    U = jnp.zeros((horizon, flock.control_dim))

    xt = reset(key)

    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, power_density, alpha, observation_noise)
    compute_group_empowerment_grad = jax.jit(jax.jacfwd(compute_group_empowerment))
    control_gain = jax.jit(jax.jacfwd(step, argnums = 1))


    X = jnp.zeros((steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(steps):
        key, subkey = jax.random.split(key)

        ## select action
        e = compute_group_empowerment(xt, subkey)
        grad_e = compute_group_empowerment_grad(xt, subkey)
        B = control_gain(xt, U[0], subkey)
        ut = jnp.sign(jnp.diag(grad_e @ B)) * power_density

        key, subkey = jax.random.split(key)

        xt = step(xt, ut, subkey)

        print(t, ut, e)
        X = X.at[t+1].set(xt)
    
    render_video(X, flock, path = 'vid.mp4')