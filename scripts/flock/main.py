from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

from soc_emp.envs.flock import Flock, make_reset, make_step, render, render_video, decode_state
from soc_emp.dynamics import make_unroll

from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, select_output

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

    def compute_group_empowerment(xt: Array):

        X = unroll(xt, U)
        A, B = linearize(X[:-1], U)
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

        i, e, S_new = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)
        # i, e, S_new = iterative_waterfilling(F_agent, F_noise, S, S_z, power_density, alpha)

        return i, e, S_new

    return jax.jit(compute_group_empowerment)

def build_flock_state_matrix(flock: Flock, state_type: str):
    idx = jnp.arange(0, flock.num_agents * 3)
    x, y, a = decode_state(idx, flock.num_agents)

    if state_type == 'pos':
        return jnp.stack([x, y], axis = 1)
    elif state_type == 'angle':
        return a[:, None]
    elif state_type == 'full':
        return jnp.stack([x, y, a], axis = 1)
    else:
        raise ValueError('Invalid state type.')

if __name__ == '__main__':
    seed = 10
    key = jax.random.key(seed)

    ## leader agent
    leader = 0

    ## flock parameters
    num_agents = 50
    grid_size = 10.0
    speed = 1.0
    neighbor_radius = 0.5 #10.0 # 5.0
    neighbor_falloff = 10.0
    dt = 0.05

    ## empowerment
    horizon = 5 #10 #20
    power_density = 2 * jnp.ones(num_agents)
    alpha = 0.01
    observation_noise = 1.0

    ## simulation steps
    steps = 500

    flock = Flock(num_agents, grid_size, speed, neighbor_radius, neighbor_falloff, dt)

    reset = make_reset(flock)
    step = make_step(flock)
    control_gain = jax.jit(jax.jacfwd(step, argnums = 1))

    xt = reset(key)

    render(xt, flock, show_radius = True).savefig('radius.png', dpi = 300)

    U = jnp.zeros((horizon, flock.control_dim))
    
    state_matrix = build_flock_state_matrix(flock, 'angle')
    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, power_density, alpha, observation_noise)
    compute_group_empowerment_grad = jax.jit(jax.jacfwd(select_output(compute_group_empowerment, 1)))

    print(compute_group_empowerment(xt))
    print(compute_group_empowerment_grad(xt))


    hist = jnp.zeros((steps, flock.num_agents))

    X = jnp.zeros((steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(steps):

        grad_e = compute_group_empowerment_grad(xt)
        i, e, S = compute_group_empowerment(xt)
        B = control_gain(xt, U[0])
        ut = jnp.sign(jnp.diag(grad_e @ B)) * power_density
        # ut = jnp.sign(grad_e[leader, :] @ B) * power_density
        # ut = jnp.sign(jnp.sum(grad_e @ B, axis = 0)) * power_density
        xt = step(xt, ut)

        print(t, e, ut)

        hist = hist.at[t].set(e)
        X = X.at[t+1].set(xt)
    
    fig, ax = plt.subplots(1, 1)
    for agent in range(flock.num_agents):
        ax.plot(hist[:, agent])

    fig.tight_layout()
    fig.savefig('empowerment.png', dpi = 300)
    plt.show()

    render_video(X, flock, fps = 60, save_path = "flock.mp4", dpi = 150)
