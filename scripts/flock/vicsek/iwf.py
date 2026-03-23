import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.dynamics import make_unroll
from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.utils import build_flock_state_matrix
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, waterfilling_operator


if __name__ == '__main__':

    seed = 0
    key = jax.random.key(seed)
    num_agents = 100
    grid_size = 2.0
    neighbor_radius = 4.0
    speed = 1.0
    J = 0.1
    D = 0.0
    dt = 0.05

    alpha = 0.01
    observation_noise = 0.01 #1.0
    horizon = 10
    power_density = jnp.ones(num_agents)
    power = horizon * power_density ## total probing power depends on horizon

    state_type = 'full' ## this is the only numerically stable state_type
    state_matrix = build_flock_state_matrix(num_agents, state_type)

    flock = Vicsek(num_agents, grid_size, neighbor_radius, speed, J, D, dt)

    agent_control_dim = flock.control_dim // num_agents
    message_dim = horizon * agent_control_dim ## this is the length of the message from each agent



    reset = make_reset(flock)
    step = make_step(flock)
    unroll = make_unroll(step, stochastic = True)
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))

    xt = reset(key)
    U = jnp.zeros((horizon, flock.control_dim))


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


    print(F_agent.shape, F_noise.shape)


    # ## This is the initial covariance matrix for each agent. Assume diagonal
    S = jax.vmap(jnp.diag)(power[:, None] * jnp.ones((num_agents, message_dim)) / message_dim)
    # ## this is the observation covariance matrix for each agent. Assume identity scaled by a scalar
    S_z = jnp.eye(state_matrix.shape[1]) * observation_noise 


    iterations = 10
    e_hist = jnp.zeros((iterations, num_agents))

    for i in range(iterations):

        e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_

        e_hist = e_hist.at[i].set(e)

    fig, ax = plt.subplots(1, 1)
    
    for agent in range(num_agents):
        ax.plot(e_hist[:, agent])

    fig.savefig('iwf.png', dpi = 300)
    plt.show()