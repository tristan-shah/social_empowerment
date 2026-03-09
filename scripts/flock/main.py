import os
import json
from argparse import ArgumentParser
from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

from soc_emp.envs.flock import Flock, make_reset, make_step, render, render_video, decode_state, make_compute_order_parameter
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
    


def make_run_dir(args) -> str:
    """Create and return a results directory path based on run parameters."""
    run_name = (
        f"behavior{args.behavior}"
        f"_agents{args.agents}"
        f"_grid{args.grid_size}"
        f"_speed{args.speed}"
        f"_radius{args.radius}"
        f"_dt{args.dt}"
        f"_horizon{args.horizon}"
        f"_power{args.power_density}"
        f"_alpha{args.alpha}"
        f"_noise{args.observation_noise}"
        f"_steps{args.steps}"
        f"_seed{args.seed}"
    )
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok = True)
    return run_dir
    
def main(args):

    run_dir = make_run_dir(args)

    # Save args for reproducibility
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ## hardcoded leader index
    leader = 0

    key = jax.random.key(args.seed)

    # flock = Flock(args.agents, args.grid_size, args.speed, args.radius, args.falloff, args.dt)
    flock = Flock(args.agents, args.grid_size, args.speed, args.radius, args.dt)

    reset = make_reset(flock)
    step = make_step(flock)
    control_gain = jax.jit(jax.jacfwd(step, argnums = 1))
    compute_order_parameter = make_compute_order_parameter(flock)

    xt = reset(key)

    if args.behavior == 'leader':
        render(xt, flock, show_radius=True, leader = leader).savefig(os.path.join(run_dir, 'radius.png'), dpi=300)
    else:
        render(xt, flock, show_radius=True).savefig(os.path.join(run_dir, 'radius.png'), dpi=300)

    U = jnp.zeros((args.horizon, flock.control_dim))

    state_matrix = build_flock_state_matrix(flock, 'angle')

    power_density = 2 * jnp.ones(flock.num_agents)
    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, power_density, args.alpha, args.observation_noise)
    compute_group_empowerment_grad = jax.jit(jax.jacfwd(select_output(compute_group_empowerment, 1)))


    empowerment_hist = jnp.zeros((args.steps, flock.num_agents))
    order_parameter_hist = jnp.zeros(args.steps)

    X = jnp.zeros((args.steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(args.steps):
    
        _, e, _ = compute_group_empowerment(xt)

        if args.behavior == 'leader':
                
            grad_e = compute_group_empowerment_grad(xt)
            B = control_gain(xt, U[0])
            ut = jnp.sign(grad_e[leader, :] @ B) * args.power_density

        elif args.behavior == 'egoistic':

            grad_e = compute_group_empowerment_grad(xt)
            B = control_gain(xt, U[0])
            ut = jnp.sign(jnp.diag(grad_e @ B)) * args.power_density

        elif args.behavior == 'passive':
            ut = jnp.zeros(flock.control_dim)
    

        ## propagate dynamics
        xt = step(xt, ut)

        print(t, e, ut)

        order_parameter_hist = order_parameter_hist.at[t].set(compute_order_parameter(xt))
        empowerment_hist = empowerment_hist.at[t].set(e)
        X = X.at[t+1].set(xt)
    
    fig, ax = plt.subplots(1, 1)
    for agent in range(flock.num_agents):
        ax.plot(empowerment_hist[:, agent])
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'empowerment.png'), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(order_parameter_hist)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'order_parameter.png'), dpi=300)
    plt.close(fig)

    ## save raw data
    jnp.save(os.path.join(run_dir, 'empowerment_hist.npy'), empowerment_hist)
    jnp.save(os.path.join(run_dir, 'order_parameter_hist.npy'), order_parameter_hist)
    jnp.save(os.path.join(run_dir, 'trajectory.npy'), X)

    ## make the video
    if args.behavior == 'leader':
        render_video(X, flock, fps=60, save_path=os.path.join(run_dir, 'flock.mp4'), dpi=150, leader = leader)
    else:
        render_video(X, flock, fps=60, save_path=os.path.join(run_dir, 'flock.mp4'), dpi=150)

    return None

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)

    ## flock parameters
    parser.add_argument('--agents', type = int, default = 100)
    parser.add_argument('--grid_size', type = float, default = 10.0)
    parser.add_argument('--speed', type = float, default = 1.0)
    parser.add_argument('--radius', type = float, default = 1.0)
    parser.add_argument('--dt', type = float, default = 0.05)

    ## empowerment parameters
    parser.add_argument('--horizon', type = int, default = 5)
    parser.add_argument('--power_density', type = float, default = 2.0)
    parser.add_argument('--alpha', type = float, default = 0.01)
    parser.add_argument('--observation_noise', type = float, default = 1.0)
    parser.add_argument('--steps', type = int, default = 500)

    parser.add_argument('--behavior', type = str, choices = ['passive', 'egoistic', 'leader'], default = 'leader')

    args = parser.parse_args()

    main(args)