from argparse import ArgumentParser
from pathlib import Path
import json

import jax
from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

from soc_emp.dynamics import make_unroll
from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.plot import render_image, render_video
from soc_emp.envs.flock.utils import build_flock_state_matrix, compute_order_parameter
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling
from soc_emp.utils import dict_to_string


def make_compute_group_empowerment(step: callable, state_matrix: Array, U: Array, power_density: Array, alpha: float, observation_noise: float):

    ## build helper functions
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step, stochastic = True)

    ## extract relevant shapes
    horizon = U.shape[0]
    power = horizon * power_density ## total probing power depends on horizon
    num_agents = len(power_density)
    total_control_dim = U.shape[1] ## total dimention of control
    agent_control_dim = total_control_dim // num_agents
    message_dim = horizon * agent_control_dim ## this is the length of the message from each agent

    ## This is the initial covariance matrix for each agent. Assume diagonal
    S = jax.vmap(jnp.diag)(power[:, None] * jnp.ones((num_agents, message_dim)) / message_dim)
    ## this is the observation covariance matrix for each agent. Assume identity scaled by a scalar
    S_z = jnp.eye(state_matrix.shape[1]) * observation_noise 

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

from einops import einsum
from soc_emp.empowerment import waterfilling_implicit, compute_power

def make_compute_empowerment(step: callable, U: Array, power_density: float):

    ## build helper functions
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step, stochastic = True)


    ## extract relevant shapes
    horizon = U.shape[0]
    power = horizon * power_density ## total probing power depends on horizon

    def compute_empowerment(xt: Array, key):

        keys = jax.random.split(key, horizon)

        X = unroll(xt, U, keys)
        A, B = linearize(X[:-1], U, keys)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))

        ## S is the covariance matrix of the final state.
        S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
        h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
        v = waterfilling_implicit(h2, power)
        p = compute_power(v, h2)
        e = 0.5 * jnp.sum(jnp.log(1 + p * h2))

        return e

    return jax.jit(compute_empowerment)


def main(args):

    ## hardcode leader
    LEADER = 0
    
    ## extracting the parameters into a dictionary
    params = vars(args)
    run_name = dict_to_string(params)

    ## build paths
    root = Path('results') / 'Vicsek'
    output_dir = root / run_name

    ## build the directory
    output_dir.mkdir(parents = True, exist_ok = True)
    
    ## save the parameters to a json
    with open(output_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent = 2, sort_keys = True)

    ## make a key
    key = jax.random.key(args.seed)

    state_type = 'angle' ## this is the only numerically stable state_type
    state_matrix = build_flock_state_matrix(args.num_agents, state_type)

    ## empowerment arguments
    power_density = args.power_density * jnp.ones(args.num_agents)
    # power_density = power_density.at[args.num_agents // 2:].set(0.5 * args.power_density)
    
    flock = Vicsek(args.num_agents, args.grid_size, args.radius, args.speed, args.J, args.D)
    reset = make_reset(flock)
    step = make_step(flock)
    U = jnp.zeros((args.horizon, flock.control_dim))

    xt = reset(key)

    ## save an image of the initial state
    render_image(xt, flock, show_radius = True).savefig(output_dir / 'initial_state.png', dpi = 300)

    compute_empowerment = make_compute_empowerment(step, U, args.power_density)
    compute_empowerment_grad = jax.jit(jax.jacfwd(compute_empowerment))

    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, power_density, args.alpha, args.observation_noise)
    compute_group_empowerment_grad = jax.jit(jax.jacfwd(compute_group_empowerment))
    control_gain = jax.jit(jax.jacfwd(step, argnums = 1))

    ## track metrics
    empowerment_hist = jnp.zeros((args.steps, flock.num_agents))
    order_parameter_hist = jnp.zeros(args.steps)

    ## track states
    X = jnp.zeros((args.steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(args.steps):
        key, subkey = jax.random.split(key)

        ## select action
        e = compute_group_empowerment(xt, subkey)
        grad_e = compute_group_empowerment_grad(xt, subkey)
        B = control_gain(xt, U[0], subkey)

        ## select action based on chosen behavior
        if args.behavior == 'leader':
            ut = jnp.sign(grad_e[LEADER, :] @ B) * power_density

        elif args.behavior == 'feedback':

            ## assume LEADER = 0
            ut_leader = jnp.sign(grad_e[1:].sum(axis = 0) @ B[:, LEADER]) * power_density[LEADER]
            ut_flock = jnp.sign(grad_e[LEADER, :] @ B[:, 1:]) * power_density[1:]
            ut = jnp.concat([ut_leader[None], ut_flock])

        elif args.behavior == 'egoistic':
            ut = jnp.sign(jnp.diag(grad_e @ B)) * power_density

        elif args.behavior == 'collective':
            ut = jnp.sign(jnp.sum(grad_e, axis = 0) @ B) * power_density

        elif args.behavior == 'passive':
            ut = jnp.zeros(flock.control_dim)

        elif args.behavior == 'vanilla':
            grad_e = compute_empowerment_grad(xt, subkey) ## compute the standard empowerment gradient
            ut = jnp.sign(grad_e @ B) * power_density

        ## step forward the dynamics
        key, subkey = jax.random.split(key)
        xt = step(xt, ut, subkey)

        print(t, ut, e)

        order_parameter_hist = order_parameter_hist.at[t].set(compute_order_parameter(xt, args.num_agents))
        empowerment_hist = empowerment_hist.at[t].set(e)
        X = X.at[t+1].set(xt)


    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Empowerment (nats)')
    for agent in range(flock.num_agents):
        ax.plot(empowerment_hist[:, agent])
    fig.tight_layout()
    fig.savefig(output_dir / 'empowerment.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Order Parameter')
    ax.plot(order_parameter_hist)
    fig.tight_layout()
    fig.savefig(output_dir / 'order_parameter.png', dpi=300)
    plt.close(fig)

    ## save raw data
    jnp.save(output_dir / 'empowerment_hist.npy', empowerment_hist)
    jnp.save(output_dir / 'order_parameter_hist.npy', order_parameter_hist)
    jnp.save(output_dir / 'trajectory.npy', X)

    if args.behavior not in ['leader', 'feedback']:
        LEADER = None
    
    render_video(X, flock, path = output_dir / 'vid.mp4', leader = LEADER)

    return None


if __name__ == '__main__':

    BEHAVIORS = ['leader', 'egoistic', 'passive', 'vanilla', 'feedback', 'collective']

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--steps', type = int, default = 1000, help = 'Simulation timesteps')

    ## Vicsek parameters
    parser.add_argument('--num_agents', type = int, default = 100)
    parser.add_argument('--grid_size', type = float, default = 5.0)
    parser.add_argument('--radius', type = float, default = 0.5, help = 'Falloff radius for each agent')
    parser.add_argument('--speed', type = float, default = 1.0, help = 'Speed of each bird')
    parser.add_argument('--J', type = float, default = 0.1, help = 'How agressively to align birds')
    parser.add_argument('--D', type = float, default = 0.0, help = 'Intensity of noise')

    ## empowerment parameters
    parser.add_argument('--horizon', type = int, default = 5, help = 'Planning horizon')
    parser.add_argument('--power_density', type = float, default = 2.0)
    parser.add_argument('--alpha', type = float, default = 0.01, help = 'IWF smoothing')
    parser.add_argument('--observation_noise', type = float, default = 1.0)
    parser.add_argument('--behavior', type = str, choices = BEHAVIORS, default = 'egoistic')

    args = parser.parse_args()

    main(args)


