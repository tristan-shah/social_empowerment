from pathlib import Path
from argparse import ArgumentParser
import time

import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
from einops import einsum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from soc_emp import Dynamics
from soc_emp.dynamics import make_step, make_unroll
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling
from soc_emp.utils import smooth_angle_wrap

VALID_STATE_TYPES = ['angle', 'velocity', 'full']
VALID_CONTROL_TYPES = ['egoistic', 'ave']

def set_tendon_properties(dyn: Dynamics, stiffness: float, damping: float):
    '''
    Function for overriding the default tendon properties in the linked pendulum scenario.
    The tendon properties control the coupling between the agents
    '''

    ## setting the properties of the tendon
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[:].set(stiffness),
        tendon_damping = dyn.mjx_model.tendon_damping.at[:].set(damping)
    )

    return dyn


def make_compute_group_empowerment(step: callable, state_matrix: Array, U: Array, alpha: float, observation_noise: float):
    '''
    Factory function for computing the empowerment of a group of agents.
    '''

    ## takes jacobians along a trajectory of states and controls
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    ## unrolls a trajectory
    unroll = jax.jit(make_unroll(step))

    ## extract relevant shapes
    num_agents, state_dim = state_matrix.shape
    horizon, total_control_dim = U.shape

    ## total control dimention is split among the agents
    agent_control_dim = total_control_dim // num_agents
    ## length of the message from each agent is its control dim times the horizon
    message_dim = horizon * agent_control_dim

    ## this is the observation covariance matrix for each agent
    S_z = jnp.eye(state_dim) * observation_noise 

    def compute_group_empowerment(xt: Array, power_density: Array):

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

        ## total probing power depends on horizon
        power = horizon * power_density

        ## this is the initial covariance matrix for each agent.
        ## total probing power is spread evenly along the diagonal
        S = jax.vmap(jnp.diag)(power[:, None] * jnp.ones((num_agents, message_dim)) / message_dim)

        ## calculate iterative water-filling
        _, e, _ = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)

        return e
    
    return compute_group_empowerment


def build_linked_pendulum_state_matrix(state_type: str):

    assert state_type in VALID_STATE_TYPES
    
    if state_type == 'angle':
        state_matrix = jnp.array([[0], [1]])

    elif state_type == 'velocity':
        state_matrix = jnp.array([[2], [3]])

    elif state_type == 'full':
        state_matrix = jnp.array([[0, 2], [1, 3]])
    
    return state_matrix


def compute_egoistic_control(grad_e: Array, control_gain: Array, power_density: Array):
    empowerment_control_sensitivity = grad_e @ control_gain
    ut = jnp.sign(jnp.diag(empowerment_control_sensitivity)) * power_density
    return ut


def compute_ave_control(grad_e: Array, control_gain: Array, power_density: Array):
    '''
    All agents increase the empowerment of agent 0
    '''
    empowerment_control_sensitivity = grad_e[0] @ control_gain
    ut = jnp.sign(empowerment_control_sensitivity) * power_density
    return ut


compute_control_dict = {
    'ave': compute_ave_control,
    'egoistic': compute_egoistic_control
}


def make_run(dyn: Dynamics, steps: int, state_type: str, control_type: str, horizon: int, alpha: float, observation_noise: float):

    assert state_type in VALID_STATE_TYPES
    assert control_type in VALID_CONTROL_TYPES

    ## state indexes of each agent (agent x state)
    state_matrix = build_linked_pendulum_state_matrix(state_type)

    compute_control = compute_control_dict[control_type]
    
    ## nominal control sequence
    control_dim = dyn.control_dim
    U = jnp.zeros((horizon, control_dim))

    ## build functions
    step = make_step(dyn)
    compute_control_gain = jax.jit(jax.jacfwd(step, argnums = 1))
    compute_group_empowerment = jax.jit(make_compute_group_empowerment(step, state_matrix, U, alpha, observation_noise))
    compute_group_empowerment_grad = jax.jit(jax.jacfwd(compute_group_empowerment))

    def run(power_density: Array, key):

        ## get initial state
        x0 = jnp.zeros(dyn.state_dim)

        ## --- First action: randomly ±power per dimension ---
        key, subkey = jax.random.split(key)
        random_signs = jax.random.choice(subkey, jnp.array([-1, 1]), shape=(control_dim,))
        e0 = compute_group_empowerment(x0, power_density)
        u0 = power_density * random_signs
        x1 = step(x0, u0)

        def _step_linked_pendulums(xt, _):

            ## obtain control gain
            control_gain = compute_control_gain(xt, U[0])
            # e = compute_group_empowerment(xt, power_density)
            grad_e = compute_group_empowerment_grad(xt, power_density)
            ut = compute_control(grad_e, control_gain, power_density)

            ## propagate dynamics
            xt = step(xt, ut)
            # return xt, (xt, e)
            return xt, xt
        
        # ## scan for the remaining (steps-1) timesteps
        # _, (X, E) = jax.lax.scan(_step_linked_pendulums, x1, length = steps-1)
        # X_full = jnp.concatenate([x0[None, :], x1[None, :], X])
        # E_full = jnp.concatenate([e0[None, :], E])

        # ## concatenate trajectory: initial state, after random action, then rest
        # return X_full, E_full


        ## scan for the remaining (steps-1) timesteps
        _, X = jax.lax.scan(_step_linked_pendulums, x1, length = steps-1)
        X_full = jnp.concatenate([x0[None, :], x1[None, :], X])
        return X_full


    return jax.jit(run)


def get_linked_pendulum_outcome(traj: Array):
    '''
    absolute value of the angle from the top should be less than 1 rad.
    angular velocity should be less than 2 rad / sec.
    '''

    ## check angle from the bottom (0.0 rad). top is jnp.pi rad
    left_angle_from_bottom = jnp.abs(smooth_angle_wrap(traj[:, 0]))
    right_angle_from_bottom = jnp.abs(smooth_angle_wrap(traj[:, 1]))

    ## get final state
    left_up = jnp.all(left_angle_from_bottom[-50:] >= 2.1)
    right_up = jnp.all(right_angle_from_bottom[-50:] >= 2.1)

    neither_up = jnp.logical_and(jnp.logical_not(left_up), jnp.logical_not(right_up))

    outcome = jnp.where(neither_up, 0,
                jnp.where(jnp.logical_and(left_up, jnp.logical_not(right_up)), 1,
                jnp.where(jnp.logical_and(jnp.logical_not(left_up), right_up), 2,
                3)))

    return outcome


def plot_outcome_hetamap(m: Array, horizon: int, powers: Array, dt: float, path: str):
        
    # Custom colormap and norm
    colors = ['lightgray', 'blue', 'orange', 'green']
    labels = ['Neither', 'Left', 'Right', 'Both']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(m, cmap = cmap, norm = norm, origin = 'lower')

    # Convert powers to numpy and determine tick spacing
    powers_np = np.array(powers)
    full_tick_positions = np.arange(len(powers))
    tick_spacing = 4  # show every 4th tick (adjust as needed)

    # Select every nth tick
    sparse_tick_positions = full_tick_positions[::tick_spacing]
    sparse_tick_labels = np.round(powers_np[::tick_spacing], 2)

    # Set sparse ticks and labels
    ax.set_xticks(sparse_tick_positions)
    ax.set_xticklabels(sparse_tick_labels, rotation=90)
    ax.set_yticks(sparse_tick_positions)
    ax.set_yticklabels(sparse_tick_labels)

    ax.set_xlabel('Right Agent Power')
    ax.set_ylabel('Left Agent Power')
    ax.set_title(f'Horizon = {horizon * dt} (seconds)')

    # Add colorbar with custom ticks/labels
    cbar = plt.colorbar(img, ax = ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Pendulum Up')

    plt.tight_layout()
    fig.savefig(path, dpi = 300)
    plt.close(fig)
    return


def main(args):

    num_devices = jax.device_count()
    print(f'GPU devices: {num_devices}')
    print(f'Horizon = {args.horizon}')
    print(f'Alpha = {args.alpha}')
    print(f'Noise = {args.observation_noise}')
    print(f'Batch Size = {args.device_batch_size}')
    print(f'State Type = {args.state_type}')
    print(f'Control Type = {args.control_type}')

    key = jax.random.key(args.seed)

    ## sweep power levels
    # power_density_levels = jnp.linspace(0.5, 3.0, args.resolution)
    power_density_levels = jnp.linspace(args.min_power, 3.0, args.resolution)

    name = f'state_type={args.state_type}-horizon={args.horizon}-alpha={args.alpha}-observation_noise={args.observation_noise}-stiffness={args.stiffness}-damping={args.damping}-steps={args.steps}-min_power={args.min_power}'
    output_dir = Path(f'results/linked_pendulum/control_type={args.control_type}') / name
    output_dir.mkdir(parents = True, exist_ok = True)

    ## create indexes for the heatmap
    I, J = jnp.meshgrid(jnp.arange(args.resolution), jnp.arange(args.resolution), indexing='ij')
    ## all possible combinations of powers
    pairs = jnp.stack([power_density_levels[I], power_density_levels[J]], axis=-1).reshape(-1, 2)
    ## compute number of batches, effective_batch_size
    effective_batch_size = args.device_batch_size * num_devices
    num_pairs = pairs.shape[0]
    num_batches = (num_pairs + effective_batch_size - 1) // effective_batch_size


    print(f'GPU devices: {jax.devices()}')

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path, dt = args.dt)
    dyn = set_tendon_properties(dyn, args.stiffness, args.damping)

    run = make_run(dyn, args.steps, args.state_type, args.control_type, args.horizon, args.alpha, args.observation_noise)
    vmap_run = jax.jit(jax.vmap(run))
    ## executes vmap_run in parallel on all available devices
    pmap_run = jax.pmap(vmap_run)

    ## create a function to evaluate the outcome of a batch of linked_pendulum runs
    batch_get_linked_pendulum_outcome = jax.vmap(get_linked_pendulum_outcome)

    ## initialize empty outcome heatmap
    outcomes = jnp.zeros((args.resolution, args.resolution))

    ## begin sweep
    print(f'Starting sweep at {time.ctime()}')
    start_time = time.time()

    for batch_idx in range(num_batches):

        # split once per batch, advance key
        key, subkey = jax.random.split(key)

        ## obtain the batch indices
        start_idx = batch_idx * effective_batch_size
        end_idx = min((batch_idx + 1) * effective_batch_size, num_pairs)
        actual_effective_batch_size = end_idx - start_idx

        ## extract the batch of power levels
        batch_pairs = pairs[start_idx:end_idx]

        ## pads the pairs and keys if the actual batch size is less than the expected batch size
        if actual_effective_batch_size < effective_batch_size:
            pad_size = effective_batch_size - actual_effective_batch_size
            batch_pairs = jnp.vstack([batch_pairs, jnp.ones((pad_size, 2))])

        ## create keys for each power pair in the batch
        batch_keys = jax.random.split(subkey, effective_batch_size)

        ## reshape so that the leading index is num_devices so pmap broadcasts correctly
        batch_pairs = batch_pairs.reshape(num_devices, args.device_batch_size, 2)
        batch_keys = batch_keys.reshape(num_devices, args.device_batch_size)

        ## run in parallel
        print()
        print(f'Starting running batch {batch_idx}')
        batch_start = time.time()
        batch_X = pmap_run(batch_pairs, batch_keys)
        batch_time = time.time() - batch_start
        print(f'Finished running batch {batch_idx}')

        ## evaluate outcomes
        print()
        print(f'Starting batch evaluation {batch_idx}')
        batch_X = batch_X.reshape(effective_batch_size, args.steps + 1, dyn.state_dim)
        ## evaluate the outcome of each simulation in the batch
        batch_outcomes = batch_get_linked_pendulum_outcome(batch_X)

        batch_outcomes = batch_outcomes.reshape(-1)[:actual_effective_batch_size]
        batch_I = I.reshape(-1)[start_idx:end_idx]
        batch_J = J.reshape(-1)[start_idx:end_idx]
        outcomes = outcomes.at[batch_I, batch_J].set(batch_outcomes)
        batch_pairs = batch_pairs.reshape(effective_batch_size, 2)[:actual_effective_batch_size]
        print(f'Finished batch evaluation {batch_idx}')

        ## save results
        print()
        print(f'Starting saving batch {batch_idx}')
        ## save trajectories, outcomes, and power pairs
        jnp.save(output_dir / f'X_batch_{batch_idx}.npy', batch_X)
        jnp.save(output_dir / f'outcomes_batch_{batch_idx}.npy', batch_outcomes)
        jnp.save(output_dir / f'pairs_batch_{batch_idx}.npy', batch_pairs)
        print(f'Finished saving batch {batch_idx}')

        ## report time
        print()
        print(f'Batch {batch_idx + 1}/{num_batches} ({end_idx}/{num_pairs} simulations) took {batch_time:.2f} seconds, saved X, outcomes, and pairs')

        plot_outcome_hetamap(outcomes, args.horizon, power_density_levels, dt = args.dt, path = output_dir / 'outcome_heatmap.png')

    ## save final outcomes and powers
    jnp.save(output_dir / 'outcomes.npy', outcomes)
    jnp.save(output_dir / 'powers.npy', power_density_levels)
    plot_outcome_hetamap(outcomes, args.horizon, power_density_levels, dt = dyn.mjx_model.opt.timestep, path = output_dir / 'outcome_heatmap.png')
    print(f'Completed sweep at {time.ctime()}, total time {time.time() - start_time:.2f} seconds')



    return


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--steps', type = int, default = 2000)
    parser.add_argument('--alpha', type = float, default = 0.01)
    parser.add_argument('--horizon', type = int, default = 100)
    parser.add_argument('--observation_noise', type = float, default = 1.0)
    parser.add_argument('--stiffness', type = float, default = 3.0)
    parser.add_argument('--damping', type = float, default = 0.1)
    parser.add_argument('--state_type', type = str, choices = VALID_STATE_TYPES, default = 'angle')
    parser.add_argument('--control_type', type = str, choices = VALID_CONTROL_TYPES, default = 'egoistic')
    parser.add_argument('--dt', type = float, default = 0.01)
    
    parser.add_argument('--device_batch_size', type = int, default = 50)
    parser.add_argument('--resolution', type = int, default = 100)
    parser.add_argument('--min_power', type = float, default = 0.1)
    args = parser.parse_args()

    main(args)