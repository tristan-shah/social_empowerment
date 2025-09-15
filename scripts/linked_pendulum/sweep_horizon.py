from argparse import ArgumentParser
from pathlib import Path
import time
from tqdm import tqdm
import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from soc_emp import Dynamics
# from soc_emp.empowerment import compute_multiagent_empowerment_grad
from variable_horizon import compute_multiagent_empowerment_grad
from soc_emp.utils import smooth_angle_wrap

'''
srun --jobid=... nvidia-smi
'''

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

def plot_outcome_hetamap(m: Array, power: float, horizons: Array, dt: float, path: str):
        
    # Custom colormap and norm
    colors = ['lightgray', 'blue', 'orange', 'green']
    labels = ['Neither', 'Left', 'Right', 'Both']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(m, cmap = cmap, norm = norm, origin = 'lower')

    # Convert powers to numpy and determine tick spacing
    horizons_np = np.array(horizons) * dt
    full_tick_positions = np.arange(len(horizons))
    tick_spacing = 5  # show every 4th tick (adjust as needed)

    # Select every nth tick
    sparse_tick_positions = full_tick_positions[::tick_spacing]
    sparse_tick_labels = horizons_np[::tick_spacing]

    # Set sparse ticks and labels
    ax.set_xticks(sparse_tick_positions)
    ax.set_xticklabels(sparse_tick_labels, rotation=90)
    ax.set_yticks(sparse_tick_positions)
    ax.set_yticklabels(sparse_tick_labels)

    ax.set_xlabel('Right Agent Horizon (s)')
    ax.set_ylabel('Left Agent Horizon (s)')
    ax.set_title(f'Power = {power}')

    # Add colorbar with custom ticks/labels
    cbar = plt.colorbar(img, ax = ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Pendulum Up')

    plt.tight_layout()
    fig.savefig(path, dpi = 300)
    plt.close(fig)
    return

def run_multiagent_empowerment(dyn: Dynamics, U: Array, horizon: Array, power: Array, alpha: float, observation_noise: float, steps: int, key):
    '''
    runs the multi agent empowerment controller where each agent selects an action proportional to the gradient
    of empowerment.
    '''

    ## obtain the initial zero state of the system
    xt = dyn.init_state()

    @jax.jit
    def _step_linked_pendulums(carry, _):
        _xt, _key = carry

        ## obtain control gain
        _, B = dyn.linearize(_xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, _xt, U, horizon, power, alpha, observation_noise)

        ## compute action
        ut = jnp.sign(jnp.diag(grad_E @ B)) * power

        ## pick a random direction with max power if the action is zero
        sub_key, _key = jax.random.split(_key)
        random_direction = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(ut.shape[0],))
        ut = ut + (ut == 0) * power * random_direction

        ## propagate dynamics
        _xt = dyn.step(_xt, ut)

        return ((_xt, _key), _xt)
    
    _, X = jax.lax.scan(_step_linked_pendulums, (xt, key), length = steps)
    return jnp.concatenate([xt[None, :], X])

## create a function to evaluate the outcome of a batch of linked_pendulum runs
batch_get_linked_pendulum_outcome = jax.vmap(get_linked_pendulum_outcome)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--power', type = float, default = 1.0)
    parser.add_argument('--alpha', type = float, default = 0.01)
    parser.add_argument('--observation_noise', type = float, default = 1.0)
    args = parser.parse_args()
    print(f'GPU devices: {jax.devices()}')

    ## Hyperparams
    key = jax.random.key(5)
    steps = 1500                                ## simulation horizon
    power = jnp.array([args.power, args.power])
    alpha = args.alpha                          ## smoothing for synchronous IWF
    observation_noise = args.observation_noise
    horizons = jnp.arange(50, 205, 5)
    device_batch_size = 50
    num_devices = jax.device_count()
    print(horizons)
    max_horizon = max(horizons)
    num_horizons = len(horizons)

    output_dir = Path(f'results/sweep_horizon/power={args.power}_alpha={alpha}_observation_noise={observation_noise}')
    output_dir.mkdir(parents = True, exist_ok = True)

    ## create a function that will execute run_multi_agent_empowerment on a batch of powers and keys 
    ## holding the number of simulation steps constant.
    batch_run_multiagent_empowerment = jax.jit(
        jax.vmap(
            lambda _dyn, _U, _horizon, _alpha, _key : run_multiagent_empowerment(_dyn, _U, _horizon, power, _alpha, observation_noise, steps, _key),
            in_axes = (None, None, 0, None, 0)
        ),
        static_argnums = 0
    )

    ## executes batch_run_multiagent_empowerment in parallel on all available devices
    pmap_batch_run_multiagent_empowerment = jax.pmap(
        batch_run_multiagent_empowerment,
        in_axes = (None, None, 0, None, 0),
        static_broadcasted_argnums = 0
    )

    ## Load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')

    # ## create zero control sequence
    U = jnp.zeros((max_horizon, dyn.control_dim))

    ## create indexes for the heatmap
    I, J = jnp.meshgrid(jnp.arange(num_horizons), jnp.arange(num_horizons), indexing='ij')
    ## all possible combinations of powers
    pairs = jnp.stack([horizons[I], horizons[J]], axis=-1).reshape(-1, 2)

    ## compute number of batches, effective_batch_size
    effective_batch_size = device_batch_size * num_devices
    num_pairs = pairs.shape[0]
    num_batches = (num_pairs + effective_batch_size - 1) // effective_batch_size

    def prepare_batch(start_idx: int, end_idx: int):

        '''
        A function to prepare a batch of simulation parameters for execution on multiple gpus.

        Args:
        start_idx: The starting index of simulations
        end_idx: The ending index of simulations

        Returns:
        batch_pairs_reshaped with size: (devices x device_batch_size x 2)
        batch_keys_reshaped with size: (devices x device_batch_size)
        '''

        ## the actual effective batch size might be smaller than the effective batch size
        actual_effective_batch_size = end_idx - start_idx
        batch_pairs = pairs[start_idx:end_idx]
        batch_keys = jax.random.split(key, actual_effective_batch_size)

        ## pads the pairs and keys if the actual batch size is less than the expected batch size
        if actual_effective_batch_size < effective_batch_size:
            pad_size = effective_batch_size - actual_effective_batch_size
            # batch_pairs = jnp.vstack([batch_pairs, jnp.zeros((pad_size, 2))])
            batch_pairs = jnp.vstack([batch_pairs, jnp.ones((pad_size, 2))])
            batch_keys = jnp.concatenate([batch_keys, jax.random.split(key, pad_size)])

        ## reshape so that the leading index is num_devices so pmap broadcasts correctly
        batch_pairs_reshaped = batch_pairs.reshape(num_devices, device_batch_size, 2)
        batch_keys_reshaped = batch_keys.reshape(num_devices, device_batch_size)
        return batch_pairs_reshaped, batch_keys_reshaped
    
    ## initialize empty heatmap
    outcomes = jnp.zeros((num_horizons, num_horizons))

    ## begin sweep
    print(f'Starting sweep at {time.ctime()}')
    start_time = time.time()

    for batch_idx in tqdm(range(num_batches), desc = f'Sweep (power={power})'):

        start_idx = batch_idx * effective_batch_size
        end_idx = min((batch_idx + 1) * effective_batch_size, num_pairs)
        actual_effective_batch_size = end_idx - start_idx
        batch_pairs, batch_keys = prepare_batch(start_idx, end_idx)

        ## run in parallel
        batch_start = time.time()
        batch_X = pmap_batch_run_multiagent_empowerment(dyn, U, batch_pairs, alpha, batch_keys)
        batch_time = time.time() - batch_start

        batch_X = batch_X.reshape(effective_batch_size, steps + 1, dyn.state_dim)
        ## evaluate the outcome of each simulation in the batch
        batch_outcomes = batch_get_linked_pendulum_outcome(batch_X)

        batch_outcomes = batch_outcomes.reshape(-1)[:actual_effective_batch_size]
        batch_I = I.reshape(-1)[start_idx:end_idx]
        batch_J = J.reshape(-1)[start_idx:end_idx]
        outcomes = outcomes.at[batch_I, batch_J].set(batch_outcomes)
        batch_pairs = batch_pairs.reshape(effective_batch_size, 2)[:actual_effective_batch_size]

        ## save trajectories, outcomes, and power pairs
        jnp.save(output_dir / f'X_batch_{batch_idx}.npy', batch_X)
        jnp.save(output_dir / f'outcomes_batch_{batch_idx}.npy', batch_outcomes)
        jnp.save(output_dir / f'pairs_batch_{batch_idx}.npy', batch_pairs)
        print(f'Batch {batch_idx + 1}/{num_batches} ({end_idx}/{num_pairs} simulations) took {batch_time:.2f} seconds, saved X, outcomes, and pairs')

    ## save final outcomes and powers
    jnp.save(output_dir / 'outcomes.npy', outcomes)
    jnp.save(output_dir / 'powers.npy', horizons)
    plot_outcome_hetamap(outcomes, args.power, horizons, dt = dyn.mjx_model.opt.timestep, path = output_dir / 'outcome_heatmap.png')
    print(f'Completed sweep at {time.ctime()}, total time {time.time() - start_time:.2f} seconds')