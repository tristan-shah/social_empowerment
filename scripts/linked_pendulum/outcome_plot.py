from pathlib import Path
import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from soc_emp import Dynamics
from soc_emp.utils import smooth_angle_wrap
# from sweep_power import plot_outcome_hetamap
from sweep_horizon import plot_outcome_hetamap


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

## create a function to evaluate the outcome of a batch of linked_pendulum runs
batch_get_linked_pendulum_outcome = jax.vmap(get_linked_pendulum_outcome)

def find_nearest_index(array: Array, values: Array):
    diffs = jnp.abs(array[:, None] - values[None, :])
    return jnp.argmin(diffs, axis = 0)

def load_horizon_data(power: float):

    assert power in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    power = jnp.array([power, power])
    name = 'power_times_horizon'
    path = Path(f'results/sweep_horizon/{name}/power={power}_alpha=0.01_observation_noise=1.0')
    batches = 29
    horizons = jnp.arange(50, 202, 2)
    num_horizons = len(horizons)

    outcomes = jnp.zeros((num_horizons, num_horizons))
    pair_map = jnp.zeros((num_horizons, num_horizons, 2))
    trajectories = jnp.zeros((num_horizons, num_horizons, 1501, dyn.state_dim))

    for batch in range(batches):
        print(batch)
 
        pairs = jnp.load(path / f'pairs_batch_{batch}.npy')
        X = jnp.load(path / f'X_batch_{batch}.npy')

        batch_I = find_nearest_index(horizons, pairs[:, 0])
        batch_J = find_nearest_index(horizons, pairs[:, 1])
        outcomes = outcomes.at[batch_I, batch_J].set(batch_get_linked_pendulum_outcome(X[0:pairs.shape[0]]))
        pair_map = pair_map.at[batch_I, batch_J].set(pairs)

        max_idx = pairs.shape[0]
        trajectories = trajectories.at[batch_I, batch_J].set(X[:max_idx])

    return outcomes, pair_map, trajectories



def load_power_data(horizon: int):

    assert horizon in [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    path = Path(f'results/sweep_power/seed=10_horizon={horizon}_alpha=0.01_observation_noise=1.0')

    powers = jnp.load(path / 'powers.npy')
    outcomes = jnp.load(path / 'outcomes.npy')

    pair_map = jnp.stack(jnp.meshgrid(powers, powers), axis = -1)
    return pair_map, outcomes


if __name__ == '__main__':
    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path=xml_path)
    print(f'Timestep = {dyn.mjx_model.opt.timestep}')


    # outcomes, pair_map, trajectories = load_horizon_data(1.1)
    pair_map, outcomes = load_power_data(180)

    left = pair_map[:, :, 0]
    right = pair_map[:, :, 1]
    diff = right - left

    unique_diff, count = jnp.unique_counts(diff)
    freq = jnp.zeros((count.shape[0], 4))

    for (i, d) in enumerate(unique_diff):
        mask = diff == d

        num_outcomes = jnp.sum(mask)

        for k in range(4):
            print(k)
            freq = freq.at[i, k].set(jnp.sum(outcomes[mask] == k) / num_outcomes)
            print(outcomes[mask])


    from scipy.ndimage import gaussian_filter1d

    # smooth each outcome curve
    freq_smooth = gaussian_filter1d(np.array(freq), sigma=2, axis=0)

    fig, ax = plt.subplots(4, 1)
    ax[0].set_ylabel('')
    ax[3].set_xlabel('Power Difference')

    ax[0].plot(unique_diff, freq_smooth[:, 0])
    ax[1].plot(unique_diff, freq_smooth[:, 1])
    ax[2].plot(unique_diff, freq_smooth[:, 2])
    ax[3].plot(unique_diff, freq_smooth[:, 3])

    fig.tight_layout()
    plt.show()