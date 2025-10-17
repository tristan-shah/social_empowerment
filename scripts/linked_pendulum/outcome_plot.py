from pathlib import Path
import jax
from jax import Array
from jax import numpy as jnp

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

if __name__ == '__main__':
    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path=xml_path)
    print(f'Timestep = {dyn.mjx_model.opt.timestep}')

    power = 2.0
    power = jnp.array([power, power])
    # name = 'power_over_horizon'
    name = 'power_times_horizon'
    folder = Path(f'results/sweep_horizon/{name}/power={power}_alpha=0.01_observation_noise=1.0')

    batches = 29
    horizons = jnp.arange(50, 202, 2)
    num_horizons = len(horizons)

    outcomes = jnp.zeros((num_horizons, num_horizons))
    pair_map = jnp.zeros((num_horizons, num_horizons, 2))
    trajectories = jnp.zeros((num_horizons, num_horizons, 1501, dyn.state_dim))

    def find_nearest_index(array: Array, values: Array):
        diffs = jnp.abs(array[:, None] - values[None, :])
        return jnp.argmin(diffs, axis=0)

    for batch in range(batches):
        print(batch)

        path = folder
 
        pairs = jnp.load(path / f'pairs_batch_{batch}.npy')
        X = jnp.load(path / f'X_batch_{batch}.npy')

        batch_I = find_nearest_index(horizons, pairs[:, 0])
        batch_J = find_nearest_index(horizons, pairs[:, 1])
        outcomes = outcomes.at[batch_I, batch_J].set(batch_get_linked_pendulum_outcome(X[0:pairs.shape[0]]))
        pair_map = pair_map.at[batch_I, batch_J].set(pairs)

        max_idx = pairs.shape[0]
        trajectories = trajectories.at[batch_I, batch_J].set(X[:max_idx])

    plot_outcome_hetamap(outcomes, power, horizons, dt = dyn.mjx_model.opt.timestep, path = f'{name}_power={power[0]}.png')