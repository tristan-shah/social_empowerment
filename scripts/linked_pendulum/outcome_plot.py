from pathlib import Path
import jax
from jax import Array
from jax import numpy as jnp

from soc_emp import Dynamics
from soc_emp.utils import smooth_angle_wrap
from sweep_power import plot_outcome_hetamap


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

    horizon = 200
    
    folder = Path(f'results/seed=10_horizon={horizon}_alpha=0.01_observation_noise=1.0_stiffness=3.0')

    batches = 100
    num_powers = 100
    powers = jnp.linspace(0.5, 3.0, num_powers)
    num_powers = powers.shape[0]

    outcomes = jnp.zeros((num_powers, num_powers))
    pair_map = jnp.zeros((num_powers, num_powers, 2))
    trajectories = jnp.zeros((num_powers, num_powers, 1501, dyn.state_dim))

    def find_nearest_index(array: Array, values: Array):
        diffs = jnp.abs(array[:, None] - values[None, :])
        return jnp.argmin(diffs, axis=0)

    for batch in range(batches):

        path = folder
 
        pairs = jnp.load(path / f'pairs_batch_{batch}.npy')
        X = jnp.load(path / f'X_batch_{batch}.npy')

        # batch_I = jnp.searchsorted(powers, pairs[:, 0])
        # batch_J = jnp.searchsorted(powers, pairs[:, 1])

        batch_I = find_nearest_index(powers, pairs[:, 0])
        batch_J = find_nearest_index(powers, pairs[:, 1])
        outcomes = outcomes.at[batch_I, batch_J].set(batch_get_linked_pendulum_outcome(X[0:pairs.shape[0]]))
        pair_map = pair_map.at[batch_I, batch_J].set(pairs)
        trajectories = trajectories.at[batch_I, batch_J].set(X)

    idx = 80
    print(pair_map[80, 80])
    # print(trajectories[idx, idx])

    dyn.render(
        trajectories[idx, idx],
        path = 'equal.mp4',
        skip = 2
    )


    # collaboration_percentage = (outcomes == 3).sum() / (outcomes > 0).sum()
    # print(horizon * dyn.mjx_model.opt.timestep, '\t', collaboration_percentage)
    # plot_outcome_hetamap(outcomes, horizon, powers, dt = dyn.mjx_model.opt.timestep, path = f'horizon={horizon}_outcome_heatmap.png')