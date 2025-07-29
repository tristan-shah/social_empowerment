from pathlib import Path
from jax import numpy as jnp

from soc_emp import Dynamics
from sweep import batch_get_linked_pendulum_outcome, plot_outcome_hetamap

if __name__ == '__main__':

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path=xml_path)
    print(f'Timestep = {dyn.mjx_model.opt.timestep}')
    
    folder = Path('results_random_init')
    horizon = 50
    batches = 18
    powers = jnp.linspace(0.5, 3.0, 30)
    num_powers = powers.shape[0]

    outcomes = jnp.zeros((num_powers, num_powers))

    for batch in range(batches):

        path = folder / f'horizon={horizon}'
        pairs = jnp.load(path / f'horizon={horizon}_pairs_batch_{batch}.npy')
        X = jnp.load(path / f'horizon={horizon}_X_batch_{batch}.npy')

        batch_I = jnp.searchsorted(powers, pairs[:, 0])
        batch_J = jnp.searchsorted(powers, pairs[:, 1])
        outcomes = outcomes.at[batch_I, batch_J].set(batch_get_linked_pendulum_outcome(X))
        print(outcomes)

    plot_outcome_hetamap(outcomes, horizon, powers, dt = dyn.mjx_model.opt.timestep, path = f'horizon={horizon}_outcome_heatmap.png')