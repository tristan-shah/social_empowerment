from pathlib import Path
from jax import numpy as jnp

from soc_emp import Dynamics
from scripts.linked_pendulum.sweep_power import batch_get_linked_pendulum_outcome, plot_outcome_hetamap

if __name__ == '__main__':
    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path=xml_path)
    print(f'Timestep = {dyn.mjx_model.opt.timestep}')
    
    folder = Path('sweep_power')

    for horizon in [50, 75, 100, 125, 150, 175, 200]:
        horizon = 100
        batches = 5
        powers = jnp.linspace(0.5, 3.0, 30)
        num_powers = powers.shape[0]

        outcomes = jnp.zeros((num_powers, num_powers))
        pair_map = jnp.zeros((num_powers, num_powers, 2))

        for batch in range(batches):

            path = folder / f'horizon={horizon}'
            # pairs = jnp.load(path / f'horizon={horizon}_pairs_batch_{batch}.npy')
            # X = jnp.load(path / f'horizon={horizon}_X_batch_{batch}.npy')

            pairs = jnp.load(path / f'pairs_batch_{batch}.npy')
            X = jnp.load(path / f'X_batch_{batch}.npy')


            batch_I = jnp.searchsorted(powers, pairs[:, 0])
            batch_J = jnp.searchsorted(powers, pairs[:, 1])
            outcomes = outcomes.at[batch_I, batch_J].set(batch_get_linked_pendulum_outcome(X[0:pairs.shape[0]]))
            pair_map = pair_map.at[batch_I, batch_J].set(pairs)
            # print(pairs.shape, X.shape)
            # print(pairs.shape)

        collaboration_percentage = (outcomes == 3).sum() / (outcomes > 0).sum()
        print(horizon * dyn.mjx_model.opt.timestep, '\t', collaboration_percentage)

    # print(pair_map[outcomes == 3])

    import jax
    p = jax.random.choice(
        jax.random.PRNGKey(20), 
        pair_map[outcomes == 1],
        shape = (10,), 
        replace = False
        )
    
    print(p)
    plot_outcome_hetamap(outcomes, horizon, powers, dt = dyn.mjx_model.opt.timestep, path = f'horizon={horizon}_outcome_heatmap.png')