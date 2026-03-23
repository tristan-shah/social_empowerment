from pathlib import Path

from jax import numpy as jnp

if __name__ == '__main__':
    horizon = 80

    root = Path('results/egoistic_angle_state/sweep_power_stop_grad/resolution=100_seed=0_horizon=80_alpha=0.01_observation_noise=1.0_stiffness=3.0')
    # root = Path('results/egoistic_full_state/resolution=100_seed=0_horizon=80_alpha=0.01_observation_noise=1.0_stiffness=3.0')

    for i in range(50):

        X = jnp.load(root / f'X_batch_{i}.npy')
        pairs = jnp.load(root / f'pairs_batch_{i}.npy')

        print(X.shape)
        print(pairs.shape)

        exit()