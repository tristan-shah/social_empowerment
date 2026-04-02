import json
from pathlib import Path

from jax import numpy as jnp

from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.plot import render_video

if __name__ == '__main__':

    root = Path('/Users/tristanshah/Desktop/code/social_empowerment/results/Vicsek/D=0.0-J=0.1-alpha=0.01-behavior=egoistic-grid_size=5.0-horizon=5-num_agents=125-observation_noise=1.0-power_density=2.0-radius=0.5-seed=6-speed=1.0-steps=4000')

    with open(root / 'params.json', 'r') as f:
        params = json.load(f)

    flock = Vicsek(params['num_agents'], params['grid_size'], params['radius'], params['speed'], params['J'], params['D'])

    X = jnp.load(root / 'trajectory.npy')

    render_video(X, flock, path = 'vid.mp4')