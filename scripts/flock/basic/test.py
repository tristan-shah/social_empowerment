import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.envs.flock import Flock, make_reset, make_step, render, render_video

if __name__ == '__main__':
    seed = 0
    key = jax.random.key(seed)
    agents = 1000
    grid_size = 50.0
    speed = 1.0
    radius = 1.0
    steps = 1000

    flock = Flock(agents, grid_size, speed, radius)
    reset = make_reset(flock)
    step = make_step(flock)

    xt = reset(key)

    X = jnp.zeros((steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(1000):
        ut = jnp.zeros(flock.control_dim)
        xt = step(xt, ut)
        X = X.at[t+1].set(xt)
        print(t)

    render_video(X, flock, save_path = 'vid.mp4')