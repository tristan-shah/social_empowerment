import jax

from soc_emp.envs.flock import Flock, make_reset, make_step, render

if __name__ == '__main__':
    seed = 0
    key = jax.random.key(seed)
    agents = 1000
    grid_size = 10.0
    speed = 1.0
    radius = 1.0

    flock = Flock(agents, grid_size, speed, radius)
    reset = make_reset(flock)
    step = make_step(flock)

    state = reset(key)

    print(state)