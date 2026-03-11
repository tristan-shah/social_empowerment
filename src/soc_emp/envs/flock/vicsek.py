from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

from soc_emp.envs.flock.utils import encode_state, decode_state, minimum_image_diff, wrap_position
from soc_emp.utils import smooth_angle_wrap

@dataclass(frozen = True)
class Vicsek:
    '''
    Implementation of the Vicsek model described in: Signatures of irreversibility in microscopic models of flocking
    https://arxiv.org/abs/2205.14505
    '''
    num_agents: int
    grid_size: float
    neighbor_radius: float
    speed: float = 1.0
    J: float = 1.0
    D: float = 1.0
    dt: float = 0.05

    @property
    def state_dim(self):
        return self.num_agents * 3
    
    @property
    def control_dim(self):
        return self.num_agents

def make_reset(flock: Vicsek):

    num_agents = flock.num_agents
    grid_size = flock.grid_size

    def reset(key):

        k1, k2, k3 = jax.random.split(key, 3)

        x = jax.random.uniform(k1, shape = (num_agents,), minval = - grid_size, maxval = grid_size)
        y = jax.random.uniform(k2, shape = (num_agents,), minval = - grid_size, maxval = grid_size)
        a = jax.random.uniform(k3, shape = (num_agents,), minval = - jnp.pi, maxval = jnp.pi)

        return encode_state(x, y, a)
    
    return jax.jit(reset)


def make_step(flock: Vicsek):
    '''
    Creates the step function for propagating the stochastic dynamical system for Vicsek flocking.
    '''

    num_agents = flock.num_agents
    grid_size = flock.grid_size
    neighbor_radius = flock.neighbor_radius
    speed = flock.speed
    J = flock.J
    D = flock.D
    dt = flock.dt

    def step(state: Array, action: Array, key):

        x, y, a = decode_state(state, flock.num_agents)

        pos = jnp.stack([x, y], axis = 1)

        ## compute position differences
        raw_diff = pos[:, None, :] - pos[None, :, :]
        diff_mi = minimum_image_diff(raw_diff, flock.grid_size)
        dist = jnp.linalg.norm(diff_mi + 1e-10, axis = 2)

        ## gaussian neighbor influence (soft adjacency matrix)
        influence = jnp.exp(-0.5 * (dist / neighbor_radius) ** 2)

        ## angle difference
        angle_diff = a[None, :] - a[:, None]

        ## gradient of entropy formula
        torque = J * (influence * jnp.sin(angle_diff)).sum(axis = 1)

        ## dW noise
        noise = jnp.sqrt(2 * D * dt) * jax.random.normal(key, shape = (num_agents,))

        ## angle update
        a = smooth_angle_wrap(a + torque * dt + action * dt + noise)

        ## position update
        vel = speed * jnp.stack([jnp.cos(a), jnp.sin(a)], axis = 1)
        pos = wrap_position(pos + vel * dt, grid_size)

        return encode_state(pos[:, 0], pos[:, 1], a)

    return jax.jit(step)