from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

from soc_emp.envs.flock.utils import encode_state, decode_state, wrap_position, minimum_image_diff
from soc_emp.utils import smooth_angle_wrap

@dataclass(frozen = True)
class Flock:
    num_agents: int
    grid_size: int = 5.0
    speed: float = 1.0
    neighbor_radius: float = 0.5
    dt: float = 0.05

    @property
    def state_dim(self):
        return self.num_agents * 3
    
    @property
    def control_dim(self):
        return self.num_agents

def make_reset(flock: Flock):

    def reset(key):

        k1, k2, k3 = jax.random.split(key, 3)
        grid_size = flock.grid_size

        x = jax.random.uniform(k1, shape = (flock.num_agents,), minval = - grid_size, maxval = grid_size)
        y = jax.random.uniform(k2, shape = (flock.num_agents,), minval = - grid_size, maxval = grid_size)
        a = jax.random.uniform(k3, shape = (flock.num_agents,), minval = - jnp.pi, maxval = jnp.pi)

        return encode_state(x, y, a)
    
    return jax.jit(reset)

def make_step(flock: Flock):

    def step(state: Array, action: Array):

        x, y, a = decode_state(state, flock.num_agents)

        ## velocity (pre-update)
        v0x = jnp.cos(a) * flock.speed ## (N x 2)
        v0y = jnp.sin(a) * flock.speed ## (N x 2)

        pos = jnp.stack([x, y], axis = 1)
        vel = jnp.stack([v0x, v0y], axis = 1)

        ## compute position differences
        raw_diff = pos[:, None, :] - pos[None, :, :]
        diff_mi = minimum_image_diff(raw_diff, flock.grid_size)
        dist = jnp.linalg.norm(diff_mi + 1e-10, axis = 2)

        ## compute soft neighbor influence
        influence = jnp.exp(-0.5 * (dist / flock.neighbor_radius) ** 2)

        # influence = influence * (1 - jnp.eye(flock.num_agents)) ## removes self influence? more interesting behavior with this
        neighbors = influence.sum(axis = 1, keepdims = True) + 1e-8  # avoid div by zero

        avg_neighbor_vel = (influence @ vel) / neighbors

        v1x = v0x + avg_neighbor_vel[:, 0]
        v1y = v0y + avg_neighbor_vel[:, 1]

        ## control biases heading
        a_des = jnp.arctan2(v1y, v1x) + action
        a_dot = smooth_angle_wrap(a_des - a)

        ## change in position is velocity
        pos_dot = vel

        ## euler integration
        raw_pos = pos + pos_dot * flock.dt
        pos = wrap_position(raw_pos, flock.grid_size)
        a = smooth_angle_wrap(a + a_dot * flock.dt)

        ## package state
        state = encode_state(pos[:, 0], pos[:, 1], a)

        return state

    return jax.jit(step)


def make_compute_order_parameter(flock: Flock):

    def compute_order_parameter(state: Array) -> Array:
        '''
        Compute the Vicsek order parameter φ ∈ [0, 1].

        φ = (1/N) |Σ_i v̂_i|

        where v̂_i = (cos(aᵢ), sin(aᵢ)) is the unit heading of agent i.
        φ = 1 means perfect alignment, φ ≈ 0 means disordered.
        '''
        _, _, a = decode_state(state, flock.num_agents)

        mean_cos = jnp.mean(jnp.cos(a))
        mean_sin = jnp.mean(jnp.sin(a))

        return jnp.sqrt(mean_cos ** 2 + mean_sin ** 2)

    return jax.jit(compute_order_parameter)

def build_flock_state_matrix(flock: Flock, state_type: str):
    idx = jnp.arange(0, flock.num_agents * 3)
    x, y, a = decode_state(idx, flock.num_agents)

    if state_type == 'pos':
        return jnp.stack([x, y], axis = 1)
    elif state_type == 'angle':
        return a[:, None]
    elif state_type == 'full':
        return jnp.stack([x, y, a], axis = 1)
    else:
        raise ValueError('Invalid state type.')