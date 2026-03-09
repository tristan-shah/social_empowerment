from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

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

def encode_state(x: Array, y: Array, a: Array) -> Array:
    '''
    Takes in vectors representing x, y position and angle and returns a flat state vector.
    '''
    return jnp.stack([x, y, a], axis = 1).ravel()

def decode_state(state: Array, num_agents: int) -> tuple:
    '''
    Takes in a flat state vector and number of agents and returns x, y position and angle vectors.
    '''
    
    s = jnp.reshape(state, (num_agents, 3))
    x = s[:, 0]
    y = s[:, 1]
    a = s[:, 2]
    return (x, y, a)

def minimum_image_diff(diff: Array, grid_size: float) -> Array:
    '''
    Apply the minimum image convention to a pairwise displacement vector.

    For each component d, returns  d - 2L * round(d / (2L)).
    This gives the shortest displacement on the torus and is differentiable
    (the gradient is 1 almost everywhere, discontinuous only on a set of
    measure zero at the wrap boundary).

    Args:
        diff: (..., 2) array of raw displacements.
        L:    half-length of the simulation box (box spans [-L, L]).

    Returns:
        (..., 2) array of minimum-image displacements.
    '''
    box = 2.0 * grid_size
    return diff - box * jnp.round(diff / box)

def wrap_position(pos: jnp.ndarray, L: float) -> jnp.ndarray:
    '''
    Wrap positions onto the torus [-L, L)^2.

    Uses  x_wrapped = ((x + L) mod 2L) - L
    jnp.mod is differentiable (gradient = 1 almost everywhere).
    '''
    return jnp.mod(pos + L, 2.0 * L) - L
    # return pos - 2.0 * L * jnp.round(pos / (2.0 * L)) ## doesnt prevent nans

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
        # eps = 1e-7
        # v1x_safe = jnp.where(jnp.abs(v1x) < eps, eps, v1x)  
        # a_des = jnp.arctan2(v1y, v1x_safe) + action ## tried this, doesnt prevent nans
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