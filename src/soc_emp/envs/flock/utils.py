from jax import Array
from jax import numpy as jnp

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


def compute_order_parameter(state: Array, num_agents: int) -> Array:
    '''
    Compute the Vicsek order parameter φ ∈ [0, 1].

    φ = (1/N) |Σ_i v̂_i|

    where v̂_i = (cos(aᵢ), sin(aᵢ)) is the unit heading of agent i.
    φ = 1 means perfect alignment, φ ≈ 0 means disordered.
    '''
    _, _, a = decode_state(state, num_agents)

    mean_cos = jnp.mean(jnp.cos(a))
    mean_sin = jnp.mean(jnp.sin(a))

    return jnp.sqrt(mean_cos ** 2 + mean_sin ** 2)

def build_flock_state_matrix(num_agents: int, state_type: str):
    
    idx = jnp.arange(0, num_agents * 3)
    x, y, a = decode_state(idx, num_agents)

    if state_type == 'pos':
        return jnp.stack([x, y], axis = 1)
    elif state_type == 'angle':
        return a[:, None]
    elif state_type == 'full':
        return jnp.stack([x, y, a], axis = 1)
    else:
        raise ValueError('Invalid state type.')