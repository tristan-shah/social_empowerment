import jax
from jax import Array
from jax import numpy as jnp
from jax.experimental import sparse

import matplotlib.pyplot as plt

from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.plot import render_image, render_video

# def compute_empowerment(dyn: Dynamics, xt: Array, U: Array, P: float):
#     X = unroll(dyn, xt, U)
#     A, B = jax.vmap(dyn.linearize)(X[:-1], U)
#     F = compute_F_from_A_B(A, B)
#     F = jnp.permute_dims(F, (1, 0, 2))
    
#     ## S is the covariance matrix of the final state.
#     S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
#     h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
#     v = waterfilling_implicit(h2, P)
#     p = compute_power(v, h2)
#     e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
#     return e

def make_unroll(step: callable):
    '''
    Unrolls dynamical system with randomness
    '''

    def unroll(x0: Array, U: Array, key):

        def body_fn(carry: tuple, u: Array):
            x, key = carry
            key, subkey = jax.random.split(key)
            x_next = step(x, u, subkey)
            carry = (x_next, key)
            return carry, x_next
        
        (_, key), X = jax.lax.scan(body_fn, init = (x0, key), xs = U)
        X = jnp.concatenate([x0[None, :], X], axis = 0)

        return X, key
    
    return jax.jit(unroll)

if __name__ == '__main__':
    seed = 10
    key = jax.random.key(seed)
    num_agents = 2000
    grid_size = 10.0
    neighbor_radius = 0.5
    speed = 1.0
    J = 0.1
    D = 0.1
    steps = 1000
    
    flock = Vicsek(num_agents, grid_size, neighbor_radius, speed, J, D)
    reset = make_reset(flock)
    step = make_step(flock)

    xt = reset(key)
    ut = jnp.zeros(flock.control_dim)


    X = jnp.zeros((steps + 1, flock.state_dim))
    X = X.at[0].set(xt)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        xt = step(xt, ut, subkey)

        print(t)
        X = X.at[t+1].set(xt)

    
    
    
    render_video(X, flock, path = 'vid.mp4')