import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum

from soc_emp import Dynamics

tol = 1e-6

def unroll(dyn: Dynamics, xt: Array, U: Array):
    '''
    Jax compatable simulation loop.
    '''

    def body_fun(xt: Array, ut: Array):
        xt_next = dyn.step(xt, ut)
        return xt_next, xt_next
    
    xt, _ = jax.lax.scan(body_fun, xt, U)

    return xt

compute_F = jax.jit(jax.jacfwd(unroll, argnums = 2), static_argnums = 0)

@jax.jit
def waterfilling_solver(noise_levels: Array, total_power: float):
    '''
    Code courtesy of Noam Smilovich. 
    '''
    clipped_noise = jnp.maximum(noise_levels, tol)
    inverse = 1.0 / clipped_noise

    inverse_sorted = jnp.sort(inverse)
    n = len(inverse_sorted)
    P = jnp.zeros_like(inverse_sorted)

    def compute_P(carry, i):
        P_prev = carry
        P_next = P_prev + i * (inverse_sorted[i] - inverse_sorted[i-1])
        return P_next, P_next

    _, P = jax.lax.scan(compute_P, 0.0, jnp.arange(1, n))
    P = jnp.concatenate([jnp.array([0.0]), P])  # P[0] = 0

    def cond_fun(state):
        bot, top = state
        return top - bot > 1

    def body_fun(state):
        bot, top = state
        mid = (bot + top) // 2
        new_bot = jax.lax.cond(total_power >= P[mid], lambda: mid, lambda: bot)
        new_top = jax.lax.cond(total_power >= P[mid], lambda: top, lambda: mid)
        return new_bot, new_top

    bot, _ = jax.lax.while_loop(cond_fun, body_fun, (0, n))

    mu = inverse_sorted[bot] + (total_power - P[bot]) / (bot + 1)
    return mu

def compute_empowerment(dyn: Dynamics, xt: Array, U: Array, P: float):

    F = compute_F(dyn, xt, U)
    S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
    h2 = jnp.linalg.eigvalsh(S)
    v = waterfilling_solver(h2, P)
    p = jnp.maximum(jnp.array(0.0), v - 1 / h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    return e