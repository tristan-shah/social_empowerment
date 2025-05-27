import jax.numpy as jnp
from jax import lax, jacfwd, config
import jax
from jax.lax import custom_root

tol = 1e-6
inf = jnp.array(10e10)

@jax.jit
def waterfilling_solver(noise_levels, total_power):
    clipped_noise = jnp.maximum(noise_levels, tol)
    inverse = 1.0 / clipped_noise
    # inverse = jnp.where(jnp.abs(noise_levels) < tol, jnp.inf, 1.0 / (noise_levels + 1e-8 * (noise_levels == 0)))
    # inverse_sorted = jnp.sort(jnp.nan_to_num(inverse, nan=1e10))  # Replace inf with a large number
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

    bot, top = jax.lax.while_loop(cond_fun, body_fun, (0, n))

    mu = inverse_sorted[bot] + (total_power - P[bot]) / (bot + 1)
    return mu

@jax.jit
def waterfilling_implicit(noise_levels, total_power):
    safe_noise = jnp.maximum(noise_levels, tol)

    def f(mu):
        return jnp.sum(jnp.maximum(0.0, mu - 1.0 / safe_noise)) - total_power
    
    initial_guess = 0.0
    
    def solve(f, initial_guess):
        return waterfilling_solver(noise_levels, total_power)
    
    def tangent_solve(g, y):
        return y / g(1.0)
    
    return custom_root(f, initial_guess, solve, tangent_solve)

@jax.jit
def eigen_implicit(A):
    def f(eigenvalues):
        return jnp.trace(A) - jnp.sum(eigenvalues)
    
    initial_guess = 0.0
    
    def solve(f, initial_guess):
        return jnp.linalg.eigvalsh(A)
    
    def tangent_solve(g, y):
        return y / g(1.0)
    
    return custom_root(f, initial_guess, solve, tangent_solve)

def E(Integrate, T, P, X, dt, a_size):
    a = jnp.zeros((T, a_size))    
    m = jax.jacfwd(lambda action: Integrate(dt, X, action))(a)
    if type(m) == tuple:
        m = m[0]
    # m_mask = jnp.zeros_like(m, dtype=bool).at[-1, :2, :, :].set(True)
    # m = m * m_mask
    
    m = m.reshape((-1, T, a_size))
    # s = jnp.einsum('napq, nbpq -> ab', m, m)
    # s = jnp.einsum('anpq, bnpq -> ab', m, m)
    s = jnp.einsum('apq, bpq -> ab', m, m)
    # s = 0.5 * (s + s.T)
    # s = jnp.maximum(s, 0.0)
    h2 = eigen_implicit(s)
    # h2 = jnp.maximum(h2, 0.0)
    v = waterfilling_implicit(h2, P)
    p = jnp.maximum(jnp.array(0.0), v -1 / jnp.maximum(h2, tol))
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    # e = lax.cond(jnp.isnan(e), lambda: jnp.array(0.0), lambda: e)
    return e, (e, h2)

def E_pendulum(Integrate, T, P, X, dt):
    a = jnp.zeros(T)
    m = jax.jacfwd(lambda action: Integrate(dt, T, X, action))(a)
    s = m @ m.T
    h2 = eigen_implicit(s)
    h2 = jnp.maximum(h2, 1e-12)
    v = waterfilling_implicit(h2, P)
    p = jnp.maximum(jnp.array(0.0), v -1 / h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    return e, (e, h2)
