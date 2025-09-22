import jax
from jax import Array
from jax import numpy as jnp
from mujoco.mjx import Data

def smooth_angle_wrap(theta: float):
    return jax.lax.atan2(jax.lax.sin(theta), jax.lax.cos(theta))

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

def get_state(data: Data):
    return jnp.concatenate([data.qpos, data.qvel])

def select_output(f: callable, index: int):
    return lambda *args, **kwargs: f(*args, **kwargs)[index]

@jax.jit
def cubic_spline_interp(x: Array, y: Array, x_new: Array) -> Array:
    '''
    Code generated with ChatGPT

    Natural cubic spline interpolation (single trajectory, JAX compatible).

    Args:
        x: shape (n,), strictly increasing knots
        y: shape (n, d), data values
        x_new: shape (m,), query points

    Returns:
        y_new: shape (m, d), interpolated values
    '''
    n, d = y.shape
    h = x[1:] - x[:-1]                    # (n-1,)
    alpha = (y[1:] - y[:-1]) / h[:, None] # (n-1, d)

    ## Build tridiagonal system for c (second derivatives)
    A = jnp.zeros((n, n))
    A = A.at[0, 0].set(1.0)
    A = A.at[-1, -1].set(1.0)
    rhs = jnp.zeros((n, d))

    def body(i, vals):
        A, rhs = vals
        A = A.at[i, i-1].set(h[i-1])
        A = A.at[i, i].set(2 * (h[i-1] + h[i]))
        A = A.at[i, i+1].set(h[i])
        rhs = rhs.at[i].set(3 * (alpha[i] - alpha[i-1]))
        return A, rhs

    A, rhs = jax.lax.fori_loop(1, n-1, body, (A, rhs))
    c = jnp.linalg.solve(A, rhs)  # (n, d)

    ## Spline coefficients
    a = y[:-1]                                   # (n-1, d)
    b = alpha - (h[:, None] * (2*c[:-1] + c[1:])) / 3.0
    d_coef = (c[1:] - c[:-1]) / (3*h)[:, None]

    ## Interval indices for queries
    idx = jnp.searchsorted(x, x_new) - 1
    idx = jnp.clip(idx, 0, n-2)

    dx = (x_new - x[idx])[:, None]  # (m, 1)

    ## Gather spline coefficients
    a_i = a[idx]
    b_i = b[idx]
    c_i = c[idx]
    d_i = d_coef[idx]

    return a_i + b_i*dx + c_i*dx**2 + d_i*dx**3

@jax.jit
def zero_order_interp(x: Array, y: Array, x_new: Array) -> Array:
    '''
    Zero-order (piecewise constant) interpolation, JAX compatible.

    Args:
        x: shape (n,), strictly increasing knots
        y: shape (n, d), data values
        x_new: shape (m,), query points

    Returns:
        y_new: shape (m, d), interpolated values
    '''
    n, d = y.shape

    # Find interval indices: right bin edge, subtract 1 -> nearest left knot
    idx = jnp.searchsorted(x, x_new, side = "right") - 1
    idx = jnp.clip(idx, 0, n-1)  # clamp to valid range

    # Gather corresponding y-values
    return y[idx]

import jax
import jax.numpy as jnp
from jax import Array

@jax.jit
def linear_interp(x: Array, y: Array, x_new: Array) -> Array:
    '''
    Linear interpolation, JAX compatible.

    Args:
        x: shape (n,), strictly increasing knots
        y: shape (n, d), data values
        x_new: shape (m,), query points

    Returns:
        y_new: shape (m, d), interpolated values
    '''
    n, d = y.shape

    # Find interval indices (left endpoint of interval)
    idx = jnp.searchsorted(x, x_new) - 1
    idx = jnp.clip(idx, 0, n - 2)  # keep inside valid range

    # Compute normalized position within interval
    x0 = x[idx]
    x1 = x[idx + 1]
    t = (x_new - x0) / (x1 - x0)

    # Get corresponding y-values
    y0 = y[idx]
    y1 = y[idx + 1]

    # Linear interpolation
    return (1.0 - t[:, None]) * y0 + t[:, None] * y1