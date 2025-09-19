import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll
from soc_emp.utils import smooth_angle_wrap

@jax.jit
def cubic_spline_interp(x: Array, y: Array, x_new: Array) -> Array:
    """
    Natural cubic spline interpolation (single trajectory, JAX compatible).

    Args:
        x: shape (n,), strictly increasing knots
        y: shape (n, d), data values
        x_new: shape (m,), query points

    Returns:
        y_new: shape (m, d), interpolated values
    """
    n, d = y.shape
    h = x[1:] - x[:-1]                    # (n-1,)
    alpha = (y[1:] - y[:-1]) / h[:, None] # (n-1, d)

    # Build tridiagonal system for c (second derivatives)
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

    # Spline coefficients
    a = y[:-1]                                   # (n-1, d)
    b = alpha - (h[:, None] * (2*c[:-1] + c[1:])) / 3.0
    d_coef = (c[1:] - c[:-1]) / (3*h)[:, None]

    # Interval indices for queries
    idx = jnp.searchsorted(x, x_new) - 1
    idx = jnp.clip(idx, 0, n-2)

    dx = (x_new - x[idx])[:, None]  # (m, 1)

    # Gather spline coefficients
    a_i = a[idx]
    b_i = b[idx]
    c_i = c[idx]
    d_i = d_coef[idx]

    return a_i + b_i*dx + c_i*dx**2 + d_i*dx**3

def compute_pendulum_error(X: Array):
    assert X.shape[1] == 2

    r = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        X[:, 1]
    ], axis = 1)

    assert r.shape == X.shape
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(22)
    horizon = 50
    power = 1.0
    steps = 1500

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)
    ctrl_range = dyn.mjx_model.actuator_ctrlrange
    low = ctrl_range[:, 0]
    high = ctrl_range[:, 1]
    dx = dyn.state_dim
    du = dyn.control_dim
    Q = jnp.eye(dx)
    Q = Q.at[1, 1].set(0.01)
    R = jnp.eye(du) * 1.0

    x0 = jnp.zeros(dx)

    x = jnp.arange(0, horizon)

    knots = 10
    idx = jnp.linspace(0, len(x)-1, knots, dtype = int)

    xp = x[idx]
    ## initial theta
    yp = jax.random.normal(key, (knots, du)).clip(low, high)

    N = 10
    eps = jax.random.normal(key, (N, knots, du))
    yp_dist = (jnp.zeros_like(yp) + eps).clip(low, high)

    y = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(xp, yp_dist, x).clip(low, high)
    print(y.shape)

    fig, ax = plt.subplots(1, 1)
    for i in range(y.shape[0]):
        ax.plot(y[i, :], alpha = 0.2)
        ax.scatter(xp, yp_dist[i], alpha = 0.2)
    plt.show()

    


    # plt.plot(y)
    # plt.scatter(xp, yp)
    # plt.show()

    ## build spline
    # coeffs = cubic_spline_coeffs(xp, yp)
    # print(coeffs)
    # y = cubic_spline_eval(x, coeffs).clip(low, high)

    # print(y.shape)
    # X = unroll(dyn, x0, y)
    # e = compute_pendulum_error(X)
    # J = einsum(e, Q, e, 't x1, x1 x2, t x2 ->')
    # print(J)