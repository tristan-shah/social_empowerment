import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll
from soc_emp.utils import smooth_angle_wrap, cubic_spline_interp

@jax.jit
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
    horizon = 600
    N = 1000
    sigma = 0.1
    knots = 70

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

    ## initial state
    x0 = jnp.zeros(dx)

    x = jnp.arange(0, horizon)

    ## choose the number and spacing of the knots
    knots_idx = jnp.linspace(0, len(x)-1, knots, dtype = int)

    ## select x points at knot locations
    x_knots = x[knots_idx]
    ## initial theta should be zero prior
    U_knots = jnp.zeros((knots, du))

    cost_hist = []

    ## predictive sampling loop
    for i in range(200):

        eps = jax.random.normal(key, (N, knots, du))
        U_knots_dist = (U_knots + eps * sigma).clip(low, high)
        U = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(x_knots, U_knots_dist, x).clip(low, high)

        ## paralel simulation
        X = jax.vmap(unroll, in_axes = (None, None, 0))(dyn, x0, U)
        ## evaluate error of each trajectory
        e = jax.vmap(compute_pendulum_error, in_axes = 0)(X)

        ## evaluate cost of batch and determine the best index
        J = einsum(e, Q, e, 'b t x1, x1 x2, b t x2 -> b')
        best_idx = jnp.argmin(J)

        ## select the knots from the best index
        U_knots = U[best_idx, knots_idx, :]
        print(i, J[best_idx])

        cost_hist.append(J[best_idx])

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Predictive Sampling Cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.plot(cost_hist)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Trajectory')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta Dot')
    ax.plot(X[best_idx, :, 0], X[best_idx, :, 1])
    ax.scatter(X[best_idx, 0, 0], X[best_idx, 0, 1], color = 'green', label = 'Start')
    ax.scatter(X[best_idx, -1, 0], X[best_idx, -1, 1], color = 'red', label = 'End')
    ax.legend()
    plt.show()

    # dyn.render(X[best_idx], path = 'test.mp4', skip = 2)


    # '''
    # Plotting splines
    # '''
    # N = 10
    # knots = 5
    # sigma = 0.25

    # ## choose the number and spacing of the knots
    # knots_idx = jnp.linspace(0, len(x)-1, knots, dtype = int)
    # ## select x points at knot locations
    # x_knots = x[knots_idx]
    # eps = jax.random.normal(key, (N, knots, du))
    # U_knots_dist = (eps * sigma).clip(low, high)
    # U = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(x_knots, U_knots_dist, x).clip(low, high)


    # fig, ax = plt.subplots(1, 1)
    # ax.set_title(f'Control Trajectories N = {N}')
    # ax.set_xlabel('Horizon')
    # ax.set_ylabel('Control')
    # ax.set_ylim(-1.0, 1.0)
    # for i in range(N):
    #     ax.plot(U[i, :, 0], alpha = 1.0)
    #     ax.scatter(x_knots, U_knots_dist[i, :, 0], alpha = 1.0)
    # plt.show()