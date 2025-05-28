import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx, viewer

import matplotlib.pyplot as plt
from dm_control import mjcf

from soc_emp.dynamics import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer, ModelPredictiveController

def compute_error(X: Array):
    goal = jnp.zeros_like(X)
    goal = goal.at[:, 0].set(5.0)
    r = X - goal
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    dyn = Dynamics(path = 'xml/blob.xml')
    # viewer.launch(dyn.model)

    mjx_data = mjx.make_data(dyn.mjx_model)
    x0 = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])

    Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    R = jnp.eye(dyn.control_dim) * 0.1
    Q = Q.at[0, 0].set(1.0)

    opt = TrajectoryOptimizer(dyn, compute_error, Q, R)

    T = 500
    U = jax.random.normal(key, (T, dyn.control_dim))

    X, U = opt.forward(x0, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    cost = []
    for i in range(10):
        J, X, U, A, B = opt.update(x0, X, U, A, B)
        print(i, J)
        cost.append(J)

    dyn.render(
        X, 
        path = 'ball.mp4', 
        lookat = jnp.array([2.0, 0.0, 0.5]),
        elevation = -20.0,
        distance = 5.0)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('iLQR Trajectory Optimization')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Trajectory Cost')

    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel('Control Signal')

    ax[0].plot(cost)
    ax[1].plot(U)
    fig.savefig('ball_cost.png', dpi = 300)
