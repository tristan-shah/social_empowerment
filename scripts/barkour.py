import jax
from jax import numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer


def compute_error(x: Array):
    goal = jnp.zeros_like(x)
    goal = goal.at[:, 2].set(1.0)
    r = x - goal
    return r

def compute_error(x: Array):
    goal = jnp.zeros_like(x)
    # Standing position:
    goal = goal.at[:, 0].set(0.0)      # X position (forward)
    goal = goal.at[:, 1].set(0.0)      # Y position
    goal = goal.at[:, 2].set(0.5)      # Body height (adjust to dog's standing height)
    goal = goal.at[:, 3].set(1.0)      # Quaternion w (identity rotation)
    goal = goal.at[:, 4].set(0.0)      # Quaternion x
    goal = goal.at[:, 5].set(0.0)      # Quaternion y
    goal = goal.at[:, 6].set(0.0)      # Quaternion z
    r = x - goal
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    dyn = Dynamics(path = 'xml/google_barkour_vb/scene_mjx.xml')

    ctrl_mag = dyn.mjx_model.actuator_ctrlrange[:, 1] - dyn.mjx_model.actuator_ctrlrange[:, 0]

    Q = jnp.eye(dyn.state_dim) * 0.0
    Q = Q.at[0, 0].set(1.0)      # X position
    Q = Q.at[1, 1].set(1.0)      # X position
    Q = Q.at[2, 2].set(1.0)     # Height
    Q = Q.at[3, 3].set(0.5)      # Quaternion w
    Q = Q.at[4, 4].set(0.5)      # Quaternion x
    Q = Q.at[5, 5].set(0.5)      # Quaternion y
    Q = Q.at[6, 6].set(0.5)      # Quaternion z
    R = jnp.eye(dyn.control_dim) * 0.05 #10.0
    R = jnp.fill_diagonal(R, ctrl_mag, inplace = False) * 0.05

    opt = TrajectoryOptimizer(dyn, compute_error, Q, R)

    T = 2000
    U = jax.random.normal(key, (T, dyn.control_dim)) * 0.0

    xt = dyn.init_state()
    xt = xt.at[2].set(0.1)

    X, U = opt.forward(xt, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    cost = []
    for i in range(30):
        J, X, U, A, B = opt.update(xt, X, U, A, B)
        print(i, J)
        cost.append(J)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('iLQR Trajectory Optimization')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Trajectory Cost')

    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel('Control Signal')

    ax[0].plot(cost)
    ax[1].plot(U)
    fig.tight_layout()

    fig.savefig('barkour_cost.png', dpi = 300)

    dyn.render(X, path = 'barkour.mp4', skip = 5)



    # X = jnp.zeros((T+1, dyn.state_dim))

    # X = X.at[0].set(xt)
    # U = jax.random.normal(key, (T, dyn.control_dim))

    # for t in range(T):
    #     ut = U[t]
    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)
    #     print(xt)

    # dyn.render(X, path = 'barkour.mp4', skip = 5)