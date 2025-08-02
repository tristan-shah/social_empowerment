import jax
from jax import numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer

def compute_error(x: Array):
    goal = jnp.zeros_like(x)
    # Keyframe "home" pose
    goal = goal.at[:, 0].set(2.0)  # Body x
    goal = goal.at[:, 1].set(0.0)  # Body y
    # goal = goal.at[:, 2].set(0.28)  # Body height
    goal = goal.at[:, 3].set(1.0)  # Quaternion w
    goal = goal.at[:, 4].set(0.0)  # Quaternion x
    goal = goal.at[:, 5].set(0.0)  # Quaternion y
    goal = goal.at[:, 6].set(0.0)  # Quaternion z
    return x - goal

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    # dyn = Dynamics(path='xml/google_barkour_vb/scene_mjx.xml')
    # dyn = Dynamics(path = 'xml/unitree_go2/scene_mjx.xml', integrator = 'rk4')
    dyn = Dynamics(path = 'xml/unitree_go2/scene_mjx.xml', integrator = 'euler')
    print("model.nq:", dyn.model.nq, "model.nv:", dyn.model.nv)
    print("dyn.nq:", dyn.nq, "dyn.nv:", dyn.nv)
    print("dyn.state_dim:", dyn.state_dim, "dyn.control_dim:", dyn.control_dim)

    # Initialize Q with correct velocity slice
    Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    Q = Q.at[0, 0].set(1.0)
    # Q = Q.at[1, 1].set(1.0)
    Q = Q.at[2, 2].set(1.0)  # Body height
    Q = Q.at[3:7, 3:7].set(jnp.eye(4) * 0.5)  # Quaternion
    # Q = Q.at[7:19, 7:19].set(jnp.eye(12) * 0.1)  # Leg joints
    # Q = Q.at[19:37, 19:37].set(jnp.eye(18) * 0.01)  # Velocities
    R = jnp.eye(dyn.control_dim) * 0.1

    print("Q shape:", Q.shape)  # Should be (37, 37)

    opt = TrajectoryOptimizer(dyn, compute_error, Q, R)
    T = 1000
    ctrl_keyframe = jnp.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    U = jnp.tile(ctrl_keyframe, (T, 1)) + jax.random.uniform(key, (T, dyn.control_dim)) * 0.01
    xt = dyn.init_state()

    X, U = opt.forward(xt, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    cost = []
    for i in range(5):
        J, X, U, A, B = opt.update(xt, X, U, A, B)
        print(f"Iteration {i}, Cost: {J}")
        cost.append(J)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(cost)
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 0].set_ylabel('Cost')
    ax[0, 1].plot(X[:, 2], label='Height')
    ax[0, 1].set_ylabel('Height (m)')
    ax[1, 0].plot(X[:, 3:7], label=['qw', 'qx', 'qy', 'qz'])
    ax[1, 0].set_ylabel('Quaternion')
    ax[1, 1].plot(U, label=[f'Control {i}' for i in range(dyn.control_dim)])
    ax[1, 1].set_ylabel('Controls')
    fig.tight_layout()
    fig.savefig('barkour_states.png', dpi = 300)
    dyn.render(X, path='barkour.mp4', skip = 5)