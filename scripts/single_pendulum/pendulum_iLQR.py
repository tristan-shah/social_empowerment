import jax
from jax import numpy as jnp
from jax import Array
from mujoco import mjx
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer, ModelPredictiveController
from soc_emp.utils import smooth_angle_wrap, diff_qpos, split_state

def compute_pendulum_error(X: Array):
    assert X.shape[1] == 2

    r = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        X[:, 1]
    ], axis = 1)

    assert r.shape == X.shape
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    x0 = jnp.zeros((dyn.state_dim,))
    x0 = x0.at[0].set(0.0)


    # x0_bar = x0.at[:].set([3.5, 0.0])
    # x0 = x0.at[:].set([9.4, 0.0])

    # qpos_bar, qvel_bar = split_state(x0_bar, dyn.nq)
    # qpos, qvel = split_state(x0, dyn.nq)

    # print(
    #     diff_qpos(dyn.model, qpos, qpos_bar)
    # )
    
    '''
    Open Loop
    '''
    # T = 600
    # Q = jnp.eye(dyn.state_dim)
    # R = jnp.eye(dyn.control_dim) * 0.005
    # Q = Q.at[1, 1].set(0.01)

    # opt = TrajectoryOptimizer(
    #     dyn, 
    #     compute_pendulum_error,
    #     Q,
    #     R)
    
    # U = jax.random.normal(key, (T, dyn.control_dim))
    # X, U = opt.forward(x0, U = U)
    # A, B = opt.batch_linearize(X[:-1], U)

    # cost = []
    # for i in range(50):
    #     J, X, U, A, B = opt.update(x0, X, U, A, B)
    #     print(i, J)
    #     cost.append(J)

    # fig, ax = plt.subplots(1, 2)

    # fig.suptitle('iLQR Trajectory Optimization')
    # ax[0].set_xlabel('Iteration')
    # ax[0].set_ylabel('Trajectory Cost')

    # ax[1].set_xlabel('Timestep')
    # ax[1].set_ylabel('Control Signal')

    # ax[0].plot(cost)
    # ax[1].plot(U)
    # fig.savefig('pendulum_cost.png', dpi = 300)

    # dyn.render(X, path = 'pendulum.mp4', skip = 2)

    '''
    MPC
    '''
    Q = jnp.eye(dyn.state_dim)
    Q = Q.at[1, 1].set(0.01)
    R = jnp.eye(dyn.control_dim) * 1.0

    opt = TrajectoryOptimizer(dyn, compute_pendulum_error, Q, R)
    mpc = ModelPredictiveController(opt, 200)

    xt = jnp.copy(x0)

    steps = 600

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    cost = []
    for t in range(steps):
        ut, J = mpc(xt)
        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)
        print(t, xt, ut, J)
        cost.append(J)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('iLQR Trajectory Optimization')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Trajectory Cost')
    ax.plot(cost)
    fig.savefig('pendulum_cost.png', dpi = 300)

    dyn.render(X, path = 'pendulum.mp4', skip = 2)
