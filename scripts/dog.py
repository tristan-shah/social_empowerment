import jax
from jax import numpy as jnp
from jax import Array
from mujoco import mjx
import matplotlib.pyplot as plt

from soc_emp.dynamics import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer, ModelPredictiveController

def compute_google_barkour_vb_error(X: Array):
    goal = jnp.zeros(X.shape[1])
    goal = goal.at[0].set(1.0) ## go to the right
    goal = goal.at[1].set(0.0) ## stay in a straight line
    goal = goal.at[2].set(0.2) ## torso should be at this height
    r = X - goal
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    dyn = Dynamics(path = 'mujoco_menagerie/google_barkour_vb/scene_mjx.xml')

    mjx_data = mjx.make_data(dyn.mjx_model)

    x0 = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    ## set the initial position of the dog
    x0 = x0.at[0].set(-1.0)
    x0 = x0.at[2].set(0.1)

    ## penalize the x y z coordinates of the dog
    Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    Q = Q.at[0, 0].set(1.0)
    Q = Q.at[1, 1].set(1.0)
    Q = Q.at[2, 2].set(1.0)

    ## control penalty
    R = jnp.eye(dyn.control_dim) * 1e-2

    opt = TrajectoryOptimizer(
        dyn,
        compute_google_barkour_vb_error,
        Q,
        R)

    '''
    Open Loop
    '''
    T = 300
    U = jax.random.normal(key, (T, dyn.control_dim)).clip(opt.ctrl_range[:, 0], opt.ctrl_range[:, 1])

    X, U = opt.forward(x0, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    cost = []
    for i in range(50):
        J, X, U, A, B = opt.update(x0, X, U, A, B)
        print(i, J)
        cost.append(J)


    dyn.render(X, path = 'dog.mp4')

    fig, ax = plt.subplots(1, 1)
    ax.set_title('iLQR Trajectory Optimization')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Trajectory Cost')
    ax.plot(cost)
    fig.savefig('dog_cost.png', dpi = 300)
    plt.show()

    '''
    MPC
    '''
    # mpc = ModelPredictiveController(opt, 50)
    # xt = jnp.copy(x0)

    # steps = 500

    # X = jnp.zeros((steps + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # cost = []
    # for t in range(steps):
    #     ut, J = mpc(xt)
    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)
    #     # print(t, xt, ut, J)
    #     print(t, ut, J)
    #     cost.append(J)

    # fig, ax = plt.subplots(1, 1)
    # fig.suptitle('iLQR Trajectory Optimization')
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Trajectory Cost')
    # ax.plot(cost)
    # fig.savefig('dog_cost.png', dpi = 300)

    # dyn.render(X, path = 'dog.mp4')
