import jax
from jax import numpy as jnp
from jax import Array
from mujoco import mjx

from soc_emp.dynamics import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer

def compute_google_barkour_vb_error(X: Array):
    goal = jnp.zeros(X.shape[1])
    goal = goal.at[0].set(1.0)
    r = X - goal
    return r

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    dyn = Dynamics(path = 'mujoco_menagerie/google_barkour_vb/scene_mjx.xml')

    mjx_data = mjx.make_data(dyn.mjx_model)

    x0 = jnp.concatenate([
        mjx_data.qpos,
        mjx_data.qvel
    ])

    x0 = x0.at[0].set(-1.0)
    x0 = x0.at[2].set(0.1)

    Q = jnp.zeros((dyn.state_dim, dyn.state_dim)).at[0, 0].set(1.0)
    R = jnp.eye(dyn.control_dim)

    opt = TrajectoryOptimizer(
        dyn, 
        compute_google_barkour_vb_error,
        Q,
        R)

    T = 500
    U = jax.random.normal(key, (T, dyn.control_dim))

    X, U = opt.forward(x0, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    for i in range(50):
        J, X, U, A, B = opt.update(x0, X, U, A, B)
        print(i, J)

    # X = jnp.zeros((T + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # for t in range(T):
    #     ut = jnp.zeros(dyn.control_dim)
    #     xt = dyn.step(xt, ut)
    #     print(xt)
    #     X = X.at[t+1].set(xt)

    # r = X - goal

    # print(r.shape)
    # print(r @ Q)

    dyn.render(X, path = 'dog.mp4')