import jax
from jax import numpy as jnp
from jax import Array
from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.ilqr import TrajectoryOptimizer
from soc_emp.utils import smooth_angle_wrap

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

    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    x0 = jnp.zeros((dyn.state_dim,))
    x0 = x0.at[0].set(0.0)

    T = 400
    Q = jnp.eye(dyn.state_dim)
    R = jnp.eye(dyn.control_dim) * 0.005
    Q = Q.at[1, 1].set(0.0)

    opt = TrajectoryOptimizer(
        dyn, 
        compute_pendulum_error,
        Q, 
        R)
    
    U = jax.random.normal(key, (T, dyn.control_dim))

    X, U = opt.forward(x0, U = U)
    A, B = opt.batch_linearize(X[:-1], U)

    for i in range(50):
        J, X, U, A, B = opt.update(x0, X, U, A, B)
        print(i, J)
    
    import matplotlib.pyplot as plt
    plt.plot(U)
    plt.show()
    dyn.render(X, path = 'pendulum.mp4')