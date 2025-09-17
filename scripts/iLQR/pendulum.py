from jax import numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll
from soc_emp.utils import smooth_angle_wrap

from ilqr import iLQR

def compute_pendulum_error(X: Array):
    assert X.shape[1] == 2

    r = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        X[:, 1]
    ], axis = 1)

    assert r.shape == X.shape
    return r

if __name__ == '__main__':

    horizon = 200

    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.01, integrator = 'euler')

    Q = jnp.eye(dyn.state_dim)
    R = jnp.eye(dyn.control_dim)

    opt = iLQR(dyn, Q, R)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(1.0)
    U = jnp.zeros((horizon, dyn.control_dim))# + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.1

    X = unroll(dyn, xt, U)

    A, B = opt.batch_linearize(X[:-1], U)

    e = compute_pendulum_error(X)
    k, K, P_hist = opt.backward(e, A, B, U)

    print(K)

    fig, ax = plt.subplots(1, 2)

    fig.suptitle(f'Initial Theta: {xt[0]}')
    ax[0].set_title('Policy Gradient')
    ax[0].set_xlabel('Horizon')
    ax[0].plot(K[:, 0, 0])
    ax[0].plot(K[:, 0, 1])

    ax[1].set_title('Riccati Solution')
    ax[1].set_xlabel('Horizon')
    ax[1].plot(P_hist[:, 0, 0])
    ax[1].plot(P_hist[:, 0, 1])
    ax[1].plot(P_hist[:, 1, 0])
    ax[1].plot(P_hist[:, 1, 1])
    plt.show()