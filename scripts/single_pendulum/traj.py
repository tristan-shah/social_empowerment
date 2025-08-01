import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, get_last_state

if __name__ == '__main__':
    steps = 500
    trials = 100

    dyn = Dynamics(path = 'xml/custom/pendulum.xml')

    x0 = jnp.array([3.1, 0.1])
    U_bar = jnp.zeros((steps, dyn.control_dim))

    ## linearize around zero control
    X = unroll(dyn, x0, U_bar)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U_bar)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    '''
    Create a batch of control trajectories where the initial control 
    has a value and all the rest are zero across the horizon.
    '''
    u0 = jnp.linspace(-1.0, 1.0, trials)[:, None]
    U_batch = jnp.zeros((trials, steps, dyn.control_dim))
    U_batch = U_batch.at[:, 0, :].set(u0)

    xT_linear = X[-1] + jnp.einsum('x t u, b t u -> b x', F, U_batch)
    xT_true = jax.vmap(get_last_state, in_axes = (None, None, 0))(dyn, x0, U_batch)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel(r'Control $u_0$')
    ax[0].set_ylabel(r'$\theta$ at time T')
    ax[0].plot(u0, xT_linear[:, 0], label = 'Linearization')
    ax[0].plot(u0, xT_true[:, 0], label = 'True')
    ax[0].legend()

    ax[1].set_xlabel(r'Control $u_0$')
    ax[1].set_ylabel(r'$\dot\theta$ at time T')
    ax[1].plot(u0, xT_linear[:, 1], label = 'Linearization')
    ax[1].plot(u0, xT_true[:, 1], label = 'True')
    ax[1].legend()
    fig.tight_layout()
    plt.show()