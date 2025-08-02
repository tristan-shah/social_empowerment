import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from einops import einsum

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, get_last_state

if __name__ == '__main__':
    steps = 500
    trials = 1000

    dyn = Dynamics(path = 'xml/custom/linked_pendulums.xml')
    # dyn = Dynamics(path = 'xml/custom/pendulum.xml')

    x0 = jnp.array([jnp.pi - 0.001, jnp.pi, 0.0, 0.0])
    U_bar = jnp.zeros((steps, dyn.control_dim))

    ## linearize around zero control
    X_bar = unroll(dyn, x0, U_bar)
    A, B = jax.vmap(dyn.linearize)(X_bar[:-1], U_bar)

    Hxx = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 0), argnums = 0))(X_bar[:-1], U_bar)
    
    Huu = jnp.zeros((dyn.state_dim, dyn.control_dim, dyn.control_dim)) ## Huu(0)
    Huu = Huu + jax.jacfwd(jax.jacfwd(dyn.step, argnums = 1), argnums = 1)(X_bar[0], U_bar[0]) ## Huu(1)

    ## hessian propagation
    S = B[0]
    for t in range(1, steps):
        Huu = einsum(S, Hxx[t], S, 'x1 u1, f x1 x2, x2 u2 -> f u1 u2') + einsum(A[t], Huu, 'y x, x u1 u2 -> y u1 u2')
        S = A[t] @ S
        print(t)

    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    # dyn.render(X_bar, path = 'test.mp4', skip = 3)

    '''
    Create a batch of control trajectories where the initial control 
    has a value and all the rest are zero across the horizon.
    '''
    u0 = jnp.stack([
        jnp.linspace(-1.0, 1.0, trials),
        jnp.zeros(trials),
    ], axis = 1)

    print(u0.shape)

    U_batch = jnp.zeros((trials, steps, dyn.control_dim))
    U_batch = U_batch.at[:, 0, :].set(u0)
    print(U_batch.shape)

    constant_term = X_bar[-1]
    linear_term = einsum(F, U_batch, 'x t u, b t u -> b x')
    quadratic_term = 0.5 * einsum(u0, Huu, u0, 'b u1, x u1 u2, b u2 -> b x')

    xT_quadratic = constant_term + linear_term + quadratic_term
    xT_true = jax.vmap(get_last_state, in_axes = (None, None, 0))(dyn, x0, U_batch)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f'x0 = {x0}')
    ax[0, 0].set_title('Left Pendulum')
    ax[0, 0].set_xlabel(r'Control $u_0$')
    ax[0, 0].set_ylabel(fr'$\theta_0$ at t = {steps+1}')
    ax[0, 0].plot(u0[:, 0], xT_true[:, 0], label = 'True')
    ax[0, 0].plot(u0[:, 0], xT_quadratic[:, 0], label = 'Quadratic')
    ax[0, 0].set_ylim(-10.0, 10.0)
    # ax[0].legend()

    ax[0, 1].set_title('Right Pendulum')
    ax[0, 1].set_xlabel(r'Control $u_0$')
    ax[0, 1].set_ylabel(fr'$\theta_1$ at t = {steps+1}')
    ax[0, 1].plot(u0[:, 0], xT_true[:, 1])
    ax[0, 1].plot(u0[:, 0], xT_quadratic[:, 1])
    ax[0, 1].set_ylim(-10.0, 10.0)
    # ax[1].legend()

    ax[1, 0].set_title('Left Pendulum')
    ax[1, 0].set_xlabel(r'Control $u_0$')
    ax[1, 0].set_ylabel(fr'$\dot\theta_0$ at t = {steps+1}')
    ax[1, 0].plot(u0[:, 0], xT_true[:, 2])
    ax[1, 0].plot(u0[:, 0], xT_quadratic[:, 2])
    ax[1, 0].set_ylim(-10.0, 10.0)
    # ax[2].legend()

    ax[1, 1].set_title('Right Pendulum')
    ax[1, 1].set_xlabel(r'Control $u_0$')
    ax[1, 1].set_ylabel(fr'$\dot\theta_1$ at t = {steps+1}')
    ax[1, 1].plot(u0[:, 0], xT_true[:, 3])
    ax[1, 1].plot(u0[:, 0], xT_quadratic[:, 3])
    ax[1, 1].set_ylim(-10.0, 10.0)
    # ax[3].legend()

    fig.legend()
    fig.tight_layout()
    plt.show()