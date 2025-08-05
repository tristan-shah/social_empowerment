import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
from einops import einsum

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, waterfilling_implicit, compute_power

@jax.jit
def compute_F(A: Array):
    
    I = jnp.eye(A.shape[-1])

    def body_fun(_I: Array, a: Array):
        return _I @ a, _I @ a
    
    F, _ = jax.lax.scan(body_fun, I, A, reverse = True)
    return F

@jax.jit
def compute_sigma(G: Array, power: float):
    '''
    computes the covariance matrix of the control
    '''
    _, h, M = jnp.linalg.svd(G, full_matrices = False)
    h2 = (h ** 2).clip(min = 1e-12)
    v = waterfilling_implicit(h2, power)
    p = compute_power(v, h2)
    return M @ jnp.diag(p) @ M.T

def compute_empowerment(dyn: Dynamics, x0: Array, U: Array, power: float):
    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    ## this is a single action channel between the current action and the final state
    G = compute_F(A[1:]) @ B[0]
    sigma = compute_sigma(G, power)
    I = jnp.eye(dyn.state_dim)
    e = 0.5 * jnp.log(jnp.linalg.det(I + G @ sigma @ G.T))
    return e

if __name__ == '__main__':

    horizon = 5#400
    trials = 1000
    power = 1.0
    alpha = 1.0

    # dyn = Dynamics(path = 'xml/custom/pendulum.xml')
    # x0 = jnp.array([1.0, 0.0])
    # x0 = jnp.array([3.1, 0.0])
    # x0 = jnp.array([3.12, 0.0])
    # x0 = jnp.array([3.13, 0.0])
    # x0 = jnp.array([3.14, 0.0])

    dyn = Dynamics(path = 'xml/custom/double_pendulum.xml')
    # x0 = jnp.array([jnp.pi+0.00001, -0.001, 0.0, 0.0])
    x0 = jnp.array([jnp.pi+0.00001, -0.005, 0.0, 0.0])
    # x0 = jnp.array([jnp.pi+0.1, -0.001, 0.0, 0.0])


    dt = dyn.mjx_model.opt.timestep
    ## initial guess is zero control
    u_star = jnp.zeros((dyn.control_dim))

    hist = []
    for i in range(1):

        U_bar = jnp.zeros((horizon, dyn.control_dim))
        U_bar = U_bar.at[0].set(u_star)

        X = unroll(dyn, x0, U_bar)
        A, B = jax.vmap(dyn.linearize)(X[:-1], U_bar)
        # this is a single action channel between the current action and the final state
        G = compute_F(A[1:]) @ B[0]
        sigma = compute_sigma(G, power)
        I = jnp.eye(dyn.state_dim)
        e_bar = 0.5 * jnp.log(jnp.linalg.det(I + G @ sigma @ G.T))

        ## compute instantanious state hessians along the nominal trajectory
        Hxx = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 0), argnums = 0))(X[:-1], U_bar)
        ## second order sensitivity of x_0 to u_0 is zero
        Huu = jnp.zeros((dyn.state_dim, dyn.control_dim, dyn.control_dim))
        ## second order sensitivity of x_1 to u_0
        Huu = Huu + jax.jacfwd(jax.jacfwd(dyn.step, argnums = 1), argnums = 1)(X[0], U_bar[0])

        S = B[0]
        ## hessian propagation
        for t in range(1, horizon):
            ## second order update
            Huu = einsum(S, Hxx[t], S, 'x1 u1, f x1 x2, x2 u2 -> f u1 u2') + einsum(A[t], Huu, 'y x, x u1 u2 -> y u1 u2')
            ## propagate first order sensitivity of x_t to u_0
            S = A[t] @ S

        ## compute the base term which does not depend on control
        T = I + G @ sigma @ G.T
        T_inv = jnp.linalg.inv(T)

        ## compute the term that is linear in control
        vu = einsum(
            T_inv, 
            einsum(Huu, sigma, G, 'x1 u1 u2, u1 u3, x2 u3 -> u2 x1 x2'),
            'x1 x2, u1 x2 x3 -> u1 x1 x3')
        vu = jnp.trace(vu, axis1 = 1, axis2 = 2)

        ## compute the term that is quadratic in control
        Vuu = einsum(
            T_inv,
            einsum(Huu, sigma, Huu, 'x1 u1 u2, u1 u3, x2 u3 u4 -> u2 x1 x2 u4'),
            'x1 x2, u1 x2 x3 u2 -> u1 x1 x3 u2')
        Vuu = jnp.trace(Vuu, axis1 = 1, axis2 = 2)
        Vuu = Vuu.clip(max = 1000)

        ## solve the quadratic for the maximum
        u_star = jnp.linalg.inv(Vuu) @ (Vuu @ u_star + vu * alpha)
        u_star = jnp.clip(u_star, min = dyn.mjx_model.actuator_ctrlrange[:, 0], max = dyn.mjx_model.actuator_ctrlrange[:, 1])

        print(i, e_bar, vu, Vuu, u_star)
        hist.append(e_bar)

    fig, ax = plt.subplots(1, 1)
    ax.plot(hist)
    plt.show()

    '''
    plot empowerment as a function of control
    '''
    u0 = jnp.linspace(-1.0, 1.0, trials)[:, None]

    linear_term = (u0 - u_star) @ vu
    quadratic_term = einsum((u0 - u_star), Vuu, (u0 - u_star), 'b u1, u1 u2, b u2 -> b')

    e_hat_linear = e_bar + linear_term
    e_hat_quadratic = e_bar + linear_term + 0.5 * quadratic_term
    e_hat_neg_quadratic = e_bar + linear_term - 0.5 * quadratic_term

    U_batch = jnp.zeros((trials, horizon, dyn.control_dim))
    U_batch = U_batch.at[:, 0, :].set(u0)

    batch_compute_empowerment = jax.vmap(compute_empowerment, in_axes = (None, None, 0, None))
    e = batch_compute_empowerment(dyn, x0, U_batch, power)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(fr'Double Pendulum: $\theta_0={round(x0[0], 3)}$, $\theta_1={round(x0[1], 3)}$, Horizon $={horizon * dt}$ (s)')
    # ax.set_title(fr'Single Pendulum: $\theta={round(x0[0], 3)}$, Horizon $={horizon * dt}$ (s)')
    ax.set_xlabel(r'Control $u_0$')
    ax.set_ylabel('Empowerment')
    ax.plot(u0, e, label = 'True')
    ax.plot(u0, e_hat_linear, label = 'Linear')
    ax.plot(u0, e_hat_quadratic, label = 'Quadratic')
    ax.plot(u0, e_hat_neg_quadratic, label = 'Negative Quadratic')

    ax.set_ylim(e.min() * 0.9, e.max() * 1.1)
    ax.legend()
    fig.tight_layout()
    # fig.savefig(f'single_pendulum_horizon={horizon * dt}_x0={x0}.png', dpi = 300)
    # fig.savefig(f'double_pendulum_horizon={horizon * dt}_x0={x0}.png', dpi = 300)
    fig.savefig('short.png', dpi = 300)
    plt.show()