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
    steps = 100
    trials = 1000
    power = 1.0

    dyn = Dynamics(path = 'xml/custom/pendulum.xml')

    x0 = jnp.array([jnp.pi-0.01, 0.0])
    U_bar = jnp.zeros((steps+1, dyn.control_dim))
    X = unroll(dyn, x0, U_bar)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U_bar)
    # this is a single action channel between the current action and the final state
    G = compute_F(A[1:]) @ B[0]
    sigma = compute_sigma(G, power)
    I = jnp.eye(dyn.state_dim)
    e = 0.5 * jnp.log(jnp.linalg.det(I + G @ sigma @ G.T))

    Hxx = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 0), argnums = 0))(X[:-1], U_bar)    
    Huu = jnp.zeros((dyn.state_dim, dyn.control_dim, dyn.control_dim)) ## Huu(0)
    Huu = Huu + jax.jacfwd(jax.jacfwd(dyn.step, argnums = 1), argnums = 1)(X[0], U_bar[0]) ## Huu(1)

    ## hessian propagation
    S = B[0]
    for t in range(1, steps):
        Huu = einsum(S, Hxx[t], S, 'x1 u1, f x1 x2, x2 u2 -> f u1 u2') + einsum(A[t], Huu, 'y x, x u1 u2 -> y u1 u2')
        S = A[t] @ S

    T = I + G @ sigma @ G.T
    T_inv = jnp.linalg.inv(T)
    
    vu = einsum(
        T_inv, 
        einsum(Huu, sigma, G, 'x1 u1 u2, u1 u3, x2 u3 -> u2 x1 x2'), 
        'x1 x2, u1 x2 x3 -> u1 x1 x3'
    )

    vu = jnp.trace(vu, axis1 = 1, axis2 = 2)

    Vuu = einsum(
        T_inv,
        einsum(Huu, sigma, Huu, 'x1 u1 u2, u1 u3, x2 u3 u4 -> u2 x1 x2 u4'),
        'x1 x2, u1 x2 x3 u2 -> u1 x1 x3 u2'
    )

    Vuu = jnp.trace(Vuu, axis1 = 1, axis2 = 2)

    print(e)
    print(vu)
    print(Vuu)


    '''
    plot empowerment as a function of control
    '''
    u0 = jnp.linspace(-1.0, 1.0, trials)[:, None]

    e_hat_linear = e + u0 @ vu
    e_hat_quadratic = e + u0 @ vu - 0.5 * einsum(u0, Vuu, u0, 'b u1, u1 u2, b u2 -> b')

    U_batch = jnp.zeros((trials, steps+1, dyn.control_dim))
    U_batch = U_batch.at[:, 0, :].set(u0)

    batch_compute_empowerment = jax.vmap(compute_empowerment, in_axes = (None, None, 0, None))
    e = batch_compute_empowerment(dyn, x0, U_batch, power)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r'Control $u_0$')
    ax.set_ylabel('Empowerment')
    ax.plot(u0, e)
    ax.plot(u0, e_hat_linear)
    ax.plot(u0, e_hat_quadratic)
    plt.show()