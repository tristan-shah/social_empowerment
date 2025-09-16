import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum, rearrange

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, waterfilling_implicit, compute_power, compute_F_from_A_B

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

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    horizon = 50
    power = 1.0
    steps = 1500

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.01, integrator = 'euler')

    I = jnp.eye(dyn.state_dim)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(-1.0)
    U = jnp.zeros((horizon, dyn.control_dim)) + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.1

    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    noise = I * 1.0

    ## compute instantanious state hessians along the nominal trajectory
    # Hxx = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 0), argnums = 0))(X[:-1], U)
    # Hxu = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 0), argnums = 1))(X[:-1], U)
    # Hux = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 1), argnums = 0))(X[:-1], U)
    # Huu = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = 1), argnums = 1))(X[:-1], U)

    (Hxx, Hxu), (Hux, Huu) = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = (0, 1)), argnums = (0, 1)))(X[:-1], U)

    print(Hxx.shape, Hxu.shape, Hux.shape, Huu.shape)

    P = jnp.zeros((dyn.state_dim, dyn.state_dim))

    damping = jnp.eye(dyn.control_dim) * 1e-5

    for t in reversed(range(horizon)):
        
        G = B[t]
        sigma = compute_sigma(G, power)
        T = I + G @ sigma @ G.T
        T_inv = jnp.linalg.inv(T)

        ## computing cross term
        F = T_inv @ einsum(Huu[t], sigma, Hxx[t], 'x1 u1 u2, u1 u3, x2 u3 x3 -> u2 x1 x2 x3')
        F = jnp.trace(F, axis1 = 1, axis2 = 2)

        ## computing quadratic state term
        W = T_inv @ einsum(Huu[t], sigma, Huu[t], 'x1 u1 u2, u1 u3, x2 u3 u4 -> u2 x1 x2 u4')
        W = jnp.trace(W, axis1 = 1, axis2 = 2)
        
        grad_pi = - jnp.linalg.inv(W + damping + B[t].T @ P @ B[t]) @ (B[t].T @ P @ A[t] + F)
        print(grad_pi)
        break


    # # # print(sigma.shape)
    # # # T = I + G @ sigma @ G.T
    # # # T_inv = jnp.linalg.inv(T)
    # # # print(T_inv)
    # # # print(G @ sigma @ G.T)