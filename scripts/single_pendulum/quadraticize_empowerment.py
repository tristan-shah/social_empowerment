import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum
from mujoco import mjx
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F_from_A_B, waterfilling_implicit, compute_power
from soc_emp.utils import split_state, get_state


def make_step(dyn: Dynamics):

    model = dyn.mjx_model
    data_template = mjx.make_data(model)
    nq = dyn.nq

    def step(xt: Array, ut: Array):
        qpos, qvel = split_state(xt, nq)
        data = data_template.replace(qpos = qpos, qvel = qvel, ctrl = ut)
        data = mjx.step(model, data)
        return get_state(data)
    
    return jax.jit(step)

def make_unroll(step: callable, U: Array):
        
    def unroll(xt: Array):
        '''
        Jax compatable simulation loop.
        '''

        def body_fun(xt_: Array, ut_: Array):
            xt_next = step(xt_, ut_)
            return xt_next, xt_next
        
        _, X = jax.lax.scan(body_fun, xt, U)
        return jnp.concatenate([xt[None, :], X])
        
    return jax.jit(unroll)

def make_compute_empowerment(dyn: Dynamics, T: int, P: float):

    U = jnp.zeros((T, dyn.control_dim))

    step = make_step(dyn)
    unroll = make_unroll(step, U)
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))

    def compute_empowerment(xt: Array):
        X = unroll(xt)
        fx, fu = linearize(X[:-1], U)
        F = compute_F_from_A_B(fx, fu)
        F = jnp.permute_dims(F, (1, 0, 2))
        S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
        h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
        v = waterfilling_implicit(h2, P)
        p = compute_power(v, h2)
        e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
        return e
    
    return jax.jit(compute_empowerment)

if __name__ == '__main__':
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.01)

    step = make_step(dyn)

    T = 50
    P = 1.0

    E = make_compute_empowerment(dyn, T, P)

    '''
    dead center
    '''
    # Ex = jax.jacfwd(E)
    # Exx = jax.jacfwd(Ex)
    x0 = jnp.array([jnp.pi, 0.0])
    # print(E(x0))
    # print(Ex(x0))
    # print(Exx(x0))


    e0 = jnp.array(0.06286619121888896)
    ex = jnp.array([1.84098266e-17, 3.97343916e-18])
    exx = jnp.array(
            [[-0.35558834, -0.07909864],
            [-0.07909864, -0.01778186]])
    
    eigvals, eigvecs = jnp.linalg.eig(exx)


    print('Eigenvalues:')
    print(eigvals.real)
    print('Eigenvectors:')
    print(eigvecs.real)

    # # ======================================================
    # # Build grid
    # # ======================================================

    # theta_min, theta_max = 0.0, 2 * jnp.pi
    # theta_dot_min, theta_dot_max = -4 * jnp.pi, 4 * jnp.pi

    # n_theta = 120
    # n_theta_dot = 120

    # theta_grid = jnp.linspace(theta_min, theta_max, n_theta)
    # theta_dot_grid = jnp.linspace(theta_dot_min, theta_dot_max, n_theta_dot)

    # Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)

    # # Shape: (n_theta*n_theta_dot, 2)
    # grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    # # ======================================================
    # # Full empowerment evaluation
    # # ======================================================
    # e_full = jax.vmap(E)(grid_points)
    # e_full = e_full.reshape(n_theta_dot, n_theta)

    # # ======================================================
    # # Taylor approximation
    # # ======================================================
    # delta = grid_points - x0

    # # Quadratic expansion
    # e_taylor = (
    #     e0
    #     + delta @ ex
    #     + 0.5 * jnp.sum((delta @ exx) * delta, axis=1)
    # )

    # e_taylor = e_taylor.reshape(n_theta_dot, n_theta)

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # Set shared z-limits for better comparison
    # zmin = min(e_full.min(), e_taylor.min())
    # zmax = max(e_full.max(), e_taylor.max())
    # ax.set_zlim(zmin, zmax)

    # # Plot full empowerment as a solid surface
    # surf_full = ax.plot_surface(
    #     Theta, Theta_dot, e_full,
    #     cmap='inferno', alpha=0.9, linewidth=0, antialiased=True
    # )

    # # Plot Taylor approximation as a wireframe for clarity
    # wire_taylor = ax.plot_wireframe(
    #     Theta, Theta_dot, e_taylor,
    #     color='cyan', linewidth=1, rstride=3, cstride=3
    # )

    # # Labels and title
    # ax.set_xlabel("Theta")
    # ax.set_ylabel("Theta Dot")
    # ax.set_zlabel("Empowerment E")
    # ax.set_title("Full Empowerment vs Taylor Approximation at (π,0)")
    # ax.set_zlim(0.01, 0.06)

    # # Add a colorbar for the full surface
    # fig.colorbar(surf_full, ax=ax, shrink=0.6, pad=0.1, label='Full E(x)')

    # # Optional: rotate the view for better perspective
    # ax.view_init(elev=35, azim=45)

    # plt.show()