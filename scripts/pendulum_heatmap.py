import jax
from jax import numpy as jnp
from jax import Array
from mujoco import mjx
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_empowerment, compute_empowerment_grad

if __name__ == '__main__':
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    T = 25 #50
    P = 1.0

    U = jnp.zeros((T, dyn.control_dim))

    resolution = 50
    theta = jnp.linspace(0.0, 2 * jnp.pi, resolution)
    theta_dot = jnp.linspace(-4 * jnp.pi, 4 * jnp.pi, resolution)

    grad_E = jnp.zeros((resolution, resolution, dyn.state_dim))
    empowerment_landscape = jnp.zeros((resolution, resolution))

    for i in range(len(theta)):
        for j in range(len(theta_dot)):
            x0 = jnp.array([theta[i], theta_dot[j]])

            grad_e = compute_empowerment_grad(dyn, x0, U, P)
            grad_E = grad_E.at[i, j].set(grad_e)

            e = compute_empowerment(dyn, x0, U, P)
            empowerment_landscape = empowerment_landscape.at[i, j].set(e)
            print(i, j, e, grad_e)
            # print(i, j, e)


    Theta, Theta_dot = jnp.meshgrid(theta, theta_dot, indexing = 'ij')
    # Extract gradient components
    grad_theta = grad_E[:, :, 0]
    grad_theta_dot = grad_E[:, :, 1]

    # Compute gradient magnitudes
    norm = jnp.sqrt(grad_theta**2 + grad_theta_dot**2) + 1e-8

    # Clamp the magnitudes to a reasonable range
    min_len = 0.1
    max_len = 1.0
    norm_clamped = jnp.clip(norm, min = min_len, max = max_len)

    # Rescale gradients to use clamped magnitudes
    grad_theta = grad_theta / norm * norm_clamped
    grad_theta_dot = grad_theta_dot / norm * norm_clamped


    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'MuJoCo Empowerment Landscape Horizon = {T}')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta Dot')
    fig.colorbar(
        ax.imshow(
            empowerment_landscape.T,
            extent=[theta[0].item(), theta[-1].item(), theta_dot[0].item(), theta_dot[-1].item()],
            origin='lower',
            aspect='auto',
            cmap = 'inferno'
            )
        )

    ## Gradient field (vector field)
    ax.quiver(
        Theta,
        Theta_dot,
        grad_theta,
        grad_theta_dot,
        color ='white',
        pivot = 'middle',
        alpha = 0.9,
        scale = 10,
        width = 0.003
    )
    
    plt.show()