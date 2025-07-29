import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment

if __name__ == '__main__':


    ## hyperparams
    key = jax.random.key(5)
    alpha = 0.01
    empowerment_horizon = 100
    power = jnp.array([1.0, 1.0])
    agent_0_theta = 2.5

    ## load in xml
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    stiff = dyn.model.tendon_stiffness[0]

    resolution = 100
    theta = jnp.linspace(-2*jnp.pi, 2 * jnp.pi, resolution)
    theta_dot = jnp.linspace(-8 * jnp.pi, 8 * jnp.pi, resolution)

    U = jnp.zeros((empowerment_horizon, dyn.control_dim))

    E = jnp.zeros((resolution, resolution, 2))

    for i in range(len(theta)):
        for j in range(len(theta_dot)):
            x0 = jnp.array([agent_0_theta, theta[i], 0.0, theta_dot[j]])
            iterations, e = compute_multiagent_empowerment(dyn, x0, U, power, alpha, key)
            print(i, j, e)

            E = E.at[i, j].set(e)

    # Shared colormap and value range
    vmin = jnp.min(E)
    vmax = jnp.max(E)

    Theta, Theta_dot = jnp.meshgrid(theta, theta_dot, indexing = 'ij')

    ## plot the empowerment landscape
    # fig, ax = plt.subplots(1, 2, figsize = (12, 5))
    fig, ax = plt.subplots(1, 1, figsize = (6, 5))
    fig.suptitle(f'Horizon = {empowerment_horizon}, Stiffness = {stiff}')
    # fig.suptitle(f'Horizon = {empowerment_horizon}')

    ax.set_aspect('equal')
    ax.set_title(r'Empowerment of Agent 0 (Fixed $\theta_0 =$ ' + f'{round(agent_0_theta, 2)})')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\dot\theta_1$')

    # ax[0].set_aspect('equal')
    # ax[0].set_title(r'Empowerment of Agent 0 (Fixed $\theta_0 =$ ' + f'{round(agent_0_theta, 2)})')
    # ax[0].set_xlabel(r'$\theta_1$')
    # ax[0].set_ylabel(r'$\dot\theta_1$')

    # ax[1].set_title('Empowerment of Agent 1')
    # ax[1].set_aspect('equal')
    # ax[1].set_xlabel(r'$\theta_1$')
    # ax[1].set_ylabel(r'$\dot\theta_1$')

    fig.colorbar(
        # ax[0].imshow(
        ax.imshow(
            E[:, :, 0].T,
            extent = [theta[0].item(), theta[-1].item(), theta_dot[0].item(), theta_dot[-1].item()],
            origin = 'lower',
            aspect = 'auto',
            cmap = 'inferno',
            vmin=vmin,
            vmax=vmax
            ),
            label = 'Empowerment (nats)'
        )
    
    # fig.colorbar(
    #     ax[1].imshow(
    #         E[:, :, 1].T,
    #         extent = [theta[0].item(), theta[-1].item(), theta_dot[0].item(), theta_dot[-1].item()],
    #         origin = 'lower',
    #         aspect = 'auto',
    #         cmap = 'inferno',
    #         vmin=vmin,
    #         vmax=vmax
    #         ),
    #         label = 'Empowerment (nats)'
    #     )
    
    fig.tight_layout()
    fig.savefig(f'horizon={empowerment_horizon}_stiff={stiff}_agent_0_theta={round(agent_0_theta, 2)}.png', dpi = 300, bbox_inches='tight')