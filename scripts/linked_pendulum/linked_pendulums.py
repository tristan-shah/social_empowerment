import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad
from soc_emp.utils import smooth_angle_wrap

def kinetic_energy(xt: Array):
    omega_0 = xt[2]
    omega_1 = xt[3]
    return 0.5 * jnp.stack([omega_0, omega_1]) ** 2

def potential_energy(xt: Array):
    theta_0 = xt[0]
    theta_1 = xt[1]
    g = 9.81
    return g * (1 - jnp.cos(jnp.stack([theta_0, theta_1])))

def is_up(pe: Array):
    return (pe[-50:] >= 15.0).all()

def plot_outcome_hetamap(m: Array, horizon: int, powers: Array, path: str):
        
    # Custom colormap and norm
    colors = ['lightgray', 'blue', 'orange', 'green']
    labels = ['Neither', 'Left', 'Right', 'Both']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(m, cmap=cmap, norm=norm, origin = 'lower')

    # Convert powers to numpy and determine tick spacing
    powers_np = np.array(powers)
    full_tick_positions = np.arange(len(powers))
    tick_spacing = 4  # show every 4th tick (adjust as needed)

    # Select every nth tick
    sparse_tick_positions = full_tick_positions[::tick_spacing]
    sparse_tick_labels = np.round(powers_np[::tick_spacing], 2)

    # Set sparse ticks and labels
    ax.set_xticks(sparse_tick_positions)
    ax.set_xticklabels(sparse_tick_labels, rotation=90)
    ax.set_yticks(sparse_tick_positions)
    ax.set_yticklabels(sparse_tick_labels)

    ax.set_xlabel('Right Agent Power')
    ax.set_ylabel('Left Agent Power')
    ax.set_title(f'Horizon = {horizon}')

    # Add colorbar with custom ticks/labels
    cbar = plt.colorbar(img, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('Pendulum Up')

    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return

def plot_iteration_hetamap(m: Array, horizon: int, powers: Array, path: str):

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(m, origin = 'lower')

    # Convert powers to numpy and determine tick spacing
    powers_np = np.array(powers)
    full_tick_positions = np.arange(len(powers))
    tick_spacing = 4  # show every 4th tick (adjust as needed)

    # Select every nth tick
    sparse_tick_positions = full_tick_positions[::tick_spacing]
    sparse_tick_labels = np.round(powers_np[::tick_spacing], 2)

    # Set sparse ticks and labels
    ax.set_xticks(sparse_tick_positions)
    ax.set_xticklabels(sparse_tick_labels, rotation=90)
    ax.set_yticks(sparse_tick_positions)
    ax.set_yticklabels(sparse_tick_labels)

    ax.set_xlabel('Right Agent Power')
    ax.set_ylabel('Left Agent Power')
    ax.set_title(f'Horizon = {horizon}')

    # Add colorbar with custom ticks/labels
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('IWF Iterations')

    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return



if __name__ == '__main__':

    ## hyperparams
    key = jax.random.key(5)
    T = 1500 ## simulation horizon
    num_agents = 2
    alpha = 0.01

    ## load in xml
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    print(dyn.mjx_model.opt.timestep)

    # print(f'Timestep = {dyn.mjx_model.opt.timestep}')

    # horizons = jnp.arange(50, 150, 10)
    # powers = jnp.linspace(0.5, 3.0, 30)

    # outcome_heatmap = jnp.zeros((len(horizons), len(powers), len(powers)))
    # iteration_heatmap = jnp.zeros((len(horizons), len(powers), len(powers)))

    # for i in range(len(powers)):
    #     for j in range(len(powers)):
    #         for k in range(len(horizons)):

    #             # empowerment_horizon = horizons[k]
    #             # power = jnp.array([powers[i], powers[j]])

    #             # empowerment_horizon = 200
    #             empowerment_horizon = 50
    #             power = jnp.array([1.0, 2.0])

    #             U = jnp.zeros((empowerment_horizon, dyn.control_dim))

    #             dx = dyn.state_dim
    #             du = dyn.control_dim // num_agents
    #             xt = dyn.init_state()

    #             ## tensor for state storage
    #             X = jnp.zeros((T + 1, dyn.state_dim))
    #             X = X.at[0].set(xt)

    #             iter_hist = jnp.zeros((T,))
    #             emp_hist = jnp.zeros((T, num_agents))
    #             ke_hist = jnp.zeros((T, num_agents))
    #             pe_hist = jnp.zeros((T, num_agents))

    #             print(xt)
    #             print(jax.devices())

    #             for t in range(T):

    #                 ## compute empowerment
    #                 iterations, e = compute_multiagent_empowerment(dyn, xt, U, power, alpha, key)

    #                 ## obtain control gain
    #                 _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) 
    #                 grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, key)

    #                 ## compute action (full power)
    #                 ut = jnp.sign(jnp.diag(grad_E @ B)) * power

    #                 # Generate random values for all entries
    #                 # key, subkey = jax.random.split(key)
    #                 # rand = jax.random.normal(subkey, shape=ut.shape)
    #                 ut = ut + (ut == 0) * power # * jnp.sign(rand)

    #                 ## log stuff
    #                 emp_hist = emp_hist.at[t].set(e)
    #                 iter_hist = iter_hist.at[t].set(iterations)
    #                 ke_hist = ke_hist.at[t].set(kinetic_energy(xt))
    #                 pe_hist = pe_hist.at[t].set(potential_energy(xt))

    #                 ## propagate dynamics
    #                 xt = dyn.step(xt, ut)
    #                 print(t, xt, ut, e, iterations)

    #                 ## log state
    #                 X = X.at[t+1].set(xt)

    #             name = f'left={power[0]}-right={power[1]}-horizon={empowerment_horizon}'

    #             fig, ax = plt.subplots(2, 1)
    #             fig.suptitle(f'Horizon = {empowerment_horizon}')
    #             # First subplot: Empowerment
    #             ax[0].plot(emp_hist[:, 0], label='Left Agent', color='blue')
    #             ax[0].plot(emp_hist[:, 1], label='Right Agent', color='orange')
    #             ax[0].set_ylabel('Empowerment', fontsize=14)
    #             ax[0].tick_params(axis='both', labelsize=12)
    #             ax[0].legend(fontsize=12)

    #             agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    #             agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    #             ax[1].plot(agent_0_angle, color = 'blue')
    #             ax[1].plot(agent_1_angle, color = 'orange')
    #             ax[1].set_xlabel('Timestep', fontsize = 14)
    #             ax[1].set_ylabel('Angle From Top', fontsize = 14)
    #             ax[1].tick_params(axis='both', labelsize = 12)

    #             fig.tight_layout()
    #             # fig.savefig(f'results/{name}.png', dpi = 300)

    #             fig.savefig(f'test.png', dpi = 300)
    #             plt.show()
    #             plt.close(fig)

    #             break
    #         break
    #     break

    #             # # skip = 2
    #             # # dyn.render(
    #             # #     X,
    #             # #     path = f'results/{name}.mp4',
    #             # #     skip = skip)
                

    #             # left_up = is_up(pe_hist[:, 0])
    #             # right_up = is_up(pe_hist[:, 1])
    #             # neither_up = not left_up and not right_up
    #             # both_up = left_up and right_up

    #             # if neither_up:
    #             #     outcome = 0
    #             # elif left_up and not right_up:
    #             #     outcome = 1
    #             # elif not left_up and right_up:
    #             #     outcome = 2
    #             # elif both_up:
    #             #     outcome = 3
                
    #             # outcome_heatmap = outcome_heatmap.at[k, i, j].set(outcome)
    #             # plot_outcome_hetamap(outcome_heatmap[k], horizons[k], powers, path = f'results/horizon={empowerment_horizon}_outcome_heatmap.png')
                
    #             # iteration_heatmap = iteration_heatmap.at[k, i, j].set(jnp.mean(iter_hist))
    #             # plot_iteration_hetamap(iteration_heatmap[k], horizons[k], powers, path = f'results/horizon={empowerment_horizon}_iteration_heatmap.png')