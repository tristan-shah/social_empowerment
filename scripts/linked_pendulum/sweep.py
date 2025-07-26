import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad
from soc_emp.utils import smooth_angle_wrap

def get_pendulum_outcome(traj: Array):
    # assert traj.ndim  == 2

    ## check angle from the bottom (0.0 rad). top is jnp.pi rad
    left_angle_from_bottom = jnp.abs(smooth_angle_wrap(traj[:, 0]))
    right_angle_from_bottom = jnp.abs(smooth_angle_wrap(traj[:, 1]))

    ## get final state
    left_up = jnp.all(left_angle_from_bottom[-50:] >= 2.1)
    right_up = jnp.all(right_angle_from_bottom[-50:] >= 2.1)

    neither_up = jnp.logical_and(jnp.logical_not(left_up), jnp.logical_not(right_up))
    both_up = jnp.logical_and(left_up, right_up)

    outcome = jnp.where(neither_up, 0,
                jnp.where(jnp.logical_and(left_up, jnp.logical_not(right_up)), 1,
                jnp.where(jnp.logical_and(jnp.logical_not(left_up), right_up), 2,
                3)))

    return outcome

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
    ax.set_xticklabels(sparse_tick_labels, rotation = 90)
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

def run_multiagent_empowerment(dyn: Dynamics, U: Array, power: Array, alpha: float, steps: int, key):

    xt = dyn.init_state()

    @jax.jit
    def _step_linked_pendulums(_xt: Array, _):

        ## obtain control gain
        _, B = dyn.linearize(_xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, _xt, U, power, alpha, key)

        ## compute action (full power)
        ut = jnp.sign(jnp.diag(grad_E @ B)) * power

        # Generate random values for all entries if gradient is zero
        ut = ut + (ut == 0) * power

        ## propagate dynamics
        _xt = dyn.step(_xt, ut)
        return (_xt, _xt)
    
    _, X = jax.lax.scan(_step_linked_pendulums, xt, length = steps)
    return jnp.concatenate([xt[None, :], X])

'''
absolute value of the angle from the top should be less than 1 rad.
angular velocity should be less than 2 rad / sec.
'''

if __name__ == '__main__':

    ## check if gpu device is available
    print(jax.devices())

    ## hyperparams
    key = jax.random.key(5)
    steps = 1500 ## simulation horizon
    num_agents = 2
    alpha = 0.01

    ## load in xml
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    print(f'Timestep = {dyn.mjx_model.opt.timestep}')

    ## create variables to sweep over
    horizons = jnp.arange(50, 150, 10)
    powers = jnp.linspace(0.5, 3.0, 10)

    ## check a particular setting
    horizon = 100#horizons[0]
    power = jnp.array([2.0, 1.0])

    ## build a sequence of zero actions
    U = jnp.zeros((horizon, dyn.control_dim))

    num_powers = powers.shape[0]
    I, J = jnp.meshgrid(jnp.arange(num_powers), jnp.arange(num_powers), indexing = 'ij')
    pairs = jnp.stack([powers[I], powers[J]], axis = -1).reshape(-1, 2)

    ## create a function that will execute run_multi_agent_empowerment in parallel
    batch_run_multiagent_empowerment = jax.vmap(
        run_multiagent_empowerment, 
        in_axes = (None, None, 0, None, None, 0))
    
    batch_get_pendulum_outcome = jax.vmap(get_pendulum_outcome)
    
    ## broadcast function over pairs and keys
    keys = jax.random.split(key, pairs.shape[0])
    X = batch_run_multiagent_empowerment(dyn, U, pairs, alpha, steps, keys)

    outcomes = batch_get_pendulum_outcome(X).reshape(num_powers, num_powers)
    print(outcomes)
    plot_outcome_hetamap(outcomes, horizon, powers, path = f'horizon={horizon}_outcome_heatmap.png')

    # time = horizon * dyn.mjx_model.opt.timestep

    '''
    loop code
    '''
    # outcome_heatmap = jnp.zeros((len(horizons), len(powers), len(powers)))
    # iteration_heatmap = jnp.zeros((len(horizons), len(powers), len(powers)))

    # for i in range(len(powers)):
    #     for j in range(len(powers)):
    #         for k in range(len(horizons)):

    #             # empowerment_horizon = horizons[k]
    #             # power = jnp.array([powers[i], powers[j]])

    #             # empowerment_horizon = 200
    #             empowerment_horizon = 25
    #             power = jnp.array([1.5, 1.5])

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
    #             ax[0].plot(emp_hist[:, 0], label='Agent 0', color='blue')
    #             ax[0].plot(emp_hist[:, 1], label='Agent 1', color='orange')
    #             ax[0].set_ylabel('Empowerment (Nats)', fontsize=14)
    #             ax[0].tick_params(axis='both', labelsize=12)
    #             ax[0].legend(fontsize=12)

    #             agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    #             agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    #             ax[1].plot(agent_0_angle, color = 'blue')
    #             ax[1].plot(agent_1_angle, color = 'orange')
    #             ax[1].set_xlabel('Timestep', fontsize = 14)
    #             ax[1].set_ylabel('Angle From Top (Rad)', fontsize = 14)
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