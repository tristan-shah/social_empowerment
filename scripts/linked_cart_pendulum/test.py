import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.utils import smooth_angle_wrap


'''
To recover a near-rigid scenario we must set the anchor spring stiffness to around (15000.0) and the damping to around (500.0).
'''

'''
[
    ## qpos: left_position [0], left_angle [1], right_position [2], right_angle [3],
    ## qvel: left_position_vel [4], left_angular_vel [5], right_position_vel [6], right_angular_vel [7]
]

left_state = [0, 1, 4, 5]
right_state = [2, 3, 6, 7]
'''

from soc_emp.empowerment import unroll, batch_diag, compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, select_output, compute_multiagent_control

LEFT_AGENT_STATE = jnp.array([0, 1, 4, 5])
RIGHT_AGENT_STATE = jnp.array([2, 3, 6, 7])

def compute_multiagent_empowerment(
        dyn: Dynamics, 
        x0: Array, 
        U: Array, 
        power: Array, 
        alpha: float,
        observation_noise: float):

    num_agents = len(power)
    horizon = U.shape[0]
    du = dyn.control_dim // num_agents
    dm = du * horizon

    '''
    original
    '''
    S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)

    # hardcoded noise perturbation
    S_z = jnp.eye(4) * observation_noise

    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    F_agent, F_noise = split_channel_matrix(F, num_agents)

    '''
    egoistic double pendulum. Each agent only cares about its own state (angle, angular velocity).
    '''
    F_agent = jnp.stack([
        F_agent[0, LEFT_AGENT_STATE, :],
        F_agent[1, RIGHT_AGENT_STATE, :]
        ], axis = 0)
    
    ## "direct indexing" produces some unintuitive slices when indexes are seperated by ":"
    ## THIS CODE WILL PRODUCE INCORRECT RESULTS. IM LEAVING IT HERE AS AN EXAMPLE OF WHAT NOT TO DO.
    # F_noise = jnp.stack([
    #     F_noise[0, :, [0, 2], :],
    #     F_noise[1, :, [1, 3], :]
    # ], axis = 0)

    ## chained indexing allows to select the correct submatrices
    F_noise = jnp.stack([
        F_noise[0][:, LEFT_AGENT_STATE, :],
        F_noise[1][:, RIGHT_AGENT_STATE, :]
    ], axis = 0)

    i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)
    return i, e, S

compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = 0)
compute_multiagent_empowerment_grad = jax.jit(
    jax.jacfwd(
        select_output(compute_multiagent_empowerment, 1), 
        argnums = 1),
    static_argnums = 0)

def set_anchor_gain(dyn: Dynamics, gain: float):
    anchor_idx = jnp.array([0, 1])
    k0 = dyn.mjx_model.tendon_stiffness[anchor_idx]
    c0 = dyn.mjx_model.tendon_damping[anchor_idx]
    k = k0 * gain
    c = c0 * jnp.sqrt(gain)

    ## setting the properties of the anchor tendon
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[anchor_idx].set(k),
        tendon_damping = dyn.mjx_model.tendon_damping.at[anchor_idx].set(c)
    )

    return dyn

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    seed = 10
    key = jax.random.key(seed)
    steps = 3000  ## simulation horizon
    horizon = 70
    power = jnp.array([1.80, 1.02])
    alpha = 0.01
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_cart_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(dyn.state_dim, dyn.control_dim)
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    anchor_gain = 0.00001
    dyn = set_anchor_gain(dyn, anchor_gain)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))

    ## initial state of pendula (all zeros)
    xt = dyn.init_state()

    
    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    
    assert steps == empowerment.shape[0]
    
    for t in range(steps):
        key, sub_key = jax.random.split(key)

        ## measure empowerment and its gradient
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power * horizon, alpha, observation_noise)
        i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power * horizon, alpha, observation_noise)

        # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, observation_noise)
        # i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power, alpha, observation_noise)


        ## log the number of IWF iterations and empowerment
        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)

        if t == 0:
            ## choose a random action on the first step
            random_signs = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
            ut = power * random_signs
        else:
            ## choose an empowerment maximizing action
            _, B = dyn.linearize(xt, U[0])
            ut = compute_multiagent_control(grad_E, B, power, key)

        ## step the dynamics and record the result
        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        ## print out some relevant quantities
        print(t, xt, ut, e, i)

    run_name = f'seed={seed}_anchor-gain={anchor_gain}_horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    dyn.render(X, path = run_name + '.mp4', skip = 3)

    fig, ax = plt.subplots(3, 1)
    ## plot empowerment
    ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    ax[0].set_ylabel('Empowerment', fontsize = 14)
    ax[0].tick_params(axis = 'both', labelsize = 12)
    ax[0].legend(fontsize = 12)
    ax[0].set_xticks([])
    ax[0].set_xlim(0, steps)

    ## plot angle from top
    agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 3] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top', fontsize = 14)
    ax[1].tick_params(axis = 'both', labelsize = 12)
    ax[1].set_xlim(0, steps)
    ax[1].set_xlabel('Interaction Time (s)', fontsize = 14)

    n_ticks = 5
    positions = jnp.linspace(0, steps - 1, n_ticks)
    labels = jnp.linspace(0.0, steps * dt, n_ticks)

    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(labels, rotation = 'horizontal')
    
    ax[2].plot(iterations)
    ax[2].set_xlim(0, steps)

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()