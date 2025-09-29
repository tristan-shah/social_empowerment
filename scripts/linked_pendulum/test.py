import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad, compute_multiagent_control
from soc_emp.utils import smooth_angle_wrap

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    # seed = 4
    seed = 8
    key = jax.random.key(seed)
    steps = 1500  ## simulation horizon
    alpha = 0.01
    horizon = 120
    observation_noise = 1.0

    # power = jnp.array([1.88, 2.20])
    power = jnp.array([2.20, 1.88])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))
    
    xt = dyn.init_state()

    # '''
    # START: plot iwf
    # '''
    # from soc_emp.empowerment import batch_diag, unroll, compute_F_from_A_B, split_channel_matrix, waterfilling_operator
    # num_agents = len(power)
    # horizon = U.shape[0]
    # du = dyn.control_dim // num_agents
    # dm = du * horizon

    # # S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)
    # S = batch_diag(jnp.ones((num_agents, dm)))
    # # hardcoded noise perturbation
    # S_z = jnp.eye(2) * observation_noise

    # X = unroll(dyn, xt, U)
    # A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    # F = compute_F_from_A_B(A, B)
    # F = jnp.permute_dims(F, (1, 0, 2))

    # F_agent, F_noise = split_channel_matrix(F, num_agents)

    # '''
    # egoistic double pendulum. Each agent only cares about its own state (angle, angular velocity).
    # '''
    # F_agent = jnp.stack([
    #     F_agent[0, [0, 2], :],
    #     F_agent[1, [1, 3], :]
    #     ], axis = 0)

    # ## chained indexing allows to select the correct submatrices
    # F_noise = jnp.stack([
    #     F_noise[0][:, [0, 2], :],
    #     F_noise[1][:, [1, 3], :]
    # ], axis = 0)

    # print(F_agent.shape, F_noise.shape)

    # iterations = 10
    # empowerment = jnp.zeros((iterations, num_agents))

    # for i in range(iterations):
    #     e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
    #     S = alpha * S + (1 - alpha) * S_
    #     print(e)
    #     empowerment = empowerment.at[i].set(e)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(empowerment[:, 0])
    # ax.plot(empowerment[:, 1])
    # plt.show()
    # '''
    # END: plot actual iwf
    # '''

    # delta = jax.random.normal(key, power.shape) * 0.01
    # print(delta)
    # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, observation_noise)
    # print(grad_E)
    # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power + delta, alpha, observation_noise)
    # print(grad_E)

    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    
    for t in range(steps):
        ## obtain control gain
        _, B = dyn.linearize(xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, observation_noise)
        print(grad_E)
        exit()

        key, sub_key = jax.random.split(key)
        ut = compute_multiagent_control(grad_E, B, power, sub_key)

        i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power, alpha, observation_noise)

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)
        print(t, xt, ut, e, i)

    run_name = f'seed={seed}_horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    dyn.render(X, path = run_name + '.mp4', skip = 3)


    fig, ax = plt.subplots(3, 1)
    ## plot empowerment
    ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    ax[0].set_ylabel('Empowerment', fontsize = 14)
    ax[0].tick_params(axis = 'both', labelsize = 12)
    ax[0].legend(fontsize = 12)
    ax[0].set_xticks([])
    ax[0].set_xlim(0, 1500)

    ## plot angle from top
    agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top', fontsize = 14)
    ax[1].tick_params(axis = 'both', labelsize = 12)
    ax[1].set_xlim(0, 1500)
    ax[1].set_xlabel('Interaction Time (s)', fontsize = 14)

    n_ticks = 5
    positions = np.linspace(0, empowerment.shape[0] - 1, n_ticks)
    labels = np.linspace(0.0, 15.0, n_ticks)

    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(labels, rotation = 'horizontal')
    
    ax[2].plot(iterations)

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()