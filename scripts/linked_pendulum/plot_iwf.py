import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad
from soc_emp.utils import smooth_angle_wrap

'''
Collaborative power pairs for horizon = 50
'''
collaborative_pairs = jnp.array([
    [1.62068966, 2.82758621],
    [2.39655172, 1.10344828],
    [1.10344828, 1.79310345],
    [1.79310345, 3.        ],
    [1.96551724, 1.01724138],
    [1.44827586, 1.18965517],
    [1.44827586, 1.62068966],
    [1.44827586, 1.87931034]
    ])

'''
Dominating power pairs for horizon = 100
'''
dominating_pairs = jnp.array([
    [1.10344828, 0.75862069],
    [0.84482759, 1.44827586],
    [1.96551724, 0.75862069],
    [1.62068966, 0.5862069 ],
    [1.27586207, 0.5       ],
    [2.56896552, 1.62068966],
    [1.79310345, 0.93103448],
    [1.44827586, 0.5       ],
])

'''
Dominating power pairs for horizon = 50
'''
dominating_pairs = jnp.array([
    [1.79310345, 0.93103448],
    [1.44827586, 0.5       ],
    [2.56896552, 1.62068966],
    [1.27586207, 0.5       ],
    [1.62068966, 0.5862069 ],
    [1.96551724, 0.75862069],
    [1.10344828, 0.75862069],
    [2.39655172, 0.93103448]
])

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(4)
    steps = 1500  ## simulation horizon
    alpha = 0.01
    horizon = 50

    pair_idx = 7
    # power = collaborative_pairs[pair_idx]
    power = dominating_pairs[pair_idx]
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))
    
    xt = dyn.init_state()

    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    frobenious_norm = jnp.zeros(steps)
    
    for t in range(steps):
        ## obtain control gain
        _, B = dyn.linearize(xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, observation_noise)
        ## compute action
        ut = jnp.sign(jnp.diag(grad_E @ B)) * power
        ## pick a random direction with max power if the action is zero
        sub_key, key = jax.random.split(key)
        random_direction = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
        ut = ut + (ut == 0) * power * random_direction

        i, e, S = compute_multiagent_empowerment(dyn, xt, U, power, alpha, observation_noise)

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)

        frobenious_norm = frobenious_norm.at[t].set(
            jnp.sqrt(jnp.sum((S[0] - S[1]) ** 2))
        )

        print(t, xt, ut, e, i, frobenious_norm[t])

    ## generate plots
    run_name = f'horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    # dyn.render(X, path = run_name + '.mp4', skip = 3)
    # jnp.save(f'colab_alignment_{pair_idx}', frobenious_norm)
    jnp.save(f'domin_alignment_{pair_idx}', frobenious_norm)


    fig, ax = plt.subplots(1, 1)
    ## alignement of covariance matrices
    ax.plot(frobenious_norm)
    ax.set_xlabel('Timestep', fontsize = 14)
    ax.set_ylabel('Strategy Alignment', fontsize = 14)
    ax.tick_params(axis = 'both', labelsize = 12)
    plt.show()





    # fig, ax = plt.subplots(5, 1, figsize = (10, 8))
    # fig.suptitle(f'Horizon = {horizon * dt} (seconds), Left Power = {power[0]}, Right Power = {power[1]}, Alpha = {alpha}, Noise = {observation_noise}')

    # ## plot empowerment
    # ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    # ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    # ax[0].set_xlabel('Timestep', fontsize = 14)
    # ax[0].set_ylabel('Empowerment', fontsize = 14)
    # ax[0].tick_params(axis = 'both', labelsize = 12)
    # ax[0].legend(fontsize = 8)

    # ## plot angle from top
    # agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    # agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    # ax[1].plot(agent_0_angle, color = 'blue')
    # ax[1].plot(agent_1_angle, color = 'orange')
    # ax[1].set_xlabel('Timestep', fontsize = 14)
    # ax[1].set_ylabel('Angle From Top', fontsize = 14)
    # ax[1].tick_params(axis = 'both', labelsize = 12)

    # ## plot IWF iterations
    # ax[2].plot(iterations)
    # ax[2].set_xlabel('Timestep', fontsize = 14)
    # ax[2].set_ylabel('Iterations', fontsize = 14)
    # ax[2].tick_params(axis = 'both', labelsize = 12)

    # ## alignement of covariance matrices
    # ax[3].plot(frobenious_norm)
    # ax[3].set_xlabel('Timestep', fontsize = 14)
    # ax[3].set_ylabel('Frobenious Norm', fontsize = 14)
    # ax[3].tick_params(axis = 'both', labelsize = 12)

    # ## power allocation over time
    # ax[4].plot(jnp.diag(S[0]), label = 'Left Agent', color = 'blue')
    # ax[4].plot(jnp.diag(S[1]), label = 'Right Agent', color = 'orange')
    # ax[4].set_xlabel('Horizon Timestep', fontsize = 14)
    # ax[4].set_ylabel('Variance', fontsize = 14)
    # ax[4].tick_params(axis = 'both', labelsize = 12)

    # fig.tight_layout()
    # fig.savefig(run_name + '.png', dpi = 300)
    # plt.show()