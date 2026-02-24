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
    seed = 123123
    key = jax.random.key(seed)
    steps = 2000
    alpha = 0.01
    horizon = 150
    observation_noise = 1.0
    stiffness = 3.0
    damping = 0.1

    power = jnp.array([0.9, 1.4])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep

    ## setting the properties of the tendon
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[:].set(stiffness),
        tendon_damping = dyn.mjx_model.tendon_damping.at[:].set(damping)
    )

    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))

    ## initial state of pendula (all zeros)
    xt = dyn.init_state()

    '''
    run controller
    '''
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

        ## log the number of IWF iterations and empowerment
        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)

        '''
        Original egoistic
        '''
        if t == 0:
            ## choose a random action on the first step
            random_signs = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
            ut = power * random_signs
        else:
            ## choose an empowerment maximizing action
            _, B = dyn.linearize(xt, U[0])
            ut = compute_multiagent_control(grad_E, B, power, key)
        # else:
        #     '''
        #     Master servant
        #     '''
        #     _, B = dyn.linearize(xt, U[0])
        #     ## choose an action that maximizes the empowerment of agent 0
        #     ## gradient of agent 0's empowerment
        #     ut = jnp.sign(grad_E[0, :] @ B) * power

        ## step the dynamics and record the result
        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        ## print out some relevant quantities
        print(t, xt, ut, e, i)

    run_name = f'left_master-seed={seed}_horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}-stiff={stiffness}-damp={damping}'

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
    agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top', fontsize = 14)
    ax[1].tick_params(axis = 'both', labelsize = 12)
    ax[1].set_xlim(0, steps)
    ax[1].set_xlabel('Interaction Time (s)', fontsize = 14)

    n_ticks = 5
    positions = np.linspace(0, steps - 1, n_ticks)
    labels = np.linspace(0.0, steps * dt, n_ticks)

    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(labels, rotation = 'horizontal')
    
    ax[2].plot(iterations)

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()