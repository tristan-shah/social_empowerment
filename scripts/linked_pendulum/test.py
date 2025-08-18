import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad
from soc_emp.utils import smooth_angle_wrap

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(0)
    # key = jax.random.key(1)
    steps = 1500  ## simulation horizon
    # alpha = 0.01
    alpha = 0.2
    horizon = 50
    power = jnp.array([1.0, 1.0])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))
    
    xt = dyn.init_state()
    xt = xt.at[0].set(0.0)
    xt = xt.at[1].set(0.0)

    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    
    for t in range(steps):
        ## obtain control gain
        _, B = dyn.linearize(xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, key)
        print(grad_E)
        break
    #     ## compute action
    #     ut = jnp.sign(jnp.diag(grad_E @ B)) * power
    #     ## pick a random direction with max power if the action is zero
    #     sub_key, key = jax.random.split(key)
    #     random_direction = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
    #     ut = ut + (ut == 0) * power * random_direction

    #     i, e = compute_multiagent_empowerment(dyn, xt, U, power, alpha, key)

    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)

    #     iterations = iterations.at[t].set(i)
    #     empowerment = empowerment.at[t].set(e)
    #     print(t, xt, ut, e, i)

    # run_name = f'horizon={horizon}_power={power}'

    # dyn.render(X, path = run_name + '.mp4', skip = 3)

    # fig, ax = plt.subplots(3, 1)
    # fig.suptitle(f'Horizon = {horizon * dt} (seconds)')

    # ## plot empowerment
    # ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    # ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    # ax[0].set_ylabel('Empowerment', fontsize = 14)
    # ax[0].tick_params(axis = 'both', labelsize = 12)
    # ax[0].legend(fontsize = 12)

    # ## plot angle from top
    # agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    # agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    # ax[1].plot(agent_0_angle, color = 'blue')
    # ax[1].plot(agent_1_angle, color = 'orange')
    # ax[1].set_ylabel('Angle From Top', fontsize = 14)
    # ax[1].tick_params(axis = 'both', labelsize = 12)

    # ## plot IWF iterations
    # ax[2].plot(iterations)
    # ax[2].set_xlabel('Timestep', fontsize = 14)
    # ax[2].set_ylabel('Iterations', fontsize = 14)
    # ax[2].tick_params(axis = 'both', labelsize = 12)

    # fig.tight_layout()
    # fig.savefig(run_name + '.png', dpi = 300)
    # plt.show()