import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp.utils import smooth_angle_wrap
from soc_emp import Dynamics
from soc_emp.dynamics import make_step

from sweep_power import make_run, set_tendon_properties, make_compute_group_empowerment, build_linked_pendulum_state_matrix

if __name__ == '__main__':
    stiffness = 3.0
    damping = 0.1
    dt = 0.01
    ## hyperparams
    steps = 2000
    state_type = 'angle'
    control_type = 'ave'
    horizon = 150
    alpha = 0.01
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    dyn = set_tendon_properties(dyn, stiffness, damping)

    run = make_run(dyn, steps, state_type, control_type, horizon, alpha, observation_noise)

    step = make_step(dyn)
    state_matrix = build_linked_pendulum_state_matrix(state_type)
    U = jnp.zeros((horizon, dyn.control_dim))

    compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, alpha, observation_noise)

    power_density = jnp.array([0.1, 3.8])

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    X = run(power_density, subkey)



    run_name = f'control_type={control_type}-state_type={state_type}-horizon={horizon}_power={power_density}_alpha={alpha}_noise={observation_noise}-stiff={stiffness}-damp={damping}'
    empowerment = jax.vmap(compute_group_empowerment, in_axes = (0, None))(X, power_density)


    fig, ax = plt.subplots(2, 1)
    ## plot empowerment
    ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    ax[0].set_ylabel('Empowerment')
    ax[0].tick_params(axis = 'both')
    ax[0].legend()
    ax[0].set_xticks([])
    ax[0].set_xlim(0, steps)

    ## plot angle from top
    agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top')
    ax[1].tick_params(axis = 'both')
    ax[1].set_xlim(0, steps)

    n_ticks = 5
    positions = jnp.linspace(0, steps - 1, n_ticks)
    labels = jnp.linspace(0.0, steps * dt, n_ticks)
    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(labels, rotation = 'horizontal')

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()


    dyn.render(X, path = run_name + '.mp4', skip = 3)