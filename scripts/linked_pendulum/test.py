## for server rendering
import os
os.environ['MUJOCO_GL'] = 'egl'

## the rest of the imports
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment
from soc_emp.utils import smooth_angle_wrap


if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(5)
    steps = 1500  ## simulation horizon
    num_agents = 2
    alpha = 0.01
    horizon = 50  ## single horizon

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path=xml_path)
    print(f'Timestep = {dyn.mjx_model.opt.timestep}')
    print(f'Horizon = {horizon}')

    U = jnp.zeros((horizon, dyn.control_dim))

    batch = 4
    outcomes = jnp.load(f'results_random_init/horizon={horizon}/horizon={horizon}_outcomes_batch_{batch}.npy')
    pairs = jnp.load(f'results_random_init/horizon={horizon}/horizon={horizon}_pairs_batch_{batch}.npy')
    X = jnp.load(f'results_random_init/horizon={horizon}/horizon={horizon}_X_batch_{batch}.npy')

    idx = jnp.where(outcomes == 3)[0]

    if len(idx) == 0:
        raise ValueError('Outcome not found.')
    i = idx[2]

    outcome = outcomes[idx[i]]
    power = pairs[idx[i]]
    traj = X[idx[i]]

    print(idx)
    print(outcome)
    print(power)
    print(traj.shape)

    batch_compute_multiagent_empowerment = jax.vmap(lambda _xt: compute_multiagent_empowerment(dyn, _xt, U, power, alpha = 0.01, key = key))
    iterations, empowerment = batch_compute_multiagent_empowerment(traj)

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(f'Horizon = {horizon}')

    ## plot empowerment
    ax[0].plot(empowerment[:, 0], label='Left Agent', color='blue')
    ax[0].plot(empowerment[:, 1], label='Right Agent', color='orange')
    ax[0].set_ylabel('Empowerment', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].legend(fontsize=12)

    ## plot angle from top
    agent_0_angle = jnp.abs(smooth_angle_wrap(traj[:, 0] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(traj[:, 1] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top', fontsize = 14)
    ax[1].tick_params(axis='both', labelsize = 12)

    ## plot IWF iterations
    ax[2].plot(iterations)
    ax[2].set_xlabel('Timestep', fontsize = 14)
    ax[2].set_ylabel('Iterations', fontsize = 14)
    ax[2].tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    fig.savefig(f'test.png', dpi = 300)
    plt.show()