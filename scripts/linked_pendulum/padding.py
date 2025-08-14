import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, waterfilling_operator

def linked_pendulum_empowerment(dyn: Dynamics, x0: Array, U: Array, horizon: Array, power: Array, alpha: float):

    pad = horizon.max() - horizon

    ## perform simulation
    X = unroll(dyn, x0, U)
    ## compute jacobians
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    ## agent 0 adresses
    agent_0_state_adr = jnp.array([0, 2])
    # agent_0_ctrl_adr = jnp.array([0])
    agent_0_horizon = horizon[0]
    agent_0_pad = pad[0]

    ## agent 1 adresses
    agent_1_state_adr = jnp.array([1, 3])
    # agent_1_ctrl_adr = jnp.array([1])
    agent_1_horizon = horizon[1]
    agent_1_pad = pad[1]

    ## compute agent 0 F
    F_0 = compute_F_from_A_B(A[:agent_0_horizon], B[:agent_0_horizon])
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon_0 x control)
    F_0 = jnp.pad(F_0, ((0, 0), (0, agent_0_pad), (0, 0)))

    ## compute agent 1 F
    F_1 = compute_F_from_A_B(A[:agent_1_horizon], B[:agent_1_horizon])
    F_1 = jnp.permute_dims(F_1, (1, 0, 2)) ## (state x horizon_0 x control)
    F_1 = jnp.pad(F_1, ((0, 0), (0, agent_1_pad), (0, 0)))

    F_0_agent = F_0[agent_0_state_adr, :, 0] ## effect of agent 0 on agent 0 state
    F_0_noise = F_0[agent_0_state_adr, :, 1] ## effect of agent 1 on agent 0 state

    F_1_agent = F_1[agent_1_state_adr, :, 1] ## effect of agent 1 on agent 1 state
    F_1_noise = F_1[agent_1_state_adr, :, 0] ## effect of agent 0 on agent 1 state

    ## diagonal elements of the big sensitivity matrix
    F_agent = jnp.stack([
        F_0_agent,
        F_1_agent
    ])

    ## off diagonal elements of the big sensitivity matrix
    F_noise = jnp.stack([
        jnp.stack([0.0 * F_0_agent, F_0_noise]),
        jnp.stack([F_1_noise, 0.0 * F_1_agent])
    ])

    print(F_agent.shape)
    print(F_noise.shape)

    S = jnp.zeros((2, horizon.max(), horizon.max()))
    ## noise covariance
    # S_z = jnp.eye(2) + jnp.diag(jnp.ones(2)) * 1e-5

    observation_noise = 1e-5
    # observation_noise = 1.0
    S_z = jnp.eye(2) * observation_noise

    fig, ax = plt.subplots(1, 2)

    fig.suptitle(f'Control Variance From IWF, Observation Noise = {observation_noise}')

    ax[0].set_title('Left Agent')
    ax[0].set_xlabel('Timestep')
    ax[0].set_ylabel(r'$\mathrm{Var}(u_t)$')

    ax[1].set_title('Right Agent')
    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel(r'$\mathrm{Var}(u_t)$')

    e, S = waterfilling_operator(F_agent, F_noise, S, S_z, power)
    ax[0].plot(jnp.diag(S[0])[:agent_0_horizon], label = '1')
    ax[1].plot(jnp.diag(S[1])[:agent_1_horizon])

    e, S = waterfilling_operator(F_agent, F_noise, S, S_z, power)
    ax[0].plot(jnp.diag(S[0])[:agent_0_horizon], label = '2')
    ax[1].plot(jnp.diag(S[1])[:agent_1_horizon])

    e, S = waterfilling_operator(F_agent, F_noise, S, S_z, power)
    ax[0].plot(jnp.diag(S[0])[:agent_0_horizon], label = '3')
    ax[1].plot(jnp.diag(S[1])[:agent_1_horizon])

    fig.tight_layout()
    fig.legend(title = 'Iteration')
    fig.savefig(f'horizon={horizon}_observation_noise={observation_noise}.png', dpi = 300)
    plt.show()
    return

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(0)
    # alpha = 0.01
    alpha = 0.1

    ## how much the agent can push at each timestep
    power_density = jnp.array([1.0, 1.0])
    ## horizon length for each agent
    horizon = jnp.array([150, 150])
    ## total power allocation for IWF
    power = power_density * horizon

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon.max()}')

    dx = dyn.state_dim
    du = dyn.control_dim

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon.max(), dyn.control_dim))

    x0 = dyn.init_state()
    x0 = x0.at[0].set(-2.0)
    x0 = x0.at[1].set(1.1)

    linked_pendulum_empowerment(dyn, x0, U, horizon, power, alpha)