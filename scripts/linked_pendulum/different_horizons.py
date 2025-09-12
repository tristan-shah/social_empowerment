import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, waterfilling_operator, select_output
from soc_emp.utils import smooth_angle_wrap

MAX_ITER = 10

def linked_pendulum_empowerment(dyn: Dynamics, x0: Array, U: Array, horizon: tuple, power: Array, alpha: float):

    num_agents = 2
    # pad = horizon.max() - horizon
    max_h = max(horizon)
    pad = [max_h - h for h in horizon]

    ## perform simulation
    X = unroll(dyn, x0, U)
    ## compute jacobians
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    ## agent 0 adresses
    agent_0_state_adr = jnp.array([0, 2])
    agent_0_horizon = horizon[0]
    agent_0_pad = pad[0]

    ## agent 1 adresses
    agent_1_state_adr = jnp.array([1, 3])
    agent_1_horizon = horizon[1]
    agent_1_pad = pad[1]

    ## compute agent 0 F
    # F_0 = compute_F_from_A_B(A[:agent_0_horizon], B[:agent_0_horizon])
    F_0 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, agent_0_horizon),
        jax.lax.dynamic_slice_in_dim(B, 0, agent_0_horizon)
    )
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon_0 x control)
    F_0 = jnp.pad(F_0, ((0, 0), (0, agent_0_pad), (0, 0)))

    ## compute agent 1 F
    # F_1 = compute_F_from_A_B(A[:agent_1_horizon], B[:agent_1_horizon])
    F_1 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, agent_1_horizon),
        jax.lax.dynamic_slice_in_dim(B, 0, agent_1_horizon)
    )
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

    S = jnp.zeros((num_agents, max_h, max_h))
    ## noise covariance
    ## without adding a small perturbation the gradient of empowerment is nan
    # S_z = jnp.eye(2) + jnp.diag(jax.random.normal(key, (2))) * 1e-5
    S_z = jnp.diag(jax.random.uniform(key, (2))) * 1e-5

    '''
    Explicit iteration
    '''
    max_iter = MAX_ITER

    ## state is defined as (iteration, covariance matrices, current empowerment, previous empowerment)
    def cond_fun(state):
        i, S, e, e_prev = state
        return jnp.logical_and(
            jnp.any(jnp.abs(e - e_prev) > 1e-5),
            i < max_iter)

    def body_fun(state):
        i, S, e, e_prev = state
        e_prev = e
        e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_
        return (i + 1, S, e, e_prev)
    
    e_prev = jnp.ones(num_agents) * jnp.inf
    e = jnp.zeros(num_agents)
    i, S, e, e_prev = jax.lax.while_loop(cond_fun, body_fun, (0, S, e, e_prev))

    return i, e

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    # key = jax.random.key(0) ## if right has the power advantage
    # key = jax.random.key(5) ## if left has the power advantage
    key = jax.random.key(42)
    alpha = 0.01
    steps = 1500

    ## how much the agent can push at each timestep
    # power_density = jnp.array([1.0, 1.1])
    power_density = jnp.array([1.5, 1.3])
    ## horizon length for each agent
    horizon = (100, 100)
    ## total power allocation for IWF
    power = power_density * jnp.array(horizon)

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {max(horizon)}')

    dx = dyn.state_dim
    du = dyn.control_dim

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((max(horizon), dyn.control_dim))

    xt = dyn.init_state()

    linked_pendulum_empowerment = jax.jit(linked_pendulum_empowerment, static_argnums = (0, 3))
    linked_pendulum_empowerment_grad = jax.jit(
        jax.jacfwd(
            select_output(linked_pendulum_empowerment, 1),
            argnums = 1), 
        static_argnums = (0, 3))


    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    
    for t in range(steps):
        ## obtain control gain
        _, B = dyn.linearize(xt, U[0])
        grad_E = linked_pendulum_empowerment_grad(dyn, xt, U, horizon, power, alpha)
        ## compute action
        ut = jnp.sign(jnp.diag(grad_E @ B)) * power_density
        ## pick a random direction with max power if the action is zero
        sub_key, key = jax.random.split(key)
        random_direction = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
        ut = ut + (ut == 0) * power_density * random_direction

        i, e = linked_pendulum_empowerment(dyn, xt, U, horizon, power, alpha)

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)
        print(t, xt, ut, e, i)

    run_name = f'horizon={horizon}_power_density={power_density}'

    dyn.render(X, path = run_name + '.mp4', skip = 3)

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(f'Left Horizon = {horizon[0] * dt} (seconds), Right Horizon = {horizon[1] * dt} (seconds)')

    ## plot empowerment
    ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    ax[0].set_ylabel('Empowerment', fontsize = 14)
    ax[0].tick_params(axis = 'both', labelsize = 12)
    ax[0].legend(fontsize = 12)

    ## plot angle from top
    agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    ax[1].plot(agent_0_angle, color = 'blue')
    ax[1].plot(agent_1_angle, color = 'orange')
    ax[1].set_ylabel('Angle From Top', fontsize = 14)
    ax[1].tick_params(axis = 'both', labelsize = 12)

    ## plot IWF iterations
    ax[2].plot(iterations)
    ax[2].set_xlabel('Timestep', fontsize = 14)
    ax[2].set_ylabel('Iterations', fontsize = 14)
    ax[2].tick_params(axis = 'both', labelsize = 12)

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()