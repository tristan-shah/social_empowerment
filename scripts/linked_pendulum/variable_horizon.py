import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix, batch_diag, iterative_waterfilling, select_output, waterfilling_operator, compute_multiagent_control
from soc_emp.utils import smooth_angle_wrap

@jax.jit
def build_linked_pendulum_channel_matrix(A: Array, B: Array, horizon: Array):
    T = A.shape[0]
    ar = jnp.arange(T)
    eye_tiled = jnp.tile(jnp.eye(4)[None, :, :], (T, 1, 1))

    mask_0 = ar < horizon[0]
    A_0 = jnp.where(mask_0[:, None, None], A, eye_tiled)
    B_0 = jnp.where(mask_0[:, None, None], B, 0.0)

    mask_1 = ar < horizon[1]
    A_1 = jnp.where(mask_1[:, None, None], A, eye_tiled)
    B_1 = jnp.where(mask_1[:, None, None], B, 0.0)

    F_0 = compute_F_from_A_B(A_0, B_0)
    F_1 = compute_F_from_A_B(A_1, B_1)
    
    ## swap the indices so the state dimension is first
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon[0] x control)
    F_1 = jnp.permute_dims(F_1, (1, 0, 2)) ## (state x horizon[0] x control)

    ## split the channel matrix into the effect of each agents actions on each agents state
    F_0_agent, F_0_noise = split_channel_matrix(F_0, 2)
    F_1_agent, F_1_noise = split_channel_matrix(F_1, 2)

    ## sensitivity of agent i's action on its own state
    F_agent = jnp.stack([
        F_0_agent[0][[0, 2], :],
        F_1_agent[1][[1, 3], :]
    ])

    F_noise = jnp.stack([
        F_0_noise[0][:, [0, 2], :],
        F_1_noise[1][:, [1, 3], :]
    ])

    return F_agent, F_noise

def compute_multiagent_empowerment(
        dyn: Dynamics, 
        x0: Array, 
        U: Array,
        horizon: Array,
        power: Array,
        alpha: float,
        observation_noise: float):
    
    num_agents = 2
    max_horizon = U.shape[0]
    
    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    F_agent, F_noise = build_linked_pendulum_channel_matrix(A, B, horizon)

    ## initial agent covariance matrices
    S = batch_diag(jnp.ones((num_agents, max_horizon)))
    ## noise covariance
    S_z = jnp.eye(2) * observation_noise

    i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)

    return i, e, S

compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = 0)
compute_multiagent_empowerment_grad = jax.jit(
    jax.jacfwd(
        select_output(compute_multiagent_empowerment, 1), 
        argnums = 1),
    static_argnums = 0)

if __name__ == '__main__':

    ## system hyperparameters
    key = jax.random.key(4)
    steps = 1500  ## interaction horizon
    horizon = jnp.array([95, 100])
    power = jnp.array([1.5, 1.3])
    alpha = 0.01
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((max(horizon), dyn.control_dim))
    
    xt = dyn.init_state()
    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 2))
    
    for t in range(steps):
        ## obtain control gain
        _, B = dyn.linearize(xt, U[0])
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, horizon, power, alpha, observation_noise)

        sub_key, key = jax.random.split(key)
        ut = compute_multiagent_control(grad_E, B, power, sub_key)

        ## compute empowerment for plotting
        i, e, _ = compute_multiagent_empowerment(dyn, xt, U, horizon, power, alpha, observation_noise)

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)
        print(t, xt, ut, e, i)

    run_name = f'horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    dyn.render(X, path = run_name + '.mp4', skip = 3)

    fig, ax = plt.subplots(2, 1)
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

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()