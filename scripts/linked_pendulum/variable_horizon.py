import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix, batch_diag, iterative_waterfilling, select_output
from soc_emp.utils import smooth_angle_wrap

def pad_to(arr: Array, target_len: int, axis: int = -1):
    pad_shape = list(arr.shape)
    pad_shape[axis] = target_len
    out = jnp.zeros(pad_shape, dtype=arr.dtype)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, arr.shape[axis])
    return out.at[tuple(slices)].set(arr)

# def build_linked_pendulum_channel_matrix(A: Array, B: Array, horizon: tuple[int, int]):
def build_linked_pendulum_channel_matrix(A: Array, B: Array, horizon: Array):

    ## how much to pad
    pad_0 = A.shape[0] - horizon[0]
    pad_1 = A.shape[0] - horizon[1]

    ## compute the sensitivity of the final state in the horizon to each action
    F_0 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, horizon[0]),
        jax.lax.dynamic_slice_in_dim(B, 0, horizon[0]))
    
    F_1 = compute_F_from_A_B(
        jax.lax.dynamic_slice_in_dim(A, 0, horizon[1]),
        jax.lax.dynamic_slice_in_dim(B, 0, horizon[1]))
    
    ## swap the indices so the state dimention is first
    F_0 = jnp.permute_dims(F_0, (1, 0, 2)) ## (state x horizon[0] x control)
    F_1 = jnp.permute_dims(F_1, (1, 0, 2)) ## (state x horizon[0] x control)

    ## split the channel matrix into the effect of each agents actions on each agents state
    F_0_agent, F_0_noise = split_channel_matrix(F_0, 2)
    F_1_agent, F_1_noise = split_channel_matrix(F_1, 2)

    # ## sensitivity of agent i's action on its own state
    # F_agent = jnp.stack([
    #     jnp.pad(F_0_agent[0][[0, 2], :], ((0, 0), (0, pad_0))),
    #     jnp.pad(F_1_agent[1][[1, 3], :], ((0, 0), (0, pad_1)))
    # ])

    # F_noise = jnp.stack([
    #     jnp.pad(F_0_noise[0][:, [0, 2], :], ((0, 0), (0, 0), (0, pad_0))),
    #     jnp.pad(F_1_noise[1][:, [1, 3], :], ((0, 0), (0, 0), (0, pad_1)))
    # ])

    max_horizon = A.shape[0]  # compile-time constant
    
    ## sensitivity of agent i's action on its own state
    F_agent = jnp.stack([
        pad_to(F_0_agent[0][[0, 2], :], max_horizon),
        pad_to(F_1_agent[1][[1, 3], :], max_horizon)
    ])

    F_noise = jnp.stack([
        jnp.pad(F_0_noise[0][:, [0, 2], :], max_horizon),
        jnp.pad(F_1_noise[1][:, [1, 3], :], max_horizon)
    ])

    return F_agent, F_noise

# build_linked_pendulum_channel_matrix = jax.jit(build_linked_pendulum_channel_matrix, static_argnums = 2)
# build_linked_pendulum_channel_matrix = jax.jit(build_linked_pendulum_channel_matrix)


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

# compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = (0, 3))
# compute_multiagent_empowerment_grad = jax.jit(
#     jax.jacfwd(
#         select_output(compute_multiagent_empowerment, 1), 
#         argnums = 1),
#     static_argnums = (0, 3))

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
    max_horizon = 100
    horizon = (100, 50)
    assert max(horizon) <= max_horizon
    power = jnp.array([2.0, 2.0])
    alpha = 0.50
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((max_horizon, dyn.control_dim))
    
    xt = dyn.init_state()
    # xt = xt.at[0].set(3.1)
    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    # horizon = jnp.array([horizon[0], horizon[1]])
    print(horizon)
    # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, horizon, power, alpha, observation_noise)
    # print(grad_E)

    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)

    print(A.shape, B.shape)
    # F_agent, F_noise = build_linked_pendulum_channel_matrix(A, B, horizon)
    # print(F_agent.shape)
    
    A = jax.random.normal(key, (100, 4, 4))
    B = jax.random.normal(key, (100, 4, 2))
    build_linked_pendulum_channel_matrix(A, B, horizon)











    # iterations = jnp.zeros(steps)
    # empowerment = jnp.zeros((steps, 2))
    
    # for t in range(steps):
    #     ## obtain control gain
    #     _, B = dyn.linearize(xt, U[0])
    #     grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, horizon, power, alpha, observation_noise)
    #     ## compute action
    #     ut = jnp.sign(jnp.diag(grad_E @ B)) * power
    #     ## pick a random direction with max power if the action is zero
    #     sub_key, key = jax.random.split(key)
    #     random_direction = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
    #     ut = ut + (ut == 0) * power * random_direction

    #     i, e, _ = compute_multiagent_empowerment(dyn, xt, U, horizon, power, alpha, observation_noise)

    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)

    #     iterations = iterations.at[t].set(i)
    #     empowerment = empowerment.at[t].set(e)
    #     print(t, xt, ut, e, i)

    # run_name = f'horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    # dyn.render(X, path = run_name + '.mp4', skip = 3)

    # fig, ax = plt.subplots(2, 1)
    # # fig.suptitle(f'Horizon = {horizon * dt} (seconds)')
    # # fig.suptitle('Failure Outcome', fontsize = 14)
    # # fig.suptitle('Domination Outcome', fontsize = 14)
    # ## plot empowerment
    # ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    # ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    # ax[0].set_ylabel('Empowerment', fontsize = 14)
    # ax[0].tick_params(axis = 'both', labelsize = 12)
    # ax[0].legend(fontsize = 12)
    # ax[0].set_xticks([])
    # ax[0].set_xlim(0, 1500)

    # ## plot angle from top
    # agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    # agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    # ax[1].plot(agent_0_angle, color = 'blue')
    # ax[1].plot(agent_1_angle, color = 'orange')
    # ax[1].set_ylabel('Angle From Top', fontsize = 14)
    # ax[1].tick_params(axis = 'both', labelsize = 12)
    # ax[1].set_xlim(0, 1500)
    # ax[1].set_xlabel('Interaction Time (s)', fontsize = 14)

    # n_ticks = 5
    # positions = np.linspace(0, empowerment.shape[0] - 1, n_ticks)
    # labels = np.linspace(0.0, 15.0, n_ticks)

    # ax[1].set_xticks(positions)
    # ax[1].set_xticklabels(labels, rotation = 'horizontal')

    # fig.tight_layout()
    # fig.savefig(run_name + '.png', dpi = 300)
    # plt.show()