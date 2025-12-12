import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

from jax import Array
from jax import numpy as jnp

import matplotlib.pyplot as plt

from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, batch_diag
from soc_emp.utils import split_state, get_state

def make_step(dyn: Dynamics):

    model_template = dyn.mjx_model
    data_template = mjx.make_data(model_template)
    nq = dyn.nq

    def step(xt: Array, ut: Array, k: Array):

        qpos, qvel = split_state(xt, nq)
        data = data_template.replace(qpos = qpos, qvel = qvel, ctrl = ut)
        model = model_template.replace(tendon_stiffness = k)

        data = mjx.step(model, data)
        return get_state(data)
    
    return jax.jit(step)

def make_unroll(step: callable, U: Array):
        
    def unroll(xt: Array, k: float):
        '''
        Jax compatable simulation loop.
        '''

        def body_fun(xt_: Array, ut_: Array):
            xt_next = step(xt_, ut_, k)
            return xt_next, xt_next
        
        _, X = jax.lax.scan(body_fun, xt, U)
        return jnp.concatenate([xt[None, :], X])
        
    return jax.jit(unroll)

@jax.jit
def truncate(A: Array, B: Array, h: int):

    H = A.shape[0]
    ar = jnp.arange(H)
    mask = ar < h
    eye = jnp.eye(A.shape[1])
    eye_tiled = jnp.tile(eye[None, :, :], (H, 1, 1))

    A = jnp.where(mask[:, None, None], A, eye_tiled)
    B = jnp.where(mask[:, None, None], B, 0.0)

    return A, B

def make_compute_empowerment(step: callable, U: Array, power: Array, alpha: float, observation_noise: float):

    num_agents = len(power)
    S = batch_diag(power[:, None] * jnp.ones((num_agents, max_horizon)))
    S_z = jnp.eye(2) * observation_noise

    unroll = make_unroll(step, U)
    linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)), in_axes = (0, 0, None))

    def compute_empowerment(xt: Array, h: int, k: Array):

        ## unroll once
        X = unroll(xt, k)
        A, B = linearize(X[:-1], U, k)

        A, B = truncate(A, B, h)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))
        
        # 2. Split channels
        F_agent, F_noise = split_channel_matrix(F, 2)

        # 3. Egoistic pendulum reshaping
        F_agent = jnp.stack([
            F_agent[0, [0, 2], :],
            F_agent[1, [1, 3], :]
        ], axis = 0)

        F_noise = jnp.stack([
            F_noise[0][:, [0, 2], :],
            F_noise[1][:, [1, 3], :]
        ], axis = 0)

        # 5. Water-filling
        i, e, S_out = iterative_waterfilling(F_agent, F_noise, S, S_z, power * h, alpha)
        return e
    
    # return jax.jit(compute_empowerment)
    return compute_empowerment

@jax.jit
def step(xt: Array, ut: Array, k: Array):
    g = 9.81
    dt = 0.01

    xt_dot = jnp.stack([
        xt[2],
        xt[3],
        -g * jnp.sin(xt[0]) + ut[0] + k.squeeze() * (xt[1] - xt[0]),
        -g * jnp.sin(xt[1]) + ut[1] + k.squeeze() * (xt[0] - xt[1])
    ])

    return xt + xt_dot * dt

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')
    ## hyperparams
    seed = 0
    key = jax.random.key(seed)
    alpha = 0.01
    max_horizon = 300
    observation_noise = 1.0
    power = jnp.array([1.0, 1.0])
    k = jnp.array([0.0])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    U = jnp.zeros((max_horizon, dyn.control_dim))

    # ## tilted away from one another
    # agent_1_angle = jnp.pi - 0.1
    # agent_2_angle = jnp.pi - 0.1

    ## exactly at top
    agent_1_angle = jnp.pi
    agent_2_angle = jnp.pi

    xt = dyn.init_state()
    xt = xt.at[0].set(agent_1_angle)
    xt = xt.at[1].set(agent_2_angle)

    step = make_step(dyn)
    compute_empowerment = make_compute_empowerment(step, U, power, alpha, observation_noise)
    de_dk = jax.jacfwd(compute_empowerment, argnums = 2)
    # print(de_dk(xt, 10, k))
    # print(compute_empowerment(xt, 10, k))

    # ## compute empowerment for each horizon
    horizons = jnp.arange(50, max_horizon + 1)
    # e = jax.vmap(compute_empowerment, in_axes = (None, 0, None))(xt, horizons, k)
    de_dk = jax.vmap(jax.jacfwd(compute_empowerment, argnums = 2), in_axes = (None, 0, None))(xt, horizons, k)
    print(de_dk)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Horizon (s)')
    ax.set_ylabel('Gradient of Empowerment w.r.t Zero Spring Constant')
    ax.plot(horizons * dt, de_dk[:, 0], label = 'Agent 1')
    ax.plot(horizons * dt, de_dk[:, 1], label = 'Agent 2')
    ax.legend()
    fig.tight_layout()
    fig.savefig('de_dk.png', dpi = 300)
    plt.show()