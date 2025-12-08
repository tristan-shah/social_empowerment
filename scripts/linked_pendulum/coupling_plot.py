import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, batch_diag
from soc_emp.utils import smooth_angle_wrap, split_state, get_state

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

def make_compute_multiagent_empowerment(
        dyn: Dynamics,
        U: Array, 
        power: Array, 
        alpha: float,
        observation_noise: float):
    

    num_agents = len(power)
    horizon = U.shape[0]
    du = dyn.control_dim // num_agents
    dm = du * horizon

    step = make_step(dyn)
    unroll = make_unroll(step, U)

    linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)), in_axes = (0, 0, None))
    
    def compute_multiagent_empowerment(x0: Array, k: Array):
        '''
        original
        '''
        S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)

        # hardcoded noise perturbation
        S_z = jnp.eye(2) * observation_noise

        X = unroll(x0, k)
        A, B = linearize(X[:-1], U, k)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))

        F_agent, F_noise = split_channel_matrix(F, num_agents)

        '''
        egoistic double pendulum. Each agent only cares about its own state (angle, angular velocity).
        '''
        F_agent = jnp.stack([
            F_agent[0, [0, 2], :],
            F_agent[1, [1, 3], :]
            ], axis = 0)

        ## chained indexing allows to select the correct submatrices
        F_noise = jnp.stack([
            F_noise[0][:, [0, 2], :],
            F_noise[1][:, [1, 3], :]
        ], axis = 0)

        i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)

        return e
    
    return jax.jit(compute_multiagent_empowerment)

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')
    ## hyperparams
    seed = 12312
    key = jax.random.key(seed)
    alpha = 0.01
    horizon = 300
    observation_noise = 1.0
    power = jnp.array([1.0, 1.0])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    U = jnp.zeros((horizon, dyn.control_dim))
    
    step = make_step(dyn)
    unroll = make_unroll(step, U)
    compute_multiagent_empowerment = make_compute_multiagent_empowerment(dyn, U, power * horizon, alpha, observation_noise)

    k = jnp.array([5.0])

    agent_1_angle = jnp.pi - 0.01
    agent_2_angle = jnp.pi - 0.01

    xt = dyn.init_state()
    xt = xt.at[0].set(agent_1_angle)
    xt = xt.at[1].set(agent_2_angle)

    e = compute_multiagent_empowerment(xt, k)
    print(e)

    # X = unroll(xt, k)
    # dyn.render(X, path = 'test.mp4')
    # print(jax.jacfwd(compute_multiagent_empowerment, argnums = 1)(xt, k))

    spring_constants = jnp.linspace(0.0, 5.0, 200)
    e = jax.vmap(compute_multiagent_empowerment, in_axes = (None, 0))(xt, spring_constants)
    print(e)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Horizon = {horizon * dt} (s), Agent 1 Angle = {round(agent_1_angle, 3)}, Agent 2 Angle = {round(agent_2_angle, 3)}')
    ax.set_xlabel('Spring Constant (k)')
    ax.set_ylabel('Empowerment')
    ax.plot(spring_constants, e[:, 0], label = 'Agent 1 Empowerment')
    ax.plot(spring_constants, e[:, 1], label = 'Agent 2 Empowerment')
    ax.legend()
    fig.tight_layout()
    fig.savefig('test.png', dpi = 300)
    plt.show()