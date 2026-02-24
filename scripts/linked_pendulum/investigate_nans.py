import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import numpy as np
import matplotlib.pyplot as plt
from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad, compute_multiagent_control, batch_diag, compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, waterfilling_operator
from soc_emp.utils import smooth_angle_wrap, split_state, get_state

def make_step(dyn: Dynamics):

    model = dyn.mjx_model
    data_template = mjx.make_data(model)
    nq = dyn.nq

    def step(xt: Array, ut: Array):
        qpos, qvel = split_state(xt, nq)
        data = data_template.replace(qpos = qpos, qvel = qvel, ctrl = ut)
        data = mjx.step(model, data)
        return get_state(data)
    
    return jax.jit(step)

def make_unroll(step: callable):

    def unroll(x0: Array, U: Array):

        def body_fn(x: Array, u: Array):
            x_next = step(x, u)
            return x_next, x_next
        
        _, X = jax.lax.scan(body_fn, x0, U)
        X = jnp.concatenate([x0[None, :], X], axis = 0)

        return X
    
    return jax.jit(unroll)

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')
    ## hyperparams
    seed = 12312
    key = jax.random.key(seed)
    steps = 2000
    alpha = 0.01
    horizon = 5
    observation_noise = 1.0
    stiffness = 0.0 #3.0
    damping = 0.0 #0.1

    power = jnp.array([0.4, 10.0])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep

    ## setting the properties of the tendon
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[:].set(stiffness),
        tendon_damping = dyn.mjx_model.tendon_damping.at[:].set(damping))

    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    step = make_step(dyn)
    unroll = make_unroll(step)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))

    num_agents = len(power)
    du = dyn.control_dim // num_agents
    dm = du * horizon

    def calculate(xt: Array):
    # def calculate(xt_0: Array, xt_1: Array):

        # xt = jnp.array([xt_0[0], xt_1[0], xt_0[1], xt_1[1]])
        '''
        original
        '''
        S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)

        # hardcoded noise perturbation
        S_z = jnp.eye(2) * observation_noise

        ## forward simulation
        X = unroll(xt, U)
        A, B = jax.vmap(dyn.linearize)(X[:-1], U)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))

        F_agent, F_noise = split_channel_matrix(F, num_agents)

        AGENT_0_STATE = [0, 2]
        AGENT_1_STATE = [1, 3]

        '''
        egoistic double pendulum. Each agent only cares about its own state (angle, angular velocity).
        '''
        F_agent = jnp.stack([
            F_agent[0, AGENT_0_STATE, :],
            F_agent[1, AGENT_1_STATE, :]
            ], axis = 0)

        ## chained indexing allows to select the correct submatrices
        F_noise = jnp.stack([
            F_noise[0][:, AGENT_0_STATE, :],
            F_noise[1][:, AGENT_1_STATE, :]
        ], axis = 0)

        # i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)
        # e, S = waterfilling_operator(F_agent, F_noise, S, S_z, power)
        
        S_noise = einsum(F_noise, S, F_noise, 'a1 a2 x1 m1, a2 m1 m2, a1 a2 x2 m2 -> a1 x1 x2') + S_z
        ## eigen-decomp on noise
        # D, Q = jnp.linalg.eigh(S_noise)
        D, Q = jax.lax.stop_gradient(jnp.linalg.eigh(S_noise))

        return Q

    
    ## initial state of pendula (all zeros)
    xt = dyn.init_state()
    xt = xt.at[0].set(3.1)


    print(
        calculate(xt)
    )

    # print(
    #     jax.jacfwd(calculate)(xt)
    # )


    # S_z = jnp.eye(2)
    # D, Q = jax.jacfwd(jnp.linalg.eigh)(S_z)
    # print(Q)