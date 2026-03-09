import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.dynamics import make_step, make_unroll
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix

def set_tendon_properties(dyn: Dynamics, stiffness: float, damping: float):
    '''
    Function for overriding the default tendon properties in the linked pendulum scenario.
    The tendon properties control the coupling between the agents
    '''

    ## setting the properties of the tendon
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[:].set(stiffness),
        tendon_damping = dyn.mjx_model.tendon_damping.at[:].set(damping)
    )

    return dyn

def make_compute_group_empowerment(step: callable, state_matrix: Array, U: Array, power_density: Array, alpha: float, observation_noise: float):

    ## build helper functions
    linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    ## extract relevant shapes
    horizon = U.shape[0]
    power = horizon * power_density ## total probing power depends on horizon
    num_agents = len(power_density)
    total_control_dim = U.shape[1] ## total dimention of control
    agent_control_dim = total_control_dim // num_agents
    message_dim = horizon * agent_control_dim ## this is the length of the message from each agent

    ## this is the initial covariance matrix for each agent.
    S = jax.vmap(jnp.diag)(power[:, None] * jnp.ones((num_agents, message_dim)) / message_dim)
    S_z = jnp.eye(state_matrix.shape[1]) * observation_noise ## this is the observation covariance matrix for each agent

    def compute_group_empowerment(xt: Array):
        X = unroll(xt, U)
        A, B = linearize(X[:-1], U)
        F = compute_F_from_A_B(A, B)
        F = jnp.permute_dims(F, (1, 0, 2))
        F_agent, F_noise = split_channel_matrix(F, num_agents)

        print(F_agent.shape, F_noise.shape)
        return

    return compute_group_empowerment

VALID_STATE_TYPES = ['angle', 'velocity', 'full']

def build_linked_pendulum_state_matrix(state_type: str):

    assert state_type in VALID_STATE_TYPES
    
    if state_type == 'angle':
        state_matrix = jnp.array([[0], [1]])

    elif state_type == 'velocity':
        state_matrix = jnp.array([[2], [3]])

    elif state_type == 'full':
        state_matrix = jnp.array([[0, 2], [1, 3]])
    
    return state_matrix

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')
    ## hyperparams
    seed = 123123
    key = jax.random.key(seed)
    steps = 2000
    alpha = 0.01
    horizon = 150
    observation_noise = 1.0
    stiffness = 3.0
    damping = 0.1
    state_type = 'angle'
    assert state_type in VALID_STATE_TYPES

    ## the torque (power density of each agent)
    power_density = jnp.array([0.9, 1.4])

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dyn = set_tendon_properties(dyn, stiffness, damping)

    step = make_step(dyn)
    unroll = make_unroll(step)

    U = jnp.zeros((horizon, dyn.control_dim))


    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(3.1)
    # X = unroll(xt, U)
    # dyn.render(X, path = 'test.mp4')

    state_matrix = build_linked_pendulum_state_matrix('full')
    # compute_group_empowerment = make_compute_group_empowerment(step, state_matrix, U, power_density, alpha, observation_noise)
    # compute_group_empowerment(xt)

    AGENT_0_STATE, AGENT_1_STATE = state_matrix
    print(AGENT_0_STATE)
    print(AGENT_1_STATE)