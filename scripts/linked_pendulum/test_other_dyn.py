import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, batch_diag
from soc_emp.utils import split_state, get_state

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

# def step(xt: Array, ut: Array, k: Array):

#     g = 9.81

#     xt_dot = jnp.stack([
#         xt[2],
#         xt[3],
#         -g * jnp.sin(xt[0]) + ut[0] + k.squeeze() * (xt[1] - xt[0]),
#         -g * jnp.sin(xt[1]) + ut[1] + k.squeeze() * (xt[0] - xt[1])
#     ])
    
#     return xt + xt_dot * 0.01

def step(xt: Array, ut: Array, k: Array):
    g = 9.81
    c = 0.1
    
    tau0 = ut[0] + k.squeeze() * (xt[1] - xt[0])
    tau1 = ut[1] + k.squeeze() * (xt[0] - xt[1])

    xt_dot = jnp.stack([
        xt[2],  # dtheta1
        xt[3],  # dtheta2
        -g * jnp.sin(xt[0]) + tau0,
        -g * jnp.sin(xt[1]) + tau1
    ])

    return xt + xt_dot * 0.01

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')
    ## hyperparams
    seed = 0
    key = jax.random.key(seed)
    horizon = 1000
    k = jnp.array([5.0])
    dt = 0.01

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    U = jnp.zeros((horizon, dyn.control_dim))

    unroll = make_unroll(step, U)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(3.1)
    # xt = xt.at[1].set(3.1)
    X = unroll(xt, k)
    print(X)

    dyn.render(X, path = 'test.mp4', skip = 2)
