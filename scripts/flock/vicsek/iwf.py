from functools import partial

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from soc_emp.dynamics import make_unroll
from soc_emp.envs.flock.vicsek import Vicsek, make_reset, make_step
from soc_emp.envs.flock.utils import decode_state, minimum_image_diff, build_flock_state_matrix
from soc_emp.envs.flock.plot import render_video



from jax import Array
def make_propagate(step: callable, stochastic: bool = False):
    '''
    Unrolls dynamical system with randomness
    '''

    if stochastic:
        '''
        If the propagate should take a key for randomness
        '''
        def propagate(x0: Array, U: Array, keys: Array):

            def body_fn(x: Array, inputs: tuple):

                u, key = inputs
                x_next = step(x, u, key)

                return x_next, None
            
            xT, _ = jax.lax.scan(body_fn, init = x0, xs = (U, keys))

            return xT
    else:
        '''
        If the propagate function is deterministic
        '''
        def propagate(x0: Array, U: Array):

            def body_fn(x: Array, u: Array):
                x_next = step(x, u)
                return x_next, x_next
            
            xT = jax.lax.scan(body_fn, x0, U)

            return xT
    
    return jax.jit(propagate)



# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)
# fig.suptitle(f'Connectivity of Flock With {max_neighors} Neighbors')
# ax.set_aspect('equal')
# ax.set_xlim(-grid_size, grid_size)
# ax.set_ylim(-grid_size, grid_size)
# ax.scatter(x, y)

# # Add a circle at each point
# for (x, y) in pos:
#     circle = patches.Circle((x, y), radius=neighbor_radius, fill=False, edgecolor='green', alpha = 0.5)
#     ax.add_patch(circle)

# for i in range(num_agents):

#     xi, yi = pos[i]

#     for j in neighbor_idx[i]:
#         xj, yj = pos[j]

#         ax.plot([xi, xj], [yi, yj], color = 'red', alpha = 0.5)

# fig.tight_layout()
# fig.savefig('flock.png', dpi = 300)
# plt.show()

if __name__ == '__main__':

    seed = 0
    key = jax.random.key(seed)
    num_agents = 30
    grid_size = 5.0
    neighbor_radius = 0.5
    speed = 1.0
    J = 0.1
    D = 0.0
    dt = 0.05

    max_neighors = 3

    horizon = 5

    flock = Vicsek(num_agents, grid_size, neighbor_radius, speed, J, D, dt)

    reset = make_reset(flock)
    step = make_step(flock)
    propagate = make_propagate(step, stochastic = True)

    x0 = reset(key)

    ut = jnp.zeros(flock.control_dim)

    ## nominal control sequence
    ## U[t, i] is the control of agent i at time t
    U = jnp.zeros((horizon, flock.control_dim))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, horizon)

    x, y, a = decode_state(x0, num_agents)

    pos = jnp.stack([x, y], axis = 1)

    ## compute position differences
    raw_diff = pos[:, None, :] - pos[None, :, :]
    diff_mi = minimum_image_diff(raw_diff, flock.grid_size)
    dist = jnp.linalg.norm(diff_mi + 1e-10, axis = 2)
    ## determine closest neighbors
    neighbor_idx = jnp.argsort(dist, axis = 1)[:, 1:max_neighors+1]

    ## indices of each agents state: state_matrix[i] -> indices of the ith agent
    ## state_matrix is (num_agents x agent_state_dim)
    state_matrix = build_flock_state_matrix(num_agents, 'full')
    
    ## propagation function
    F = partial(propagate, x0, keys = keys)
    xt, jvp = jax.linearize(F, U)


    ## How to extract sensitivity of each agents state to its own actions?

    # e = jnp.zeros_like(U).at[0, 0].set(1.0)
    # print(jvp(e))