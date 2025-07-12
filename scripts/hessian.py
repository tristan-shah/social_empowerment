import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B
from soc_emp.utils import split_state

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    dyn = Dynamics(path = 'xml/custom/pendulum.xml')

    x0 = jnp.zeros((dyn.state_dim,))
    x0 = x0.at[:].set([3.0, 1.3])
    u0 = jnp.zeros((dyn.control_dim,))

    x0_bar = jnp.zeros((dyn.state_dim,))
    x0_bar = x0_bar.at[:].set([3.14, 1.3])

    qpos, _ = split_state(x0, dyn.nq)
    qpos_bar, _ = split_state(x0_bar, dyn.nq)

    r = np.zeros(dyn.nv)  # Output array, size of number of velocities
    # r = jnp.zeros(dyn.nv)
    mujoco.mj_differentiatePos(dyn.model, r, 1.0, qpos_bar, qpos)
    print(jnp.array(r))




    # f_u = jax.jacfwd(dyn.step, argnums = 1)
    # f_uu = jax.jacfwd(f_u, argnums = 0)

    # f_x = jax.jacfwd(dyn.step, argnums = 0)
    # f_xx = jax.jacfwd(f_x, argnums = 0)

    # f_xu = jax.jacfwd(f_x, argnums = 1)
    # f_ux = jax.jacfwd(f_u, argnums = 0)

    # print( f_u(x0, u0) )
    # print( f_uu(x0, u0) )

    # print()

    # print( f_x(x0, u0) )
    # print( f_xx(x0, u0) )

    # print()

    # print( f_xu(x0, u0) )
    # print( f_ux(x0, u0) )