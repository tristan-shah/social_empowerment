from typing import Optional

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from mujoco import mjx
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_empowerment

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## simulation horizon
    T = 20
    P = 1.0

    ## load in xml
    xml_path = 'xml/cell/scene.xml'
    dyn = Dynamics(path = xml_path)

    ## initialize state
    xt = jnp.concatenate([mjx.make_data(dyn.mjx_model).qpos, jnp.zeros(dyn.nv)])

    ## tensors for storage
    X = jnp.zeros((T+1, dyn.state_dim))
    U = jnp.zeros((T, dyn.control_dim))

    E = jax.jacfwd(compute_empowerment, argnums = 1)
    # e = compute_empowerment(dyn, xt, U, P)
    # print(e)
    print(E(dyn, xt, U, P))







    # fig, ax = plt.subplots(1, 1)

    # for i in range(F.shape[0]):
    #     ax.plot(F[i, :, 0])
    # plt.show()

    # X = X.at[0].set(xt)

    # for t in range(T):

    #     key, subkey = jax.random.split(key)
    #     ut = jax.random.normal(subkey, (dyn.control_dim,))
    #     ut = jnp.zeros((dyn.control_dim,))

    #     xt = dyn.step(xt, ut)
        
    #     X = X.at[t+1].set(xt)
    #     U = U.at[t].set(ut)

    # dyn.render(X, path = 'empowerment.mp4')
