import jax
from jax import numpy as jnp
from mujoco import mjx

from soc_emp import Dynamics

if __name__ == '__main__':

    T = 1000

    xml_path = 'xml/blob.xml'
    dyn = Dynamics(path = xml_path)
    
    xt = jnp.concatenate([
        mjx.make_data(dyn.mjx_model).qpos,
        jnp.zeros(dyn.nv)
    ])

    key = jax.random.PRNGKey(0)

    X = jnp.zeros((T+1, dyn.state_dim))
    X = X.at[0].set(xt)
    U = jnp.zeros((T, dyn.control_dim))

    for t in range(T):

        key, subkey = jax.random.split(key)
        ut = jax.random.normal(subkey, (dyn.control_dim,))

        print(xt)
        xt = dyn.step(xt, ut)
        
        X = X.at[t+1].set(xt)
        U = U.at[t].set(ut)

    dyn.render(X, path = 'blob.mp4')
