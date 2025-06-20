from jax import numpy as jnp
from mujoco import mjx

from soc_emp import Dynamics

if __name__ == '__main__':

    ## simulation horizon
    empowerment_horizon = 50
    max_power = 1.0
    T = 500

    ## load in xml
    xml_path = 'xml/linked_pendulums.xml'

    dyn = Dynamics(path = xml_path)

    ## initialize state
    xt = jnp.concatenate([mjx.make_data(dyn.mjx_model).qpos, jnp.zeros(dyn.nv)])
    xt = xt.at[0].set(3.1)
    xt = xt.at[1].set(2.0)

    print(xt)

    ## tensor for state storage
    X = jnp.zeros((T + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    for t in range(T):

        ut = jnp.zeros(dyn.control_dim)

        ## propagate dynamics
        xt = dyn.step(xt, ut)
        print(t, xt)

        ## log state
        X = X.at[t+1].set(xt)

    dyn.render(X, path = 'linked_pendulums.mp4')