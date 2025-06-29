from typing import Optional

from jax import numpy as jnp

from soc_emp.dynamics import Dynamics

if __name__ == '__main__':

    T = 2000
    # dyn = Dynamics(path = 'xml/mujoco_menagerie/bitcraze_crazyflie_2/scene.xml')
    dyn = Dynamics(path = 'xml/mujoco_menagerie/bitcraze_crazyflie_2/swarm.xml')
    xt = dyn.init_state()

    print(dyn.state_dim)
    print(dyn.control_dim)

    ## tensor for state storage
    X = jnp.zeros((T + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    thrust = 0.26487 + 0.001

    for t in range(T):

        # ut = jnp.zeros(dyn.control_dim)
        # ut = ut.at[0].set(thrust)

        if t <= 200:
            ut = jnp.zeros(dyn.control_dim)
            ut = ut.at[0].set(thrust)
            ut = ut.at[2].set(-1.0)
        else:
            ut = jnp.zeros(dyn.control_dim)
            ut = ut.at[0].set(thrust)
            ut = ut.at[2].set(1.0)

        ## propagate dynamics
        xt = dyn.step(xt, ut)
        print(t, xt, ut)

        ## log state
        X = X.at[t+1].set(xt)

    dyn.render(
        X, 
        skip = 10,
        path = 'swarm.mp4', 
        distance = 1.5, 
        lookat = jnp.array([0.0, 0.0, 0.1]), 
        elevation = -10)