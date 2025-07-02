import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from mujoco import mjx

import matplotlib.pyplot as plt

from soc_emp.dynamics import Dynamics
from soc_emp.empowerment import compute_empowerment, compute_empowerment_grad

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    T = 1000
    empowerment_horizon = 20
    max_power = 1.0

    dyn = Dynamics(path = 'xml/custom/blob_hill.xml')
    
    ## initialize state
    xt = dyn.init_state()

    ## zero control planning horizon
    U = jnp.zeros((empowerment_horizon, dyn.control_dim))
    X = jnp.zeros((T+1, dyn.state_dim))

    empowerment_hist = []
    for t in range(T):

        ## obtain control gain
        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) 
        ## compute gradient of empowerment
        grad_E = compute_empowerment_grad(dyn, xt, U, max_power)

        ## bang bang control
        ut = B.T @ grad_E
        ut = jnp.sign(ut) * max_power
        ut = ut.at[ut == 0].set(max_power)

        e = compute_empowerment(dyn, xt, U, max_power)
        empowerment_hist.append(e)

        # ut = jnp.zeros(dyn.control_dim)

        # print(t, xt, ut)
        print(t, xt, ut, e)

        ## propagate dynamics
        xt = dyn.step(xt, ut)

        ## log state
        X = X.at[t+1].set(xt)

    ## plotting the empowerment over time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Empowerment')
    ax.plot(empowerment_hist)
    fig.tight_layout()
    fig.savefig(f'blob_emp.png', dpi = 300)

    ## render an animation
    dyn.render(
        X, 
        path = 'blob_emp.mp4', 
        skip = 1, 
        lookat = jnp.array([2.5, 0.0, 0.5]),
        elevation = -10,
        distance = 5.0)