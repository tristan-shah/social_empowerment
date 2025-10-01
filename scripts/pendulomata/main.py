import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_empowerment, compute_empowerment_grad

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    key = jax.random.key(0)
    steps = 3000  ## simulation horizon
    power = 1.0
    alpha = 0.01
    horizon = 200
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/pendulomata.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep

    print(dyn.state_dim, dyn.control_dim)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))
    
    xt = dyn.init_state()

    quat0 = Rotation.from_rotvec([0.0, 0.0, 0.0]).as_quat()
    quat0 = jnp.roll(quat0, shift = 1)

    xt = xt.at[:4].set(quat0)
    xt = xt.at[4:8].set(quat0)
    xt = xt.at[8:12].set(quat0)
    xt = xt.at[12:16].set(quat0)

    ## tensor for state storage
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    ## zero control planning horizon
    U = jnp.zeros((horizon, dyn.control_dim))

    empowerment_hist = []
    for t in range(steps):

        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) ## obtain control gain
        grad_E = compute_empowerment_grad(dyn, xt, U, power) ## compute gradient of empowerment

        ## bang bang control
        ut = B.T @ grad_E
        ut = jnp.sign(ut) * power
        ut = ut.at[ut == 0].set(power)

        e = compute_empowerment(dyn, xt, U, power)
        empowerment_hist.append(e)
        print(t, xt, ut, e)

        ## propagate dynamics
        xt = dyn.step(xt, ut)

        ## log state
        X = X.at[t+1].set(xt)


    times = jnp.linspace(0.0, X.shape[0] * dt, X.shape[0]-1)

    ## plotting the empowerment over time
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Single Pendulum Empowerment', fontsize = 14)
    ax.set_xlim(0.0, steps * dt)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.set_ylabel('Empowerment (Nats)',  fontsize = 14)
    ax.plot(times, empowerment_hist)
    fig.tight_layout()
    fig.savefig(f'pendulomata.png', dpi = 300)

    ## render an animation
    # X = unroll(dyn, xt, U)
    dyn.render(X, path = 'pendulomata.mp4', distance = 4.0, elevation = -45, skip = 2)