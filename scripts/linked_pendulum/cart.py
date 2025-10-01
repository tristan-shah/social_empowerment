import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad, compute_multiagent_control, unroll
from soc_emp.utils import smooth_angle_wrap

if __name__ == '__main__':
    print(f'GPU devices: {jax.devices()}')

    ## hyperparams
    seed = 8
    key = jax.random.key(seed)
    steps = 1500  ## simulation horizon
    horizon = 150 #200
    power = 5.0

    # load dynamics
    xml_path = 'xml/custom/cart_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    dt = dyn.mjx_model.opt.timestep
    print(dyn.state_dim, dyn.control_dim)

    ## planning horizon should be the maximum of all agent's horizons
    U = jnp.zeros((horizon, dyn.control_dim))
    
    xt = dyn.init_state()

    from soc_emp.empowerment import compute_empowerment_grad, compute_empowerment

    ## tensor for state storage
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    empowerment_hist = []
    for t in range(steps):

        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) ## obtain control gain
        grad_E = compute_empowerment_grad(dyn, xt, U, power) ## compute gradient of empowerment

        ## bang bang control
        ut = B.T @ grad_E
        ut = jnp.sign(ut) * power
        ut = ut.at[ut == 0].set(power)

        e = compute_empowerment(dyn, xt, U,power)
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
    fig.savefig(f'pendulum_empowerment.png', dpi = 300)

    ## render an animation
    dyn.render(X, path = 'pendulum_empowerment.mp4', skip = 3)
