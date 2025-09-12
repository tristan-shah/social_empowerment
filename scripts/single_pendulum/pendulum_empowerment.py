import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_empowerment, compute_empowerment_grad
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## simulation horizon
    horizon = 50
    max_power = 1.0
    steps = 1500

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.01)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## initialize state
    xt = dyn.init_state()

    ## tensor for state storage
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    ## zero control planning horizon
    U = jnp.zeros((horizon, dyn.control_dim))

    empowerment_hist = []
    for t in range(steps):

        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) ## obtain control gain
        grad_E = compute_empowerment_grad(dyn, xt, U, max_power) ## compute gradient of empowerment

        ## bang bang control
        ut = B.T @ grad_E
        ut = jnp.sign(ut) * max_power
        ut = ut.at[ut == 0].set(max_power)

        e = compute_empowerment(dyn, xt, U, max_power)
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
    ax.set_xlim(0.0, 15.0)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.set_ylabel('Empowerment (Nats)',  fontsize = 14)
    ax.plot(times, empowerment_hist)
    fig.tight_layout()
    fig.savefig(f'pendulum_empowerment.png', dpi = 300)

    ## render an animation
    dyn.render(X, path = 'pendulum_empowerment.mp4', skip = 5)
