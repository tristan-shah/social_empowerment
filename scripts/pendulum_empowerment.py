import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_empowerment, compute_empowerment_grad
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## simulation horizon
    empowerment_horizon = 50
    max_power = 1.0
    T = 1500

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    ## initialize state
    xt = dyn.init_state()

    ## tensor for state storage
    X = jnp.zeros((T + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    ## zero control planning horizon
    U = jnp.zeros((empowerment_horizon, dyn.control_dim))

    empowerment_hist = []
    for t in range(T):

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

    ## plotting the empowerment over time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Empowerment (nats)')
    ax.plot(empowerment_hist)
    plt.show()

    ## render an animation
    dyn.render(X, path = 'pendulum_empowerment.mp4')
