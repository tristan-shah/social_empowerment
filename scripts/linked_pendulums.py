import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad

def kinetic_energy(xt: Array):
    omega_0 = xt[2]
    omega_1 = xt[3]
    return 0.5 * jnp.stack([omega_0, omega_1]) ** 2

def potential_energy(xt: Array):
    theta_0 = xt[0]
    theta_1 = xt[1]
    g = 9.81
    return g * (1 - jnp.cos(jnp.stack([theta_0, theta_1])))

if __name__ == '__main__':

    ## hyperparams
    key = jax.random.key(5)
    T = 1500 ## simulation horizon
    empowerment_horizon = 50
    num_agents = 2
    power = jnp.array([1.0, 1.0])
    alpha = 0.01

    ## load in xml
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    
    U = jnp.zeros((empowerment_horizon, dyn.control_dim))

    dx = dyn.state_dim
    du = dyn.control_dim // num_agents
    xt = dyn.init_state()

    ## tensor for state storage
    X = jnp.zeros((T + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    iter_hist = jnp.zeros((T,))
    emp_hist = jnp.zeros((T, num_agents))
    ke_hist = jnp.zeros((T, num_agents))
    pe_hist = jnp.zeros((T, num_agents))

    # xt = xt.at[:].set([ 2.3031253,  -2.3020508,  -0.01055187,  0.00946543])

    print(xt)
    print(jax.devices())

    for t in range(T):

        ## compute empowerment
        iterations, e = compute_multiagent_empowerment(dyn, xt, U, power, alpha, key)

        ## obtain control gain
        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) 
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, key)

        ## compute action
        # ut = jnp.sign(jnp.diag(grad_E @ B)) * jnp.sqrt(power)
        # ut = ut + (ut == 0) * jnp.sqrt(power)

        ut = jnp.sign(jnp.diag(grad_E @ B)) * power
        ut = ut + (ut == 0) * power

        ## log stuff
        emp_hist = emp_hist.at[t].set(e)
        iter_hist = iter_hist.at[t].set(iterations)
        ke_hist = ke_hist.at[t].set(kinetic_energy(xt))
        pe_hist = pe_hist.at[t].set(potential_energy(xt))

        ## propagate dynamics
        xt = dyn.step(xt, ut)
        print(t, xt, ut, e, iterations)

        ## log state
        X = X.at[t+1].set(xt)

    name = f'left={power[0]}-right={power[1]}-horizon={empowerment_horizon}'

    fig, ax = plt.subplots(4, 1)

    ax[0].set_ylabel('Empowerment')
    ax[0].plot(emp_hist[:, 0])
    ax[0].plot(emp_hist[:, 1])

    ax[1].set_ylabel('Kinetic Energy')
    ax[1].plot(ke_hist[:, 0])
    ax[1].plot(ke_hist[:, 1])

    ax[2].set_ylabel('Potenital Energy')
    ax[2].plot(pe_hist[:, 0])
    ax[2].plot(pe_hist[:, 1])

    ax[3].set_ylabel('Iterations')
    ax[3].plot(iter_hist)
    ax[3].set_xlabel('Timestep')

    fig.tight_layout()
    fig.savefig(f'{name}.png', dpi = 300)
    # plt.show()

    skip = 2
    dyn.render(
        X,
        path = f'{name}.mp4',
        skip = skip)