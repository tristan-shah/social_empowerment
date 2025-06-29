import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_multiagent_empowerment_grad

# from soc_emp.empowerment import compute_F, split_channel_matrix, waterfilling_operator, batch_diag

# def iwf_cond(H_agent, H_noise, S, S_z, i):
#     denom = H_agent @ S[i] @ H_agent.T + einsum(H_noise, S, H_noise, 'a x1 m1, a m1 m2, a x2 m2 -> x1 x2') + S_z
#     return H_agent.T @ jnp.linalg.inv(denom) @ H_agent

# def test(
#         dyn: Dynamics, 
#         x0: Array, 
#         U: Array, 
#         power: Array, 
#         alpha: float,
#         key):

#     num_agents = len(power)
#     horizon = U.shape[0]
#     dx = dyn.state_dim
#     du = dyn.control_dim // num_agents
#     dm = du * horizon

#     # S = jnp.zeros((num_agents, dm, dm))
#     S = batch_diag(jax.random.uniform(key, (num_agents, dm)))
#     S_z = jnp.eye(dx)# + jnp.diag(jax.random.normal(key, (dx))) * 1e-5

#     # F = compute_F(dyn, x0, U)

#     F = jax.random.normal(key, (4, U.shape[0], num_agents))
#     F_agent, F_noise = split_channel_matrix(F, num_agents)

#     # ## egoistic
#     # S_z = jnp.eye(2) + jnp.diag(jax.random.normal(key, (2))) * 1e-5
#     # F_agent = jnp.stack([
#     #     F_agent[0, [0, 2], :],
#     #     F_agent[1, [1, 3], :]
#     #     ], axis = 0)

#     # F_noise = jnp.stack([
#     #     F_noise[0, :, [0, 2], :],
#     #     F_noise[1, :, [1, 3], :]
#     # ], axis = 0)

#     iterations = 20

#     hist = jnp.zeros((iterations, 2))

#     for i in range(iterations):
#         e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
#         S = alpha * S + (1 - alpha) * S_

#         # print(e)
#         hist = hist.at[i].set(e)

#     # fig, ax = plt.subplots(1, 1)
#     # for k in range(hist.shape[1]):
#     #     ax.plot(hist[:, k])
#     # plt.show()

#     i = 0
#     H_agent = F_agent[i]
#     H_noise = F_noise[i]

#     dgds = jax.jacfwd(iwf_cond, argnums = 2)(H_agent, H_noise, S, S_z, i)
    
#     print(dgds)
#     print(dgds.shape)
#     return e

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
    alpha = 0.0

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

    hist = jnp.zeros((T, num_agents))
    ke_hist = jnp.zeros((T, num_agents))
    pe_hist = jnp.zeros((T, num_agents))

    for t in range(T):

        ## compute empowerment
        e = compute_multiagent_empowerment(dyn, xt, U, power, alpha, key)
        hist = hist.at[t].set(e)

        ## obtain control gain
        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) 
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, alpha, key)

        ## compute action
        ut = jnp.sign(jnp.diag(grad_E @ B)) * power

        if ut[0] == 0:
            ut = ut.at[0].set(power[0])
        if ut[1] == 0:
            ut = ut.at[1].set(power[1])

        ke_hist = ke_hist.at[t].set(kinetic_energy(xt))
        pe_hist = pe_hist.at[t].set(potential_energy(xt))

        ## propagate dynamics
        xt = dyn.step(xt, ut)
        print(t, xt, ut, e, ke_hist[t], pe_hist[t])

        ## log state
        X = X.at[t+1].set(xt)

    name = f'egoistic-left={power[0]}-right={power[1]}-horizon={empowerment_horizon}'

    fig, ax = plt.subplots(3, 1)

    ax[0].set_ylabel('Empowerment')
    ax[0].plot(hist[:, 0], label = 'Pendulum 0')
    ax[0].plot(hist[:, 1], label = 'Pendulum 1')

    ax[1].set_ylabel('Kinetic Energy')
    ax[1].plot(ke_hist[:, 0], label = 'Pendulum 0')
    ax[1].plot(ke_hist[:, 1], label = 'Pendulum 1')

    ax[2].set_ylabel('Potenital Energy')
    ax[2].plot(pe_hist[:, 0], label = 'Pendulum 0')
    ax[2].plot(pe_hist[:, 1], label = 'Pendulum 1')

    ax[2].set_xlabel('Timestep')

    fig.tight_layout()
    fig.savefig(f'{name}.png', dpi = 300)
    plt.show()

    skip = 2
    dyn.render(
        X,
        path = f'{name}.mp4',
        skip = skip)