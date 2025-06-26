import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from einops import einsum
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F, split_channel_matrix, waterfilling_operator, batch_diag

if __name__ == '__main__':

    ## hyperparams
    key = jax.random.key(9756)
    T = 1000 ## simulation horizon
    empowerment_horizon = 50
    max_power = 1.0
    num_agents = 2
    power = jnp.array([1.0] * num_agents)

    ## load in xml
    xml_path = 'xml/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)

    ## obtain channel matrix from dynamics
    dx = dyn.state_dim
    du = dyn.control_dim // num_agents
    xt = jnp.concatenate([mjx.make_data(dyn.mjx_model).qpos, jnp.zeros(dyn.nv)])
    xt = xt.at[0].set(-0.1)
    xt = xt.at[1].set(-3.1)

    print(xt)

    U = jnp.zeros((empowerment_horizon, dyn.control_dim))
    F = compute_F(dyn, xt, U)

    # ## random channel matrix
    # dx = 7
    # du = 10
    # F = jax.random.normal(key, (dx, empowerment_horizon, du * num_agents))
    
    # S = jnp.zeros((num_agents, du * empowerment_horizon, du * empowerment_horizon))
    S = batch_diag(jax.random.uniform(key, (num_agents, du * empowerment_horizon) ))
    S_z = jnp.eye(dx) + jnp.diag(jax.random.normal(key, (dx))) * 1e-5
    F_agent, F_noise = split_channel_matrix(F, num_agents)

    iterations = 5
    e_hist = jnp.zeros((iterations, num_agents))

    alpha = 0.5

    for i in range(iterations):
        e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_

        e_hist = e_hist.at[i].set(e)

    print(e_hist)

    fig, ax = plt.subplots(1, 1)
    for agent in range(num_agents):
        ax.plot(e_hist[:, agent])
    plt.show()




    # ## tensor for state storage
    # X = jnp.zeros((T + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # for t in range(T):

    #     ut = jnp.zeros(dyn.control_dim)

    #     ## propagate dynamics
    #     xt = dyn.step(xt, ut)
    #     print(t, xt)

    #     ## log state
    #     X = X.at[t+1].set(xt)

    # dyn.render(X, path = 'linked_pendulums.mp4')