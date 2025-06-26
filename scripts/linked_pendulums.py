import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from einops import einsum

from soc_emp import Dynamics
from soc_emp.empowerment import split_channel_matrix, waterfilling_operator

def compute_power(water_line: Array, eigs: Array):
    return jnp.clip(water_line - 1 / eigs, min = 0.0)

if __name__ == '__main__':

    ## hyperparams
    key = jax.random.key(9756)
    T = 1000 ## simulation horizon
    empowerment_horizon = 50#50
    max_power = 1.0
    num_agents = 2

    power = jnp.array([1.0, 1.0])

    ## load in xml
    xml_path = 'xml/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    
    dx = dyn.state_dim
    du = dyn.control_dim // num_agents

    ## initialize state
    xt = jnp.concatenate([mjx.make_data(dyn.mjx_model).qpos, jnp.zeros(dyn.nv)])
    xt = xt.at[0].set(0.1)
    xt = xt.at[1].set(-0.1)
    # U = jnp.zeros((empowerment_horizon, dyn.control_dim))
    # F = compute_F(dyn, xt, U)

    F = jax.random.normal(key, (dyn.state_dim, empowerment_horizon, num_agents))
    S = jnp.zeros((num_agents, du * empowerment_horizon, du * empowerment_horizon))
    S_z = jnp.eye(dx) + jnp.diag(jax.random.normal(key, (dx))) * 1e-2
    F_agent, F_noise = split_channel_matrix(F, num_agents)
    waterfilling_operator(F_agent, F_noise, S, S_z)

    # S_noise = einsum(F_noise, S, F_noise, 'a1 a2 x1 m1, a2 m1 m2, a1 a2 x2 m2 -> a1 x1 x2')






    # ## power covariances for agents. initial power allocation is zero
    # P_0 = jnp.zeros((empowerment_horizon, empowerment_horizon))
    # P_1 = jnp.zeros((empowerment_horizon, empowerment_horizon))
    # ## standard noise covariance for each agent's observation
    # ## need to regularize the identity matrix so the gradient through eigenvalue decomp is defined
    # S_z = jnp.eye(2) + jnp.diag(jax.random.normal(key, (2))) * 1e-5
    # ## noise covariance for agent 0
    # S_0 = F_0_noise @ P_1 @ F_0_noise.T + S_z
    # D, Q = jnp.linalg.eigh(S_0)                           ## eigen-decomp on noise
    # H = jnp.diag((D + 1e-12) ** -0.5) @ Q.T @ F_0         ## define new channel matrix
    # _, E, M_0 = jnp.linalg.svd(H, full_matrices = False)  ## svd on channel matrix
    # ## compute channel capacity
    # eigs_0 = E ** 2
    # mu_0 = waterfilling_implicit(eigs_0, power[0])
    # p_0 = compute_power(mu_0, eigs_0)
    # # ## update action covariance matrix
    # P_0 = M_0.T @ jnp.diag(p_0) @ M_0
    # print(p_0)







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