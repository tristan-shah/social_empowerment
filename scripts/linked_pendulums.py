import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from einops import einsum
import matplotlib.pyplot as plt

from soc_emp import Dynamics
from soc_emp.empowerment import compute_multiagent_empowerment, compute_F
from soc_emp.empowerment import waterfilling_implicit, compute_power

compute_multiagent_empowerment_grad = jax.jit(jax.jacfwd(compute_multiagent_empowerment, argnums = 1), static_argnums = 0)

def compute_linked_pendulum_empowerment(dyn: Dynamics, xt: Array, U: Array, power: Array, key):

    empowerment_horizon = U.shape[0]

    F = compute_F(dyn, xt, U)

    ## sensitivity of agent's state to its own actions
    F_0 = F[[0, 2], :, 0]
    F_1 = F[[1, 3], :, 1]

    ## sensitivity of agent's state to the other agent's actions
    F_0_noise = F[[0, 2], :, 1]
    F_1_noise = F[[1, 3], :, 0]

    ## power covariances for agents. initial power allocation is zero
    P_0 = jnp.zeros((empowerment_horizon, empowerment_horizon))
    P_1 = jnp.zeros((empowerment_horizon, empowerment_horizon))
    ## standard noise covariance for each agent's observation
    ## need to regularize the identity matrix so the gradient through eigenvalue decomp is defined
    S_z = jnp.eye(2) + jnp.diag(jax.random.normal(key, (2,)) * 1e-5)

    ## noise covariance for agent 0
    S_0 = F_0_noise @ P_1 @ F_0_noise.T + S_z
    D, Q = jnp.linalg.eigh(S_0)                           ## eigen-decomp on noise
    H = jnp.diag((D + 1e-12) ** -0.5) @ Q.T @ F_0         ## define new channel matrix
    _, E, M_0 = jnp.linalg.svd(H, full_matrices = False)  ## svd on channel matrix
    ## compute channel capacity
    eigs_0 = E ** 2
    mu_0 = waterfilling_implicit(eigs_0, power[0])
    p_0 = compute_power(mu_0, eigs_0)
    # ## update action covariance matrix
    # P_0 = M_0.T @ jnp.diag(p_0) @ M_0

    ## noise covariance for agent 1
    S_1 = F_1_noise @ P_0 @ F_1_noise.T + S_z
    D, Q = jnp.linalg.eigh(S_1)                           ## eigen-decomp on noise
    H = jnp.diag((D + 1e-12) ** -0.5) @ Q.T @ F_1         ## define new channel matrix
    _, E, M_1 = jnp.linalg.svd(H, full_matrices = False)  ## svd on channel matrix
    ## compute channel capacity
    eigs_1 = E ** 2
    mu_1 = waterfilling_implicit(eigs_1, power[1])
    p_1 = compute_power(mu_1, eigs_1)
    ## update action covariance matrix

    P_0 = M_0.T @ jnp.diag(p_0) @ M_0
    P_1 = M_1.T @ jnp.diag(p_1) @ M_1

    ## (optional) compute empowerment
    e_0 = 0.5 * jnp.sum(jnp.log(1 + p_0 * eigs_0))
    e_1 = 0.5 * jnp.sum(jnp.log(1 + p_1 * eigs_1))
    e = jnp.stack([e_0, e_1])

    return e

compute_linked_pendulum_empowerment = jax.jit(compute_linked_pendulum_empowerment, static_argnums = 0)
compute_linked_pendulum_empowerment_grad = jax.jit(jax.jacfwd(compute_linked_pendulum_empowerment, argnums = 1), static_argnums = 0)

if __name__ == '__main__':

    ## hyperparams
    key = jax.random.key(0)
    T = 5000 ## simulation horizon
    empowerment_horizon = 50
    # empowerment_horizon = 20
    num_agents = 2
    power = jnp.array([1.0] * num_agents)
    iterations = 5
    alpha = 0.0

    ## load in xml
    xml_path = 'xml/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path)
    
    U = jnp.zeros((empowerment_horizon, dyn.control_dim))

    dx = dyn.state_dim
    du = dyn.control_dim // num_agents
    xt = dyn.init_state()
    xt = xt.at[0].set(0.01)
    xt = xt.at[1].set(0.0)

    ## tensor for state storage
    X = jnp.zeros((T + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    for t in range(T):

        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) ## obtain control gain
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power, iterations, alpha, key)
        # grad_E = compute_linked_pendulum_empowerment_grad(dyn, xt, U, power, key)

        ut = jnp.sign(jnp.diag(grad_E @ B)) * jnp.sqrt(power)

        if ut[0] == 0:
            ut = ut.at[0].set(power[0])
        if ut[1] == 0:
            ut = ut.at[1].set(power[1])

        # ut = jnp.zeros(dyn.control_dim)

        ## propagate dynamics
        xt = dyn.step(xt, ut)
        print(t, xt, ut)

        ## log state
        X = X.at[t+1].set(xt)

    k = 2
    dyn.render(X, path = f'euler_k={k}_egoistic_linked_pendulum_empowerment_no_advantage_stiff=3.0.mp4', k = k)
    
    # k = 1
    # dyn.render(X, path = f'RK4_k={k}_linked_pendulum_empowerment_no_advantage_stiff=5.0.mp4', k = k)