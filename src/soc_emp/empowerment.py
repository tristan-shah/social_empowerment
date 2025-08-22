import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum, rearrange

from soc_emp import Dynamics
from soc_emp.utils import select_output

tol = 1e-6

@jax.jit
def compute_power(water_line: Array, eigs: Array):
    return jnp.clip(water_line - 1 / eigs, min = 0.0)

def unroll(dyn: Dynamics, xt: Array, U: Array):
    '''
    Jax compatable simulation loop.
    '''

    def body_fun(xt_: Array, ut_: Array):
        xt_next = dyn.step(xt_, ut_)
        return xt_next, xt_next
    
    _, X = jax.lax.scan(body_fun, xt, U)
    return jnp.concatenate([xt[None, :], X])

## jit compilation of the unroll function
unroll = jax.jit(unroll, static_argnums = 0)

def get_last_state(dyn: Dynamics, xt: Array, U: Array):
    '''
    Extracts the last state from the unroll function.
    '''
    return unroll(dyn, xt, U)[-1]

## compute the gradient of the final state w.r.t each action.
compute_F = jax.jit(jax.jacfwd(get_last_state, argnums = 2), static_argnums = 0)

@jax.jit
def compute_F_from_A_B(A: Array, B: Array):
    
    I = jnp.eye(A.shape[-1])

    def body_fun(_I: Array, ab: tuple):
        a, b = ab
        return _I @ a, _I @ b

    _, F = jax.lax.scan(body_fun, I, (A, B), reverse = True)

    return F

@jax.jit
def waterfilling_solver(noise_levels: Array, total_power: float):
    '''
    Code courtesy of Noam Smilovich. 
    '''
    clipped_noise = jnp.maximum(noise_levels, tol)
    inverse = 1.0 / clipped_noise

    inverse_sorted = jnp.sort(inverse)
    n = len(inverse_sorted)
    P = jnp.zeros_like(inverse_sorted)

    def compute_P(carry, i):
        P_prev = carry
        P_next = P_prev + i * (inverse_sorted[i] - inverse_sorted[i-1])
        return P_next, P_next

    _, P = jax.lax.scan(compute_P, 0.0, jnp.arange(1, n))
    P = jnp.concatenate([jnp.array([0.0]), P])  # P[0] = 0

    def cond_fun(state):
        bot, top = state
        return top - bot > 1

    def body_fun(state):
        bot, top = state
        mid = (bot + top) // 2
        new_bot = jax.lax.cond(total_power >= P[mid], lambda: mid, lambda: bot)
        new_top = jax.lax.cond(total_power >= P[mid], lambda: top, lambda: mid)
        return new_bot, new_top

    bot, _ = jax.lax.while_loop(cond_fun, body_fun, (0, n))

    mu = inverse_sorted[bot] + (total_power - P[bot]) / (bot + 1)
    return mu

@jax.jit
def waterfilling_implicit(noise_levels: Array, total_power: float):
    '''
    Code courtesy of Noam Smilovich. 
    '''
    safe_noise = jnp.maximum(noise_levels, tol)

    def f(mu):
        return jnp.sum(compute_power(mu, safe_noise)) - total_power
    
    initial_guess = 0.0
    
    def solve(f, initial_guess):
        return waterfilling_solver(noise_levels, total_power)
    
    def tangent_solve(g, y):
        return y/g(1.0)
    
    return jax.lax.custom_root(f, initial_guess, solve, tangent_solve)

def compute_empowerment(dyn: Dynamics, xt: Array, U: Array, P: float):
    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))
    
    ## S is the covariance matrix of the final state.
    S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
    h2 = jnp.linalg.eigvalsh(S)
    v = waterfilling_implicit(h2, P)
    p = compute_power(v, h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    return e

compute_empowerment = jax.jit(compute_empowerment, static_argnums = 0)
compute_empowerment_grad = jax.jit(jax.jacfwd(compute_empowerment, argnums = 1), static_argnums = 0)

def split_channel_matrix(F: Array, num_agents: int):
    '''
    Takes in a large channel matrix and splits it into two components:
    1.) A channel matrix which sends an agents actions to the big state
    2.) All other channel matrices interpreted as noise for the i'th agent

    We assume that each agent has the same action dimentionality. 
    For example all agents would have u \in R^2.

    Args:
    F: Channel matrix with size (combined state x time x all actions)

    Returns:
    F_agent: (agents x big state x message) sensitivity matrix of big state to agents own actions.
    F_noise: (agents x agents x big state x message) sensitivity of big state to all other agents actions.
    '''
    # assert F.shape[2] % num_agents == 0

    ## chunking the channel matrix along the action dimention to split the effect of each agent
    ## assumes that each agent has the same dimention of actions so it chunks it evenly
    F_agent = jnp.split(F, num_agents, axis = 2)
    F_agent = jnp.stack(F_agent, axis = 0)
    F_agent = rearrange(F_agent, 'a x t u -> a x (t u)') ## collapse the action dimention into time

    ## for each agent, the effect of all other agents is considered noise
    F_noise = F_agent[None, :, :, :].repeat(num_agents, axis = 0)
    ## create a mask with ones everywhere, except zeros on the diagonal
    ## zeroing out the diagonal means that the noise for each agent will not include its own channel matrix
    mask = jnp.ones_like(F_noise)
    mask = mask.at[jnp.arange(num_agents), jnp.arange(num_agents)].set(0.0)
    F_noise = F_noise * mask

    return F_agent, F_noise

batch_diag = jax.jit(jax.vmap(jnp.diag))
batch_water_filling = jax.jit(jax.vmap(waterfilling_implicit))
batch_compute_power = jax.jit(jax.vmap(compute_power))

@jax.jit
def waterfilling_operator(F_agent: Array, F_noise: Array, S: Array, S_z: Array, power: Array):
    '''
    This is the core operator which updates each agent's covariance S[agent].
    It takes in a batch of agent channel matrices (one for each agent) 
    and also the interference channel matrix from all the other agents.

    F_agent is the agent's channel matrix to its own state
    F_noise is the interference channel from all other agents
    S is the batch of covariance matrices
    S_z is the observation noise (assumed to be the same for all agents)
    power is the power budget for each agent
    '''

    S_noise = einsum(F_noise, S, F_noise, 'a1 a2 x1 m1, a2 m1 m2, a1 a2 x2 m2 -> a1 x1 x2') + S_z

    ## eigen-decomp on noise
    D, Q = jnp.linalg.eigh(S_noise)
    D_inv_sqrt = batch_diag((D + 1e-12) ** -0.5)

    ## compute new channel matrix (1/\sqrt{D}) @ Q.T @ F_agent
    H = einsum(D_inv_sqrt, Q, F_agent, 'a x1 x2, a x3 x2, a x3 m -> a x1 m')

    ## compute snr levels
    _, E, M = jnp.linalg.svd(H, full_matrices = False)
    eigs = jnp.power(E, 2.0).clip(min = 1e-12)

    ## compute waterline (for each agent) [nu_0, nu_1, ..., nu_k]
    nu = batch_water_filling(eigs, power)
    p = batch_compute_power(nu, eigs)
    
    ## update covariances
    P = batch_diag(p)
    '''
    M.T P M appears to be the reverse of what is defined in boyd's paper on IWF but it is correct because 
    SVD returns a transposed orthoganal matrix.
    '''
    ## obtain dense covariance matrices: M.T P M
    S = einsum(M, P, M, 'a x1 m1, a x1 x2, a x2 m2 -> a m1 m2')

    ## channel capacities
    e = 0.5 * jnp.sum(jnp.log(1.0 + p * eigs), axis = 1)
    return e, S

MAX_ITER = 10
# MAX_ITER = 50

def compute_multiagent_empowerment(
        dyn: Dynamics, 
        x0: Array, 
        U: Array, 
        power: Array, 
        alpha: float,
        key,
        observation_noise: float = 1.0):

    num_agents = len(power)
    horizon = U.shape[0]
    # dx = dyn.state_dim
    du = dyn.control_dim // num_agents
    dm = du * horizon

    # S = jnp.zeros((num_agents, dm, dm))
    S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)
    # S_z = jnp.eye(dx) + jnp.diag(jax.random.normal(key, (dx))) * 1e-5

    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    F_agent, F_noise = split_channel_matrix(F, num_agents)

    # hardcoded noise perturbation
    # S_z = jnp.eye(2) + jnp.diag(jax.random.normal(key, (2))) * 1e-5
    S_z = jnp.eye(2) * observation_noise
    # S_z = jnp.eye(2)
    
    ## egoistic double pendulum
    F_agent = jnp.stack([
        F_agent[0, [0, 2], :],
        F_agent[1, [1, 3], :]
        ], axis = 0)

    F_noise = jnp.stack([
        F_noise[0, :, [0, 2], :],
        F_noise[1, :, [1, 3], :]
    ], axis = 0)

    '''
    Explicit iteration
    '''
    max_iter = MAX_ITER

    def cond_fun(state):
        i, S, e, e_prev = state
        return jnp.logical_and(
            jnp.any(jnp.abs(e - e_prev) > 1e-5),
            i < max_iter
        )

    def body_fun(state):
        i, S, e, e_prev = state
        e_prev = e
        e, S_ = waterfilling_operator(F_agent, F_noise, S, S_z, power)
        S = alpha * S + (1 - alpha) * S_
        return (i + 1, S, e, e_prev)

    e_prev = jnp.ones(num_agents) * jnp.inf
    e = jnp.zeros(num_agents)
    i, S, e, e_prev = jax.lax.while_loop(cond_fun, body_fun, (0, S, e, e_prev))

    return i, e

compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = 0)
compute_multiagent_empowerment_grad = jax.jit(
    jax.jacfwd(
        select_output(compute_multiagent_empowerment, 1), 
        argnums = 1), 
    static_argnums = 0)