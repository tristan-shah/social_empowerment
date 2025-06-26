import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum, rearrange

from soc_emp import Dynamics

tol = 1e-6

def unroll(dyn: Dynamics, xt: Array, U: Array):
    '''
    Jax compatable simulation loop.
    '''

    def body_fun(xt_: Array, ut_: Array):
        xt_next = dyn.step(xt_, ut_)
        return xt_next, xt_next
    
    xt, _ = jax.lax.scan(body_fun, xt, U)

    return xt

compute_F = jax.jit(jax.jacfwd(unroll, argnums = 2), static_argnums = 0)

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
        return jnp.sum(jnp.maximum(0.0, mu - 1.0 / safe_noise)) - total_power
    
    initial_guess = 0.0
    
    def solve(f, initial_guess):
        return waterfilling_solver(noise_levels, total_power)
    
    def tangent_solve(g, y):
        return y/g(1.0)
    
    return jax.lax.custom_root(f, initial_guess, solve, tangent_solve)

def compute_empowerment(dyn: Dynamics, xt: Array, U: Array, P: float):

    F = compute_F(dyn, xt, U)
    S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
    h2 = jnp.linalg.eigvalsh(S)
    v = waterfilling_implicit(h2, P)
    p = jnp.maximum(jnp.array(0.0), v - 1 / h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    return e

compute_empowerment = jax.jit(compute_empowerment, static_argnums = 0)
compute_empowerment_grad = jax.jit(jax.jacfwd(compute_empowerment, argnums = 1), static_argnums = 0)


# def build_noise_matrix(F_agent: Array):
#     '''
#     Constructs a matrix for each agent where its own sensitivity is zeroed and the rest are treated as noise.
#     Interprets the first dimention of H_agent as the number of agents.

#     Args:
#     H_agent: (num_agents x output x input)
#     '''
#     num_agents = H_agent.shape[0]

#     ## for each agent, the effect of all other agents is considered noise
#     H_noise = H_agent.unsqueeze(0).repeat(num_agents, 1, 1, 1)

#     ## for each agent zero out its own effect
#     for agent in range(num_agents):
#         H_noise[agent, agent, :, :] *= 0.0

#     return H_noise

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
    assert F.shape[2] % num_agents == 0

    ## chunking the channel matrix along the action dimention to split the effect of each agent
    F_agent = jnp.split(F, num_agents, axis = 2)
    F_agent = jnp.stack(F_agent, axis = 0)
    F_agent = rearrange(F_agent, 'a x t u -> a x (t u)') ## collapse the action dimention into time

    ## for each agent, the effect of all other agents is considered noise
    F_noise = F_agent[None, :, :, :].repeat(num_agents, axis = 0)

    ## create a mask with ones everywhere, except zeros on the diagonal
    mask = jnp.ones_like(F_noise)
    mask = mask.at[jnp.arange(num_agents), jnp.arange(num_agents)].set(0.0)
    F_noise = F_noise * mask

    return F_agent, F_noise

diag_embed = jax.jit(jax.vmap(jnp.diag))

def waterfilling_operator(F_agent: Array, F_noise: Array, S: Array, S_z: Array):

    num_agents, state_dim, message_dim = F_agent.shape
    S_noise = einsum(F_noise, S, F_noise, 'a1 a2 x1 m1, a2 m1 m2, a1 a2 x2 m2 -> a1 x1 x2') + S_z

    ## eigen-decomp on noise
    D, Q = jnp.linalg.eigh(S_noise)
    # D_inv_sqrt = torch.diag_embed(D**-0.5)
    D = diag_embed(D)

    print(D)

    return