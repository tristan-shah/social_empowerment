import jax
from jax import numpy as jnp

from soc_emp import Dynamics

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum, rearrange
import mujoco
import imageio
import matplotlib.pyplot as plt
import numpy as np

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, select_output, batch_diag, compute_multiagent_control, waterfilling_implicit, compute_power
from soc_emp.utils import split_state, smooth_angle_wrap

def compute_multiagent_empowerment(
        dyn: Dynamics, 
        x0: Array, 
        U: Array, 
        power: Array, 
        alpha: float,
        observation_noise: float):

    num_agents = len(power)
    horizon = U.shape[0]
    dx = dyn.state_dim
    du = dyn.control_dim // num_agents
    dm = du * horizon

    '''
    original
    '''
    S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)

    # hardcoded noise perturbation
    S_z = jnp.eye(dx) * observation_noise

    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    F_agent, F_noise = split_channel_matrix(F, num_agents)

    # F_noise = jnp.zeros_like(F_noise)

    # i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)
    # return i, e, S

    F_agent = F_agent[:, 0:1, :]


    '''
    vanila single agent empowerment calculation
    '''
    S = einsum(F_agent, F_agent, 'a x1 u, a x2 u -> a x1 x2')
    h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
    v = jax.vmap(waterfilling_implicit)(h2, power)
    p = jax.vmap(compute_power)(v, h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2), axis = -1)
    return e

compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = 0)
# compute_multiagent_empowerment_grad = jax.jit(
#     jax.jacfwd(
#         select_output(compute_multiagent_empowerment, 1), 
#         argnums = 1),
#     static_argnums = 0)

compute_multiagent_empowerment_grad = jax.jit(jax.jacfwd(compute_multiagent_empowerment, argnums = 1), static_argnums = 0)

def hopper_initial_state():
    return jnp.array([0.0, -0.245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    seed = 12312
    key = jax.random.key(seed)

    steps = 500
    alpha = 0.01
    horizon = 50
    observation_noise = 1.0
    power = jnp.array([1, 1, 1])

    # load dynamics
    xml_path = 'xml/custom/unrestricted_hopper.xml'
    dyn = Dynamics(path = xml_path)
    print(dyn.state_dim, dyn.control_dim)
    dt = dyn.mjx_model.opt.timestep

    xt = hopper_initial_state()
    U = jnp.zeros((horizon, dyn.control_dim))

    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')
    
    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    iterations = jnp.zeros(steps)
    empowerment = jnp.zeros((steps, 3))
        
    for t in range(steps):
        key, sub_key = jax.random.split(key)

        ## probing power is proportional to the instant power times the horizon
        grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power * horizon, alpha, observation_noise)
        # i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power * horizon, alpha, observation_noise)
        e = compute_multiagent_empowerment(dyn, xt, U, power * horizon, alpha, observation_noise)

        ## log the number of IWF iterations and empowerment
        # iterations = iterations.at[t].set(i)
        empowerment = empowerment.at[t].set(e)

        _, B = dyn.linearize(xt, U[0])
        sensitivity = grad_E @ B
        ut = jnp.sign(jnp.diag(sensitivity)) * power
        # ut = jnp.sign(sensitivity[0, :]) * power
        # ut = jnp.sign(jnp.sum(sensitivity, axis = 0)) * power
        # print(ut)

        ## step the dynamics and record the result
        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)

        ## print out some relevant quantities
        # print(t, xt, ut, e, i)
        print(t, xt, ut, e)

    run_name = f'HOPPER-seed={seed}-horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    ## plot stats
    fig, ax = plt.subplots(2, 1)
    ## plot empowerment
    ax[0].plot(empowerment[:, 0])
    ax[0].plot(empowerment[:, 1])
    ax[0].plot(empowerment[:, 2])
    ax[0].set_ylabel('Empowerment\n(nats)')
    ax[0].tick_params(axis = 'both', labelsize = 12)

    ax[1].set_xlabel('Interaction Time (s)', fontsize = 14)
    ax[1].set_ylabel('Iterations')
    ax[1].plot(iterations)

    fig.tight_layout()
    fig.savefig(run_name + '.png', dpi = 300)
    plt.show()

    dyn.render(X, path = run_name + '.mp4', skip = 2, distance = 4)