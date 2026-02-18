import jax
from jax import Array
from jax import numpy as jnp
import mujoco
import imageio
import matplotlib.pyplot as plt
import numpy as np

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, split_channel_matrix, iterative_waterfilling, select_output, batch_diag, compute_multiagent_control
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
    du = dyn.control_dim // num_agents
    dm = du * horizon

    '''
    egoistic double pendulum. Each agent only cares about its own state (angle, angular velocity).

    0 -> position of agent 0
    1 -> angle of agent 0
    2 -> position of agent 1
    3 -> angle of agent 1
    4 -> x velocity of agent 0
    5 -> angular velocity of agent 0
    6 -> x velocity of agent 1
    7 -> angular velocity of agent 1

    [0, 1, 4, 5] -> state of agent 0
    [2, 3, 6, 7] -> state of agent 1
    '''

    # AGENT_0_STATE = [0, 1, 4, 5]
    # AGENT_1_STATE = [2, 3, 6, 7]

    ## angles and positions
    AGENT_0_STATE = [0, 1]
    AGENT_1_STATE = [2, 3]

    '''
    original
    '''
    S = batch_diag(power[:, None] * jnp.ones((num_agents, dm)) / dm)

    # hardcoded noise perturbation
    S_z = jnp.eye(len(AGENT_0_STATE)) * observation_noise

    X = unroll(dyn, x0, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    ## stripping off passive control (index 2) to push the cart
    F = F[:, :, 0:2]

    F_agent, F_noise = split_channel_matrix(F, num_agents)

    print(F_noise)
    print(F_noise.shape)


    F_agent = jnp.stack([
        F_agent[0, AGENT_0_STATE, :],
        F_agent[1, AGENT_1_STATE, :]
        ], axis = 0)
    
    ## chained indexing allows to select the correct submatrices
    F_noise = jnp.stack([
        F_noise[0][:, AGENT_0_STATE, :],
        F_noise[1][:, AGENT_1_STATE, :]
    ], axis = 0)

    i, e, S = iterative_waterfilling(F_agent, F_noise, S, S_z, power, alpha)
    return i, e, S

# compute_multiagent_empowerment = jax.jit(compute_multiagent_empowerment, static_argnums = 0)
# compute_multiagent_empowerment_grad = jax.jit(
#     jax.jacfwd(
#         select_output(compute_multiagent_empowerment, 1), 
#         argnums = 1),
#     static_argnums = 0)

def render_cart_pole(
        dyn: Dynamics, 
        X: Array, 
        path: str,
        lookat: Array = jnp.array([0.0, 0.0, 1.0]),
        distance: float = 5.0,
        azimuth: float = 90.0,
        elevation: float = 0.0,
        skip: int = 1
        ):

    # --- timing ---
    dt = float(dyn.model.opt.timestep)
    video_fps = 60
    frame_period = 1.0 / video_fps
    accum = 0.0

    renderer = mujoco.Renderer(dyn.model, height = 1080, width = 1920)

    camera = mujoco.MjvCamera()
    camera.lookat = lookat.copy()
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation

    cart_id = mujoco.mj_name2id(
        dyn.model, mujoco.mjtObj.mjOBJ_BODY, 'cart:1'
    )

    data = mujoco.MjData(dyn.model)

    with imageio.get_writer(path, fps=video_fps) as writer:

        for t in range(0, X.shape[0], skip):

            qpos, qvel = split_state(X[t], dyn.nq)
            data.qpos[:] = qpos
            data.qvel[:] = qvel

            mujoco.mj_forward(dyn.model, data)

            # camera tracking
            camera.lookat[:] = data.xpos[cart_id]

            accum += dt * skip

            # resample simulation → video time
            while accum >= frame_period:
                accum -= frame_period

                renderer.update_scene(data, camera=camera)
                img = renderer.render()
                writer.append_data(img)

    renderer.close()

if __name__ == '__main__':

    seed = 12312
    key = jax.random.key(seed)

    steps = 2000
    alpha = 0.01
    horizon = 5
    observation_noise = 1.0
    power = jnp.array([2.0, 2.0])

    ## physical parameters
    stiffness = 0.0
    damping = 0.0

    right_force = 0.5

    # load dynamics
    xml_path = 'xml/custom/free_carts/linked.xml'
    dyn = Dynamics(path = xml_path)
    print(dyn.state_dim, dyn.control_dim)
    dt = dyn.mjx_model.opt.timestep

    xt = jnp.zeros(dyn.state_dim)
    U = jnp.zeros((horizon, dyn.control_dim))

    ## setting the properties of the tendon (s)
    dyn.mjx_model = dyn.mjx_model.replace(
        tendon_stiffness = dyn.mjx_model.tendon_stiffness.at[:].set(stiffness),
        tendon_damping = dyn.mjx_model.tendon_damping.at[:].set(damping)
    )

    '''
    Quick test
    '''
    # horizon = 500
    # U = jnp.zeros((horizon, dyn.control_dim))
    # X = unroll(dyn, xt, U)

    # print(X)
    # render_cart_pole(dyn, X, path = 'test.mp4')

    # print(dyn.mjx_model.tendon_stiffness)



    # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power * horizon, alpha, observation_noise)
    i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power * horizon, alpha, observation_noise)
    print(e)


    # '''
    # Run MPC
    # '''
    # ## slight force pushing the cart right
    # U = U.at[:, 2].set(right_force)

    # print(f'Timestep = {dt}')
    # print(f'Horizon = {horizon}')

    # ## initial state of pendula (all zeros)
    # xt = dyn.init_state()
    
    # X = jnp.zeros((steps+1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # iterations = jnp.zeros(steps)
    # empowerment = jnp.zeros((steps, 2))
    
    # assert steps == empowerment.shape[0]
    
    # for t in range(steps):
    #     key, sub_key = jax.random.split(key)

    #     # ## probing power is proportional to the instant power times the horizon
    #     # grad_E = compute_multiagent_empowerment_grad(dyn, xt, U, power * horizon, alpha, observation_noise)
    #     # i, e, _ = compute_multiagent_empowerment(dyn, xt, U, power * horizon, alpha, observation_noise)

    #     ## log the number of IWF iterations and empowerment
    #     # iterations = iterations.at[t].set(i)
    #     # empowerment = empowerment.at[t].set(e)

    #     # if t == 0:
    #     #     ## choose a random action on the first step
    #     #     # random_signs = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(dyn.control_dim,))
    #     #     random_signs = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(2,))
    #     #     ut = power * random_signs
    #     # else:
    #     #     _, B = dyn.linearize(xt, U[0])
    #     #     sensitivity = grad_E @ B
    #     #     ut = jnp.sign(jnp.stack([sensitivity[0,0], sensitivity[1, 1]])) * power

    #     ut = jnp.zeros(2)
        
    #     ## add the passive action (constant force)
    #     ut = jnp.concatenate([ut, jnp.array([right_force])])

    #     ## step the dynamics and record the result
    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)

    #     ## print out some relevant quantities
    #     # print(t, xt, ut, e, i)
    #     print(t, xt, ut)

    # run_name = f'full_state-seed={seed}_force={right_force}-stiffness={stiffness}-damping={damping}-horizon={horizon}_power={power}_alpha={alpha}_noise={observation_noise}'

    # ## render animation
    # render_cart_pole(dyn, X, path = run_name + '.mp4')

    # ## plot stats
    # fig, ax = plt.subplots(3, 1)
    # ## plot empowerment
    # ax[0].plot(empowerment[:, 0], label = 'Left Agent', color = 'blue')
    # ax[0].plot(empowerment[:, 1], label = 'Right Agent', color = 'orange')
    # ax[0].set_ylabel('Empowerment\n(nats)')
    # ax[0].tick_params(axis = 'both', labelsize = 12)
    # ax[0].legend(fontsize = 12)
    # ax[0].set_xticks([])
    # ax[0].set_xlim(0, steps)

    # ## plot angle from top
    # agent_0_angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    # agent_1_angle = jnp.abs(smooth_angle_wrap(X[:, 3] - jnp.pi))
    # ax[1].plot(agent_0_angle, color = 'blue')
    # ax[1].plot(agent_1_angle, color = 'orange')
    # ax[1].set_ylabel('Angle From Top\n(rads)')
    # ax[1].tick_params(axis = 'both', labelsize = 12)
    # ax[1].set_xlim(0, steps)
    # ax[1].set_xticks([])

    # n_ticks = 5
    # positions = np.linspace(0, steps - 1, n_ticks)
    # labels = np.linspace(0.0, steps * dt, n_ticks)

    # ax[2].set_xticks(positions)
    # ax[2].set_xticklabels(labels, rotation = 'horizontal')
    # ax[2].set_xlabel('Interaction Time (s)', fontsize = 14)
    # ax[2].set_ylabel('Iterations')
    # ax[2].plot(iterations)

    # fig.tight_layout()
    # fig.savefig(run_name + '.png', dpi = 300)
    # plt.show()