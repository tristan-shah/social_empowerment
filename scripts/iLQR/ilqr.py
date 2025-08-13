import jax
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import Array
from einops import einsum
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import numpy as np

from soc_emp import Dynamics
from soc_emp.empowerment import unroll
from soc_emp.utils import split_state, smooth_angle_wrap

def sub_quat(q: Array, q_bar: Array):

    quat = Rotation.from_quat(
        jnp.roll(q, shift = -1, axis = 0)
    )
    quat_bar = Rotation.from_quat(
        jnp.roll(q_bar, shift = -1, axis = 0)
    )

    delta_theta = (quat * quat_bar.inv()).as_rotvec()
    ## small perturbation ?
    # delta_quat = jnp.concatenate([jnp.array([1.0]), 0.5 * delta_theta])

    ## large perturbation ?
    delta_quat = jnp.roll(Rotation.from_rotvec(delta_theta).as_quat(), shift = 1, axis = 0)
    return delta_quat

def compute_delta_x(m: mjx.Model, xt: Array, xt_bar: Array):

    ## initialize a vector to hold the difference: xt - xt_bar
    delta_x = jnp.zeros_like(xt)

    ## separate position from velocity. position requires special handling
    qpos_xt, qvel_xt = split_state(xt, m.nq)
    qpos_xt_bar, qvel_xt_bar = split_state(xt_bar, m.nq)

    for jnt_type, jnt_qposadr in zip(m.jnt_type, m.jnt_qposadr):

        if jnt_type == mjx.JointType.FREE:
            ## extract cartesian coordinates
            xyz = qpos_xt[jnt_qposadr:jnt_qposadr+3]
            xyz_bar = qpos_xt_bar[jnt_qposadr:jnt_qposadr+3]

            ## extract quaternions
            quat = qpos_xt[jnt_qposadr+3:jnt_qposadr+7]
            quat_bar = qpos_xt_bar[jnt_qposadr+3:jnt_qposadr+7]
            
            ## assemble difference vector
            diff = jnp.concatenate([xyz - xyz_bar, sub_quat(quat, quat_bar)])
            delta_x = delta_x.at[jnt_qposadr:jnt_qposadr+7].set(diff)

        elif jnt_type == mjx.JointType.SLIDE:
            ## perform standard subtraction for SLIDE or HINGE joints
            delta_x = delta_x.at[jnt_qposadr].set( qpos_xt[jnt_qposadr] - qpos_xt_bar[jnt_qposadr] )
        
        elif jnt_type == mjx.JointType.HINGE:
            ## wrap angles
            delta_x = delta_x.at[jnt_qposadr].set( smooth_angle_wrap(qpos_xt[jnt_qposadr] - qpos_xt_bar[jnt_qposadr]) )

        elif jnt_type == mjx.JointType.BALL:
            ## perform quaternion subtraction for BALL joints
            quat = qpos_xt[jnt_qposadr: jnt_qposadr + 4]
            quat_bar = qpos_xt_bar[jnt_qposadr: jnt_qposadr + 4]
            diff = sub_quat(quat, quat_bar)
            delta_x = delta_x.at[jnt_qposadr: jnt_qposadr + 4].set(diff)
        else:
            raise ValueError()

    ## linear subtraction for velocities
    delta_x = delta_x.at[m.nq:].set(qvel_xt - qvel_xt_bar)
    return delta_x

class iLQR:
    def __init__(self, dyn: Dynamics, Q: Array, R: Array):
        self.dyn = dyn
        self.Q = Q
        self.R = R
        self.batch_linearize = jax.vmap(dyn.linearize)
        self.low = self.dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = self.dyn.mjx_model.actuator_ctrlrange[:, 1]

    def forward(self, X: Array, U: Array, k: Array, K: Array, alpha: float):

        low = self.low
        high = self.high
        T = k.shape[0]

        xt = X[0]
        X_new = jnp.zeros_like(X)
        X_new = X_new.at[0].set(xt)
        U_new = jnp.zeros_like(k)

        for t in range(T):

            ## compute deviation in current trajectory
            delta_x = X_new[t] - X[t]

            ## compute new control
            ut = U[t] + alpha * k[t] + K[t] @ delta_x
            ut = ut.clip(low, high)

            ## propagate dynamics with new control
            xt = self.dyn.step(xt, ut)

            X_new = X_new.at[t+1].set(xt)
            U_new = U_new.at[t].set(ut)

        return X_new, U_new
        
    def backward(self, e: Array, A: Array, B: Array, U: Array):

        Q = self.Q
        R = self.R

        T = A.shape[0]
        dx = A.shape[2]
        du = B.shape[2]

        E = Q @ e[-1]
        L = Q.copy()
        M = jnp.zeros_like(Q)

        U_tilde = jnp.zeros((T, du))
        k = jnp.zeros((T, du))
        K = jnp.zeros((T, du, dx))

        for t in reversed(range(e.shape[0])):
            P = L + M
            ## inverse term
            S = jnp.linalg.inv(R + B[t].T @ P @ B[t] + jnp.eye(du) * 1e-5)
            ## gradient of policy
            grad_pi = - S @ B[t].T @ P @ A[t]
            ## compute gains
            U_tilde = U_tilde.at[t].set(S @ B[t].T @ P @ B[t] @ U[t])
            k = k.at[t].set( - S @ B[t].T @ E ) ## feedforward
            K = K.at[t].set(grad_pi) ## feedback
            ## total derivative of dynamics
            total_grad = A[t] + B[t] @ grad_pi

            ## updates
            E = total_grad.T @ E + (Q @ e[t] + grad_pi.T @ R @ U[t])
            L = total_grad.T @ L @ total_grad + Q
            M = total_grad.T @ M @ total_grad + grad_pi.T @ R @ grad_pi

        # return k, K
        return U_tilde, k, K
    
def render_go(dyn: Dynamics, X: Array, path: str):
    
    dyn.render(
        X,
        path = path,
        skip = 20, 
        distance = 2.0, 
        elevation = -20,
        lookat = jnp.array([0.8, 0.0, 0.5])
        )
    
    return None
        
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    # integrator = 'implicitfast'
    integrator = 'euler'

    # dyn = Dynamics(path = 'xml/franka_emika_panda/mjx_single_cube.xml', integrator = integrator, dt = 0.005)
    dyn = Dynamics(path = 'xml/unitree_go2/scene_mjx.xml', integrator = integrator, dt = 0.0005)
    # dyn = Dynamics(path = 'xml/unitree_g1/scene_mjx.xml', integrator = integrator, dt = 0.001)
    # dyn = Dynamics(path = 'xml/custom/ball.xml', integrator = integrator)

    print(f'State dimention: {dyn.state_dim}')
    print(f'Control dimention: {dyn.control_dim}')
    print(f'Timestep: {dyn.model.opt.timestep}')
    print(f'Friction Cone: {mjx.ConeType(dyn.model.opt.cone).name}')
    print(f'Impratio: {dyn.model.opt.impratio}')
    print()

    batch_linearize = jax.vmap(dyn.linearize)

    xt = dyn.init_state()
    home = dyn.model.keyframe('home')
    xt = xt.at[:dyn.nq].set(home.qpos)

    print(xt)

    for jnt_type, jnt_qposadr in zip(dyn.mjx_model.jnt_type, dyn.mjx_model.jnt_qposadr):
        print(mjx.JointType(jnt_type), jnt_qposadr)

    # steps = 1000
    steps = 3000

    '''for ball'''
    # ## building goal state
    # goal_qpos = jnp.array(home.qpos)
    # goal_qpos = jnp.zeros(dyn.nq)
    # goal_qpos = goal_qpos.at[0].set(1.2) ## x
    # goal_qpos = goal_qpos.at[1].set(-1.0) ## y
    # goal_qpos = goal_qpos.at[2].set(0.1) ## z
    # goal_qvel = jnp.zeros(dyn.nv)
    # goal = jnp.concatenate([goal_qpos, goal_qvel])

    # ## building Q R matrices
    # Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    # Q = Q.at[0, 0].set(1.0)
    # Q = Q.at[1, 1].set(1.0)
    # Q = Q.at[2, 2].set(1.0)
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.001) ## velocity penalty
    # # R = jnp.eye(dyn.control_dim) * 0.001
    # # R = jnp.eye(dyn.control_dim) * 0.1
    # R = jnp.eye(dyn.control_dim) * 1.0
    # U = jnp.zeros((steps, dyn.control_dim))

    '''for go2'''
    ## building goal state
    goal_qpos = jnp.array(home.qpos)
    goal_qpos = goal_qpos.at[0].set(1.0) ## x
    # goal_qpos = goal_qpos.at[1].set(0.5) ## x
    goal_qvel = jnp.zeros(dyn.nv)
    goal = jnp.concatenate([goal_qpos, goal_qvel])
    ## building Q R matrices
    Q = jnp.zeros((dyn.state_dim, dyn.state_dim))

    ## original Q weights
    # Q = Q.at[0:7,0:7].set(jnp.eye(7)) ## main body
    # Q = Q.at[7:dyn.nq, 7:dyn.nq].set(jnp.eye(dyn.nq - 7) * 0.5) ## all other parts
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.0001) ## velocity penalty
    ## original R weights
    # R = jnp.eye(dyn.control_dim) * 0.0001

    ## new Q weights

    ## main body
    Q = Q.at[0:7,0:7].set(jnp.eye(7))
    ## extremities
    Q = Q.at[7:dyn.nq, 7:dyn.nq].set(jnp.eye(dyn.nq - 7) * 0.5) ## original
    # Q = Q.at[7:dyn.nq, 7:dyn.nq].set(jnp.eye(dyn.nq - 7) * 0.01)
    ## velocity
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.0001)
    Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.00001) ## original
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.0000001)

    ## new R weights
    # R = jnp.eye(dyn.control_dim) * 0.1
    # R = jnp.eye(dyn.control_dim) * 0.001 ## original
    R = jnp.eye(dyn.control_dim) * 1.0
    U = jnp.tile(home.ctrl[None, :], (steps, 1))

    '''panda'''
    # ## building goal state
    # # goal_qpos = jnp.zeros(dyn.nq)
    # goal_qpos = jnp.array(home.qpos)
    # goal_qvel = jnp.zeros(dyn.nv)
    # goal = jnp.concatenate([goal_qpos, goal_qvel])

    # ## building Q R matrices
    # Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    # Q = Q.at[:dyn.nq, :dyn.nq].set(jnp.eye(dyn.nq))
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.0001) ## velocity penalty

    # R = jnp.eye(dyn.control_dim) * 0.1
    # U = jnp.zeros((steps, dyn.control_dim))

    '''
    run iLQR
    '''
    ilqr = iLQR(dyn, Q, R)
    X = unroll(dyn, xt, U)

    # dyn.render(
    #     X,
    #     path = 'left.mp4',
    #     skip = 20, 
    #     distance = 2.0, 
    #     elevation = -20,
    #     lookat = jnp.array([0.8, 0.0, 0.5])
    #     )

    alphas = jnp.linspace(1e-4, 1.0, 20)

    batch_forward = jax.vmap(ilqr.forward, in_axes = (None, None, None, None, 0))

    cost = []

    # iterations = 10
    iterations = 3
    for i in range(iterations):

        A, B = ilqr.batch_linearize(X[:-1], U)

        e = (X - goal)
        # k, K = ilqr.backward(e, A, B, U)
        U_tilde, k, K = ilqr.backward(e, A, B, U)

        # X_batch, U_batch = batch_forward(X, U, k, K, alphas)
        X_batch, U_batch = batch_forward(X, U_tilde, k, K, alphas)

        e_batch = X_batch - goal
        J_batch = einsum(e_batch, Q, e_batch, 'n t x1, x1 x2, n t x2 -> n') + einsum(U_batch, R, U_batch, 'n t u1, u1 u2, n t u2 -> n')

        idx = jnp.argmin(J_batch)
        X = X_batch[idx]
        U = U_batch[idx]
        J = J_batch[idx]
        print(i, J, alphas[idx])

        cost.append(J)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylabel('Trajectory Cost')
    ax[0].set_xlabel('Iterations')
    ax[0].plot(cost)

    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel('Control')
    ## plot each control signal
    for i in range(U.shape[1]):
        ax[1].plot(U[:, i], alpha = 0.2)

    # fig.savefig('ball_cost.png', dpi = 300)
    fig.savefig('100_imp_left_go_cost.png', dpi = 300)

    # dyn.render(X, path = 'ball.mp4', skip = 10, elevation = -90)
    dyn.render(
        X,
        path = '100_imp_left_go.mp4',
        skip = 20, 
        distance = 2.0, 
        elevation = -20,
        lookat = jnp.array([0.8, 0.0, 0.5])
        )