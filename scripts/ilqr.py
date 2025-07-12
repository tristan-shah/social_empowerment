import jax
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import Array
from einops import einsum
import jax.scipy.spatial
import jax.scipy.spatial.transform
import matplotlib.pyplot as plt
from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import unroll
from soc_emp.utils import split_state, smooth_angle_wrap

def sub_quat(q1: Array, q2: Array):
    '''
    Takes in two scalar-last quaternions and performs: q1 - q2. Returns a scalar first quaternion.
    '''

    ## convert to scalar last
    q1 = jnp.roll(q1, shift=-1, axis=-1)
    q2 = jnp.roll(q2, shift=-1, axis=-1)

    ## construct Rotation object
    q1 = Rotation.from_quat(q1)
    q2 = Rotation.from_quat(q2)

    ## perform subtraction
    q = (q1 * q2.inv()).as_quat(scalar_first = True)
    # return jnp.where(q[0] < 0, -q, q)
    return q

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

    def forward(self, X: Array, k: Array, K: Array):

        low = self.dyn.mjx_model.actuator_ctrlrange[:, 0]
        high = self.dyn.mjx_model.actuator_ctrlrange[:, 1]
        T = k.shape[0]
        
        xt = X[0]
        X_new = jnp.zeros_like(X)
        X_new = X_new.at[0].set(xt)
        U_new = jnp.zeros_like(k)

        for t in range(T):
            delta_x = X_new[t] - X[t]
            # delta_x = compute_delta_x(self.dyn.mjx_model, X_new[t], X[t])

            ut = k[t] + K[t] @ delta_x
            ut = ut.clip(low, high)

            xt = dyn.step(xt, ut)

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

        k = jnp.zeros((T, du))
        K = jnp.zeros((T, du, dx))

        for t in reversed(range(e.shape[0])):
            P = L + M
            ## inverse term
            S = jnp.linalg.inv(R + B[t].T @ P @ B[t] + jnp.eye(du) * 1e-6)
            # S = jnp.linalg.inv(R + B[t].T @ P @ B[t])
            ## gradient of policy
            grad_pi = - S @ B[t].T @ P @ A[t]
            ## compute gains
            k = k.at[t].set( S @ B[t].T @ (P @ B[t] @ U[t] - E) ) ## feedforward
            K = K.at[t].set(grad_pi) ## feedback
            ## total derivative of dynamics
            total_grad = A[t] + B[t] @ grad_pi

            ## updates
            E = total_grad.T @ E + (Q @ e[t] + grad_pi.T @ R @ U[t])
            L = total_grad.T @ L @ total_grad + Q
            M = total_grad.T @ M @ total_grad + grad_pi.T @ R @ grad_pi
        return k, K

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    integrator = 'implicitfast'
    dyn = Dynamics(path = 'xml/unitree_go2/scene_mjx.xml', integrator = integrator, dt = 0.0005)
    # integrator = 'euler'
    # dyn = Dynamics(path = 'xml/unitree_go2/scene_mjx.xml', integrator = integrator, dt = 0.001)
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

    for jnt_type, jnt_qposadr in zip(dyn.mjx_model.jnt_type, dyn.mjx_model.jnt_qposadr):
        print(mjx.JointType(jnt_type), jnt_qposadr)

    # steps = 300
    steps = 1000
    goal_qpos = jnp.array(home.qpos)
    # goal_qpos = goal_qpos.at[0].set(-2.0) ## x
    # goal_qpos = goal_qpos.at[1].set(-1.0) ## y
    
    goal_qpos = goal_qpos.at[0].set(0.1) ## x
    goal_qpos = goal_qpos.at[1].set(0.0) ## y
    goal_qpos = goal_qpos.at[2].set(0.28) ## y
    qoal_qpos = goal_qpos.at[3:7].set([1.0, 0.0, 0.0, 0.0]) ## quaternion

    goal_qvel = jnp.zeros(dyn.nv)
    goal_qvel = goal_qvel.at[0].set(1.0)
    goal = jnp.concatenate([goal_qpos, goal_qvel])

    Q = jnp.zeros((dyn.state_dim, dyn.state_dim))
    Q = Q.at[0, 0].set(1.0)
    Q = Q.at[1, 1].set(1.0)
    Q = Q.at[2, 2].set(1.0)

    '''for ball'''
    # Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.001) ## velocity penalty
    # R = jnp.eye(dyn.control_dim) * 0.00001

    '''for go'''
    # Q = Q.at[2, 2].set(1.0)
    Q = Q.at[dyn.nq:, dyn.nq:].set(jnp.eye(dyn.nv) * 0.00001) ## velocity penalty
    R = jnp.eye(dyn.control_dim) * 0.001
    
    ilqr = iLQR(dyn, Q, R)

    U = jnp.tile(home.ctrl[None, :], (steps, 1))
    # U = jnp.zeros((steps, dyn.control_dim))
    X = unroll(dyn, xt, U)

    cost = []
    
    for i in range(10):

        A, B = ilqr.batch_linearize(X[:-1], U)
        
        e = (X - goal)
        k, K = ilqr.backward(e, A, B, U)
        X, U = ilqr.forward(X, k, K)

        J = einsum(e, Q, e, 't x1, x1 x2, t x2 -> t').sum()
        print(i, J)

        cost.append(J)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(cost)

    for i in range(U.shape[1]):
        ax[1].plot(U[:, i])
    # fig.savefig('ball_cost.png', dpi = 300)
    fig.savefig('go_cost.png', dpi = 300)

    # dyn.render(X, path = 'ball.mp4', skip = 10, elevation = -90)
    dyn.render(X, path = 'go.mp4', skip = 10)