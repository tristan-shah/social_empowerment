from typing import Optional

import jax
from jax import Array
from jax import numpy as jnp
import numpy as np
import mujoco

from soc_emp import Dynamics
from soc_emp.utils import smooth_angle_wrap, split_state, diff_qpos

class TrajectoryOptimizer:
    '''
    Optimizes a sequence of actions in a state x0.
    '''
    def __init__(
            self, 
            dyn: Dynamics, 
            compute_error: callable,
            Q: Array = None, 
            R: Array = None):
        
        self.dyn = dyn
        self.compute_error = compute_error
        
        ## extract the control ranges from each motors
        self.ctrl_range = self.dyn.mjx_model.actuator_ctrlrange

        ## cost weighting matrices
        self.Q = Q
        self.R = R

        ## allows the linearization to occur along trajectories of states and controls
        self.batch_linearize = jax.jit(jax.vmap(self.dyn.linearize))

    def forward(
            self,
            x0: Array,
            X: Optional[Array] = None,
            U: Optional[Array] = None,
            k: Optional[Array] = None, 
            K: Optional[Array] = None,
            ):
        
        ## must provide an initial control sequence or feedback gains
        assert U is not None or (X is not None and k is not None and K is not None)

        ## dimentions of time, state, and control
        T = U.shape[0] if U is not None else k.shape[0]
        dx, du = self.dyn.state_dim, self.dyn.control_dim

        ## tensors for holding information
        X_new = jnp.zeros((T + 1, dx))
        U_new = jnp.zeros((T, du))

        ## store the first state in the state history
        X_new = X_new.at[0].set(x0)
        xt = x0.clone()

        ## forward pass of dynamics
        for t in range(T):
            if U is not None:
                ut = U[t]
            else:

                '''
                raw subtraction
                '''
                # delta_x = X_new[t] - X[t]

                '''
                Custom diff
                '''

                X_pos, X_vel = split_state(X[t], self.dyn.nq)
                X_new_pos, X_new_vel = split_state(X_new[t], self.dyn.nq)
                dqpos = diff_qpos(self.dyn.model, X_pos, X_new_pos)
                delta_x = jnp.concatenate([dqpos, X_new_vel - X_vel])

                '''
                mujoco delta_x
                '''
                # d_pos = np.zeros(self.dyn.nv)
                # X_pos, X_vel = split_state(X[t], self.dyn.nq)
                # X_new_pos, X_new_vel = split_state(X_new[t], self.dyn.nq)
                # mujoco.mj_differentiatePos(self.dyn.model, d_pos, 1.0, X_pos, X_new_pos)
                # delta_x = jnp.concatenate([d_pos, X_new_vel - X_vel])

                '''
                manual delta x
                '''
                # delta_x = jnp.stack([ ## specific to single pendulum
                #     smooth_angle_wrap(X_new[t, 0] - X[t, 0]),
                #     (X_new[t, 1] - X[t, 1])
                #     ])

                # print('hello')
                # print(X_pos.shape, X_vel.shape)
                # print(k[t].shape, K[t].shape)
                # print()

                ut = k[t] + K[t] @ delta_x

            ut = ut.clip(self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            ## step dynamics
            xt = self.dyn.step(xt, ut)
            ## update sequences
            X_new = X_new.at[t+1].set(xt)
            U_new = U_new.at[t].set(ut)

        return X_new, U_new
        
    def backward(
            self,
            r: Array,
            U: Array, 
            A: Array, 
            B: Array):
         
        T = U.shape[0]
        dx, du = self.dyn.state_dim, self.dyn.control_dim

        Q = self.Q
        R = self.R

        L = jnp.copy(Q)
        M = jnp.zeros((dx, dx))
        E = Q @ r[-1]

        k = jnp.zeros((T, du))
        K = jnp.zeros((T, du, dx))

        for t in reversed(range(T)):
            P = L + M
            ## inverse term
            S = jnp.linalg.inv(R + B[t].T @ P @ B[t])
            ## gradient of policy
            grad_pi = - S @ B[t].T @ P @ A[t]
            ## compute gains
            k = k.at[t].set(S @ B[t].T @ (P @ B[t] @ U[t] - E)) ## feedforward
            K = K.at[t].set(grad_pi) ## feedback
            ## total derivative of dynamics
            total_grad = A[t] + B[t] @ grad_pi
            ## update L history
            L = total_grad.T @ L @ total_grad + Q
            ## update M history
            M = total_grad.T @ M @ total_grad + grad_pi.T @ R @ grad_pi
            ## update error term
            E = total_grad.T @ E + (Q @ r[t] + grad_pi.T @ R @ U[t])

        return k, K
    
    def update(self, x0: Array, X: Array, U: Array, A: Array, B: Array):
        ## compute error of previous nominal trajectory
        r = self.compute_error(X)
        k, K = self.backward(r, U, A, B)
        X_new, U_new = self.forward(x0, X = X, k = k, K = K)
        A, B = self.batch_linearize(X_new[:-1], U_new)
        J_new = jnp.sum(jnp.square(r) @ self.Q)

        return J_new, X_new, U_new, A, B
    
    def __call__(
            self,
            x0: Array,
            U: Array, 
            max_iter: int = 30,
            tol: float = 0.1,
            use_all_iterations: bool = False):

        i = 0
        J = jnp.inf
        converged = False

        ## initial forward rollout
        X, U = self.forward(x0, U = U)
        A, B = self.batch_linearize(X[:-1], U)

        while i < max_iter and (not converged or use_all_iterations):
            i += 1
            # print(i)

            J_new, X_new, U_new, A, B = self.update(x0, X, U, A, B)

            if J_new > J and not use_all_iterations:
                break
            else:
                converged = (J - J_new) < tol
                X = X_new
                U = U_new
                J = J_new

        return X, U, J


class ModelPredictiveController:
    def __init__(self, optimizer: TrajectoryOptimizer, horizon: int):

        key = jax.random.PRNGKey(0)

        self.optimizer = optimizer
        self.horizon = horizon

        ## storing the current plan and action sequence
        self.X = jnp.zeros((horizon+1, optimizer.dyn.state_dim))

        low = self.optimizer.ctrl_range[:, 0]
        high = self.optimizer.ctrl_range[:, 1]

        # self.U = (jax.random.uniform(key, (horizon, optimizer.dyn.control_dim)) * 2 * optimizer.max_power - optimizer.max_power)
        self.U = (jax.random.uniform(key, (horizon, optimizer.dyn.control_dim) ) * (high - low) + low)

    def __call__(self, xt: Array):
        ## update the plan with new state xt
        X, U, J = self.optimizer(xt, self.U)
        
        self.X = X
        self.U = U

        ## select the first action
        ut = self.U[0]

        ## shift over the plan by one action
        self.U = jnp.roll(self.U, shift = -1, axis = 0)
        self.U = self.U.at[-1].set(self.U[-2])
        return ut, J