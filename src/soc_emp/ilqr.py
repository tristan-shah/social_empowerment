from typing import Optional

import jax
from jax import Array
from jax import numpy as jnp

from soc_emp import Dynamics
from soc_emp.utils import smooth_angle_wrap

class TrajectoryOptimizer:
    '''
    Optimizes a sequence of actions in a state x0.
    '''
    def __init__(
            self, 
            dyn: Dynamics, 
            max_power: float | Array, 
            compute_error: callable,
            Q: Array = None, 
            R: Array = None):

        if type(max_power) == float:
            max_power = jnp.array([max_power])

        self.dyn = dyn
        self.max_power = max_power

        self.compute_error = compute_error

        ## weighting matrices for states (Q) and control (R)
        dx = dyn.state_dim
        du = dyn.control_dim

        if Q is None:
            self.Q = jnp.eye(dx)
        else:
            self.Q = Q

        if R is None:
            self.R = jnp.eye(du)
        else:
            self.R = R

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

                delta_x = jnp.stack([ ## specific to single pendulum
                    smooth_angle_wrap(X_new[t, 0] - X[t, 0]),
                    (X_new[t, 1] - X[t, 1])
                    ])
                # delta_x = X_new[t] - X[t]
                ut = k[t] + K[t] @ delta_x

            ut = ut.clip(-self.max_power, self.max_power)

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

        L = jnp.copy(Q) #jnp.zeros((dx, dx))
        M = jnp.zeros((dx, dx))
        
        k = jnp.zeros((T, du))
        K = jnp.zeros((T, du, dx))
        E = Q @ r[-1]

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
    
    def update(self, x0, X, U, A, B):
        r = self.compute_error(X) ## compute error of previous nominal trajectory
        k, K = self.backward(r, U, A, B)
        X_new, U_new = self.forward(x0, X = X, k = k, K = K)
        A, B = self.batch_linearize(X_new[:-1], U_new)
        J_new = jnp.sum(jnp.square(r) @ self.Q)
        return J_new, X_new, U_new, A, B
    