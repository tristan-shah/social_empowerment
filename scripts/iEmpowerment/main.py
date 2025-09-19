import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum, rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm

from soc_emp import Dynamics
from soc_emp.empowerment import unroll, waterfilling_implicit, compute_power, compute_F_from_A_B

@jax.jit
def compute_sigma(F: Array, power: float):
    '''
    computes the covariance matrix of the control
    '''
    _, h, M = jnp.linalg.svd(F, full_matrices = False)
    h2 = (h ** 2).clip(min = 1e-12)
    v = waterfilling_implicit(h2, power)
    p = compute_power(v, h2)
    return M.T @ jnp.diag(p) @ M

@jax.jit
def compute_total_hessian(Hxx: Array, Hxu: Array, Hux: Array, Huu: Array, grad_pi: Array):

    total_hessian = Hxx \
        + einsum(Hxu, grad_pi, 'f x1 u, u x2 -> f x1 x2') \
        + einsum(grad_pi, Hux, 'u x1, f u x2 -> f x1 x2') \
        + einsum(grad_pi, Huu, grad_pi, 'u1 x1, f u1 u2, u2 x2 -> f x1 x2')
    
    return total_hessian

def hessian_propagation(Hxx: Array, Hxu: Array, Hux: Array, Huu: Array, A: Array, B: Array, grad_pi: Array):
    '''
    Takes in sequences of hessians and jacobians computed along a nominal trajectory and the gradient of the policy. 
    Computes hessian propagation which tells the total second order sensitivity of the final state to the initial state.

    Args:
    Hxx: (time x state x state x state)
    Hxu: (time x state x state x control)
    Hux: (time x state x control x state)
    Huu: (time x state x control x control)
    A: (time x state x state)
    B: (time x state x control)
    grad_pi: (time x control x state)

    Returns:
    A tensor representing total second order sensitivity: (state x state x state)
    '''

    dx = Hxx.shape[1]

    H = jnp.zeros((dx, dx, dx))
    J = jnp.eye(dx)

    for t in range(Hxx.shape[0]):
        total_grad = A[t] + B[t] @ grad_pi[t]
        total_hessian = compute_total_hessian(Hxx[t], Hxu[t], Hux[t], Huu[t], grad_pi[t])
        H = einsum(J, total_hessian, J, 'x1 x2, f x1 x3, x3 x4 -> f x2 x4') + einsum(total_grad, H, 'f x1, x1 x2 x3 -> f x2 x3')
        J = total_grad @ J

    return H

class iEmpowerment:
    def __init__(self, dyn: Dynamics):
        self.dyn = dyn
        self.batch_linearize = jax.vmap(dyn.linearize)
        self.low = self.dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = self.dyn.mjx_model.actuator_ctrlrange[:, 1]

    def backward(
            self, 
            Hxx: Array,
            Hxu: Array,
            Hux: Array,
            Huu: Array,
            A: Array, 
            B: Array,
            sigma: Array):
        
        dx = self.dyn.state_dim
        du = self.dyn.control_dim
        horizon = Hxx.shape[0]
        
        ## how much to regularize the inverse term
        damping = jnp.eye(du) * 1e-7
        ## storing the total sensitivity of the final state to an intial perturbation
        J = jnp.eye(dx)
        ## initial value of the riccati equation solution
        P = jnp.zeros((dx, dx))
        ## gradient of empowerment w.r.t state
        E = jnp.zeros(dx)
        ## to store the gradients of the policy
        K = jnp.zeros((horizon, du, dx))
        ## feedforward term
        k = jnp.zeros((horizon, du))

        for t in tqdm(reversed(range(horizon))):
            
            ## compute feedback channel
            G = J @ B[t]
            T = I + G @ sigma[t] @ G.T
            T_inv = jnp.linalg.inv(T)

            ## propagate second order sensitivities to the end of the horizon
            H = hessian_propagation(Hxx[t+1:], Hxu[t+1:], Hux[t+1:], Huu[t+1:], A[t+1:], B[t+1:], K[t+1:])

            ## gradient of feedback channel w.r.t state
            Gx = einsum(B[t], H, A[t], 'x1 u1, f x1 x2, x2 x3 -> f u1 x3') \
               + einsum(J, Hux[t], 'f x1, x1 u1 x2 -> f u1 x2')
            
            ## gradient of feedback channel w.r.t control
            Gu = einsum(B[t], H, B[t], 'x1 u1, f x1 x2, x2 u2 -> f u1 u2') \
               + einsum(J, Huu[t], 'f x1, x1 u1 u2 -> f u1 u2')
            
            ## cross term
            F = T_inv @ einsum(Gu, sigma[t], Gx, 'x1 u1 u2, u1 u3, x2 u3 x3 -> u2 x1 x2 x3')
            F = jnp.trace(F, axis1 = 1, axis2 = 2)

            ## quadratic state term
            V = T_inv @ einsum(Gx, sigma[t], Gx, 'x1 u1 x2, u1 u2, x3 u2 x4 -> x2 x1 x3 x4')
            V = jnp.trace(V, axis1 = 1, axis2 = 2)

            ## quadratic control term
            W = T_inv @ einsum(Gu, sigma[t], Gu, 'x1 u1 u2, u1 u3, x2 u3 u4 -> u2 x1 x2 u4')
            W = jnp.trace(W, axis1 = 1, axis2 = 2)

            ## gradient of empowerment w.r.t state
            v = T_inv @ einsum(G, sigma[t], Gx, 'x1 u1, u1 u2, x2 u2 x3 -> x1 x2 x3')
            v = jnp.trace(v, axis1 = 0, axis2 = 1)

            ## gradient of empowerment w.r.t control
            w = T_inv @ einsum(G, sigma[t], Gu, 'x1 u1, u1 u2, x2 u2 u3 -> x1 x2 u3')
            w = jnp.trace(w, axis1 = 0, axis2 = 1)

            ## compute inverse term once
            S_inv = jnp.linalg.inv(damping + W + B[t].T @ P @ B[t])

            ## gradient of the policy
            K = K.at[t].set( - S_inv @ (B[t].T @ P @ A[t] + F) )
            ## feedforward term
            # k = k.at[t].set( - S_inv @ (w + B[t].T @ E) ) ## minimization
            k = k.at[t].set( S_inv @ (w + B[t].T @ E) ) ## maximization

            ## closed loop jacobian
            total_grad = A[t] + B[t] @ K[t]
            ## step back the gradient of empowerment
            E = v + K[t].T @ w + total_grad.T @ E
            ## propagate closed loop jacobian backwards
            J = J @ total_grad
            ## step riccati equation backwards
            P = V \
                + A[t].T @ P @ A[t] \
                - A[t].T @ P @ B[t] @ S_inv @ B[t].T @ P @ A[t] \
                - A[t].T @ P @ B[t] @ S_inv @ F - F.T @ S_inv @ B[t].T @ P @ A[t] \
                - F.T @ S_inv @ F
            
            # print(t, k[t], K[t])
            
        return k, K
    
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

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    horizon = 600
    power = 1.0
    steps = 1500
    # alpha = 0.5
    alpha = 0.05

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    # dyn = Dynamics(path = xml_path, dt = 0.01, integrator = 'euler')
    dyn = Dynamics(path = xml_path)
    dx = dyn.state_dim
    du = dyn.control_dim

    I = jnp.eye(dyn.state_dim)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.0)

    ## initial exploration action sequence
    U = jnp.zeros((horizon, dyn.control_dim)) + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.1

    ## roll out the nominal trajectory
    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    ''' Fake data '''
    # A = jax.random.normal(key, (horizon, dx, dx))
    # B = jax.random.normal(key, (horizon, dx, du))

    F = rearrange(compute_F_from_A_B(A, B), 't x u -> x (t u)')
    sigma = compute_sigma(F, power)
    sigma = jnp.diag(sigma)
    sigma = sigma[:, None, None]
    ## compute instantanious hessians along the nominal trajectory

    compute_hessians = jax.vmap(jax.jacfwd(jax.jacfwd(dyn.step, argnums = (0, 1)), argnums = (0, 1)))
    (Hxx, Hxu), (Hux, Huu) = compute_hessians(X[:-1], U)

    ''' Fake data '''
    # Hxx = jax.random.normal(key, (horizon, dx, dx, dx))
    # Hxu = jax.random.normal(key, (horizon, dx, dx, du))
    # Hux = jax.random.normal(key, (horizon, dx, du, dx))
    # Huu = jax.random.normal(key, (horizon, dx, du, du))
    print(Hxx.shape, Hxu.shape, Hux.shape, Huu.shape)

    iemp = iEmpowerment(dyn)

    empowerment_hist = []

    for i in range(30):
        
        k, K = iemp.backward(Hxx, Hxu, Hux, Huu, A, B, sigma)
        X, U = iemp.forward(X, U, k, K, alpha)

        A, B = jax.vmap(dyn.linearize)(X[:-1], U)
        ## compute standard empowerment
        F = rearrange(compute_F_from_A_B(A, B), 't x u -> x (t u)')
        sigma = compute_sigma(F, power)

        empowerment = jnp.log(jnp.linalg.det(I + F @ sigma @ F.T))

        empowerment_hist.append(empowerment)

        sigma = jnp.diag(sigma)
        sigma = sigma[:, None, None]

        (Hxx, Hxu), (Hux, Huu) = compute_hessians(X[:-1], U)

        dyn.render(X, path = f'iteration={i}.mp4', skip = 2)

        fig, ax = plt.subplots(1, 5, figsize = (15, 5))

        ax[0].set_title('Feedforward')
        ax[0].set_xlabel('Horizon')
        ax[0].plot(k[:, 0])

        ax[1].set_title('Policy Grad')
        ax[1].set_xlabel('Horizon')
        ax[1].plot(K[:, 0, 0])
        ax[1].plot(K[:, 0, 1])

        ax[2].set_title('Trajectory')
        ax[2].set_xlabel('Theta')
        ax[2].set_ylabel('Theta Dot')
        ax[2].plot(X[:, 0], X[:, 1])
        ax[2].scatter(X[0, 0], X[0, 1], color = 'green', label = 'Start')
        ax[2].scatter(X[-1, 0], X[-1, 1], color = 'red', label = 'End')
        ax[2].legend()

        ax[3].plot(empowerment_hist)
        ax[3].set_title('Empowerment')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel('Empowerment')
        
        ax[4].plot(U)
        ax[4].set_title('Control')
        ax[4].set_xlabel('Horizon')

        fig.tight_layout()
        fig.savefig(f'iteration={i}.png', dpi = 300)
        # plt.show()


    # ## storing the total sensitivity of the final state to an intial perturbation
    # J = jnp.eye(dyn.state_dim)
    # ## initial value of the riccati equation solution
    # P = jnp.zeros((dyn.state_dim, dyn.state_dim))
    # ## how much to regularize the inverse term
    # damping = jnp.eye(dyn.control_dim) * 1e-7

    # ## gradient of empowerment w.r.t state
    # E = jnp.zeros(dyn.state_dim)


    # E_hist = []
    # P_hist = []

    # ## to store the gradients of the policy
    # K = jnp.zeros((horizon, dyn.control_dim, dyn.state_dim))
    # k = jnp.zeros((horizon, dyn.control_dim))

    # for t in reversed(range(horizon)):
        
    #     ## compute feedback channel
    #     G = J @ B[t]
    #     T = I + G @ sigma[t] @ G.T
    #     T_inv = jnp.linalg.inv(T)

    #     ## propagate second order sensitivities to the end of the horizon
    #     H = hessian_propagation(Hxx[t+1:], Hxu[t+1:], Hux[t+1:], Huu[t+1:], A[t+1:], B[t+1:], K[t+1:])

    #     ## gradient of feedback channel w.r.t state
    #     Gx = einsum(B[t], H, A[t], 'x1 u1, f x1 x2, x2 x3 -> f u1 x3') \
    #        + einsum(J, Hux[t], 'f x1, x1 u1 x2 -> f u1 x2')
        
    #     ## gradient of feedback channel w.r.t control
    #     Gu = einsum(B[t], H, B[t], 'x1 u1, f x1 x2, x2 u2 -> f u1 u2') \
    #        + einsum(J, Huu[t], 'f x1, x1 u1 u2 -> f u1 u2')
        
    #     ## cross term
    #     F = T_inv @ einsum(Gu, sigma[t], Gx, 'x1 u1 u2, u1 u3, x2 u3 x3 -> u2 x1 x2 x3')
    #     F = jnp.trace(F, axis1 = 1, axis2 = 2)

    #     ## quadratic state term
    #     V = T_inv @ einsum(Gx, sigma[t], Gx, 'x1 u1 x2, u1 u2, x3 u2 x4 -> x2 x1 x3 x4')
    #     V = jnp.trace(V, axis1 = 1, axis2 = 2)

    #     ## quadratic control term
    #     W = T_inv @ einsum(Gu, sigma[t], Gu, 'x1 u1 u2, u1 u3, x2 u3 u4 -> u2 x1 x2 u4')
    #     W = jnp.trace(W, axis1 = 1, axis2 = 2)

    #     ## gradient of empowerment w.r.t state
    #     v = T_inv @ einsum(G, sigma[t], Gx, 'x1 u1, u1 u2, x2 u2 x3 -> x1 x2 x3')
    #     v = jnp.trace(v, axis1 = 0, axis2 = 1)

    #     ## gradient of empowerment w.r.t control
    #     w = T_inv @ einsum(G, sigma[t], Gu, 'x1 u1, u1 u2, x2 u2 u3 -> x1 x2 u3')
    #     w = jnp.trace(w, axis1 = 0, axis2 = 1)

    #     ## compute inverse term once
    #     S_inv = jnp.linalg.inv(damping + W + B[t].T @ P @ B[t])

    #     ## gradient of the policy
    #     K = K.at[t].set(- S_inv @ (B[t].T @ P @ A[t] + F) )
    #     k = k.at[t].set( - S_inv @ (w + B[t].T @ E) )

    #     ## closed loop jacobian
    #     total_grad = A[t] + B[t] @ K[t]

    #     ## step back the gradient of empowerment
    #     E = v + K[t].T @ w + total_grad.T @ E

    #     ## propagate closed loop jacobian backwards
    #     J = J @ total_grad

    #     ## step riccati equation backwards
    #     P = V \
    #         + A[t].T @ P @ A[t] \
    #         - A[t].T @ P @ B[t] @ S_inv @ B[t].T @ P @ A[t] \
    #         - A[t].T @ P @ B[t] @ S_inv @ F - F.T @ S_inv @ B[t].T @ P @ A[t] \
    #         - F.T @ S_inv @ F
        
    #     print(t, k[t], S_inv)

    #     E_hist.append(E)
    #     P_hist.append(P)
        
    # E_hist = jnp.flip(jnp.stack(E_hist), axis = 0)
    # P_hist = jnp.flip(jnp.stack(P_hist), axis = 0)

    # fig, ax = plt.subplots(1, 4)
    # fig.suptitle(f'Initial Theta: {xt[0]}')
    # ax[0].set_title('Policy Gradient')
    # ax[0].set_xlabel('Horizon')
    # ax[0].plot(K[:, 0, 0])
    # ax[0].plot(K[:, 0, 1])

    # ax[1].set_title('Empowerment Gradient')
    # ax[1].set_xlabel('Horizon')
    # ax[1].plot(E_hist[:, 0])
    # ax[1].plot(E_hist[:, 1])

    # ax[2].set_title('Riccati Solution')
    # ax[2].set_xlabel('Horizon')
    # ax[2].plot(P_hist[:, 0, 0])
    # ax[2].plot(P_hist[:, 0, 1])
    # ax[2].plot(P_hist[:, 1, 0])
    # ax[2].plot(P_hist[:, 1, 1])

    # ax[3].set_title('Feedforward')
    # ax[3].plot(k[:, 0])

    # fig.tight_layout()
    # plt.show()