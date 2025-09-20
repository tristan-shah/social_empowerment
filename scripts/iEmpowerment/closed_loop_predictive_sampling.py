import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum
import matplotlib.pyplot as plt

from soc_emp.utils import smooth_angle_wrap, cubic_spline_interp
from soc_emp import Dynamics
from soc_emp.empowerment import unroll

@jax.jit
def compute_pendulum_error(X: Array):
    assert X.shape[1] == 2

    r = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        X[:, 1]
    ], axis = 1)

    assert r.shape == X.shape
    return r

class PredictiveSampling:
    def __init__(
            self, 
            dyn: Dynamics, 
            compute_error: callable,
            horizon: int,
            N: int,
            knots: int,
            sigma: float,
            Q: Array = None, 
            R: Array = None):
        
        self.dyn = dyn
        self.compute_error = compute_error
        self.N = N
        self.knots = knots
        self.sigma = sigma
        
        self.horizon = jnp.arange(0, horizon)
        self.knots_idx = jnp.linspace(0, len(self.horizon)-1, knots, dtype = int)
        self.horizon_knots = self.horizon[self.knots_idx]

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        ## cost weighting matrices
        self.Q = Q
        self.R = R

    def __call__(self, xt: Array, U: Array, key):

        ## extract knots from current control sequence
        U_knots = U[self.knots_idx, :]
        ## generate exploration noise
        eps = jax.random.normal(key, (N, knots, U.shape[1]))
        ## reparameterization trick
        U_knots_dist = (U_knots + eps * sigma).clip(self.low, self.high)
        ## dense evaluation along horizon
        U_dist = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(self.horizon_knots, U_knots_dist, self.horizon).clip(self.low, self.high)
        ## paralel simulation
        X_dist = jax.vmap(unroll, in_axes = (None, None, 0))(self.dyn, xt, U_dist)
        ## evaluate error of each trajectory
        e = jax.vmap(self.compute_error, in_axes = 0)(X_dist)
        ## evaluate cost of batch and determine the best index
        J = einsum(e, self.Q, e, 'b t x1, x1 x2, b t x2 -> b')
        best_idx = jnp.argmin(J)

        X = X_dist[best_idx]
        U = U_dist[best_idx]

        return X, U, J[best_idx]
    
if __name__ == '__main__':

    key = jax.random.PRNGKey(0)

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)
    Q = jnp.eye(dyn.state_dim)
    Q = Q.at[1, 1].set(0.01)
    R = jnp.eye(dyn.control_dim)

    x0 = jnp.zeros(dyn.state_dim)

    horizon = 100
    N = 1000
    knots = 10
    sigma = 0.1

    opt = PredictiveSampling(dyn, compute_pendulum_error, horizon, N, knots, sigma, Q, R)

    U = jnp.zeros((horizon, dyn.control_dim))

    fig, ax = plt.subplots(1, 2)

    cost_hist = []

    for i in range(30):
        ax[0].plot(U, color = 'blue', alpha = 0.5)
        X, U, J = opt(x0, U, key)

        cost_hist.append(J)

    ax[0].plot(U, color = 'red')
    ax[1].plot(cost_hist)
    plt.show()

    steps = 600

    xt = x0.copy()

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)


    cost_hist = []
    for t in range(steps):
        _, U, J = opt(xt, U, key)
        ut = U[0]

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)
        cost_hist.append(J)

        print(xt, J)

        ## shift over the plan by one action
        U = jnp.roll(U, shift = -1, axis = 0)
        U = U.at[-1].set(U[-2])


    fig, ax = plt.subplots(1, 1)
    ax.set_title('Horizon Cost')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cost')
    ax.plot(cost_hist)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Trajectory')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta Dot')
    ax.plot(X[:, 0], X[:, 1])
    ax.scatter(X[0, 0], X[0, 1], color = 'green', label = 'Start')
    ax.scatter(X[-1, 0], X[-1, 1], color = 'red', label = 'End')
    ax.legend()
    plt.show()

    dyn.render(X, path = 'test.mp4', skip = 2)