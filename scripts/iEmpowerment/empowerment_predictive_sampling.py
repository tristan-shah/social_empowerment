import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum, rearrange
import matplotlib.pyplot as plt

from soc_emp.utils import smooth_angle_wrap, cubic_spline_interp, zero_order_interp, linear_interp
from soc_emp import Dynamics
from soc_emp.empowerment import unroll, compute_F_from_A_B, waterfilling_implicit, compute_power

class EmpowermentPredictiveSampling:
    def __init__(
            self, 
            dyn: Dynamics, 
            horizon: int,
            N: int,
            knots: int,
            sigma: float,
            power: float):
        
        self.dyn = dyn
        self.N = N
        self.knots = knots
        self.sigma = sigma
        self.power = power
        
        self.horizon = jnp.arange(0, horizon)
        self.knots_idx = jnp.linspace(0, len(self.horizon)-1, knots, dtype = int)
        self.horizon_knots = self.horizon[self.knots_idx]

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    def __call__(self, xt: Array, U: Array, key):

        ## extract knots from current control sequence
        U_knots = U[self.knots_idx, :]
        ## generate exploration noise
        eps = jax.random.normal(key, (N, knots, U.shape[1]))
        ## reparameterization trick
        U_knots_dist = (U_knots + eps * sigma).clip(self.low, self.high)

        U_knots_dist = jnp.concatenate([
            U_knots[None, :, :], 
            U_knots_dist
        ], axis = 0)
        
        ## dense evaluation along horizon
        # U_dist = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(self.horizon_knots, U_knots_dist, self.horizon).clip(self.low, self.high)
        # U_dist = jax.vmap(zero_order_interp, in_axes = (None, 0, None))(self.horizon_knots, U_knots_dist, self.horizon).clip(self.low, self.high)
        U_dist = jax.vmap(linear_interp, in_axes = (None, 0, None))(self.horizon_knots, U_knots_dist, self.horizon).clip(self.low, self.high)

        ## paralel simulation
        X_dist = jax.vmap(unroll, in_axes = (None, None, 0))(self.dyn, xt, U_dist)

        ## paralel computation of jacobians along time and batch dimention
        A, B = jax.vmap(jax.vmap(self.dyn.linearize))(X_dist[:, :-1, :], U_dist)
        F = jax.vmap(compute_F_from_A_B)(A, B)
        F = rearrange(F, 'b t x u -> b x (t u)')

        ## batch of covariance matrices
        S = einsum(F, F, 'b x1 t, b x2 t -> b x1 x2')
        h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
        v = jax.vmap(waterfilling_implicit, in_axes = (0, None))(h2, power)
        p = jax.vmap(compute_power)(v, h2)
        e = 0.5 * jnp.sum(jnp.log(1 + p * h2), axis = 1)
        best_idx = jnp.argmax(e)

        X = X_dist[best_idx]
        U = U_dist[best_idx]

        return X, U, e[best_idx]
    
def render_go(dyn: Dynamics, X: Array, path: str):
    
    dyn.render(
        X,
        path = path,
        skip = 20, 
        distance = 2.0, 
        elevation = -20,
        lookat = jnp.array([0.0, 0.0, 0.5])
        )
    
    return None

'''
works for single pendulum
'''

# horizon = 50
# N = 1000
# knots = 25
# sigma = 0.2
# power = 1.0
# steps = 1500
# dt = 0.01

# horizon = 100
# N = 2000
# knots = 50
# sigma = 0.1
# power = 1.0
# steps = 600

# horizon = 150
# N = 2000
# knots = 75
# sigma = 0.1
# power = 1.0
# steps = 600
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    horizon = 500
    N = 10000
    knots = 250
    sigma = 0.1
    power = 1.0
    steps = 3000

    ## load in xml
    # xml_path = 'xml/custom/pendulum.xml'
    xml_path = 'xml/custom/double_pendulum.xml'
    # xml_path = 'xml/custom/cart_pole.xml'
    # xml_path = 'xml/unitree_go2/scene_mjx.xml'

    dyn = Dynamics(path = xml_path, integrator = 'euler', dt = 0.01)
    # dyn = Dynamics(path = xml_path, dt = dt)
    dt = dyn.mjx_model.opt.timestep

    # name = f'single_pendulum_interp=linear_horizon={horizon}_dt={dt}'
    # name = f'weak_cart_pole_interp=linear_horizon={horizon}_dt={dt}'
    name = f'double_pendulum_interp=linear_N={N}_horizon={horizon}_dt={dt}'
    # name = f'go2_interp=linear_horizon={horizon}_dt={dt}'

    output_dir = Path(name)
    output_dir.mkdir(parents = True, exist_ok = True)

    x0 = jnp.zeros(dyn.state_dim)

    opt = EmpowermentPredictiveSampling(dyn, horizon, N, knots, sigma, power)

    U = jnp.zeros((horizon, dyn.control_dim))

    if xml_path == 'xml/custom/cart_pole.xml':
        x0 = x0.at[1].set(jnp.pi)
        # x0 = x0.at[1].set(jnp.pi + 3.1) ## test near the top

        X = unroll(dyn, x0, U)
        dyn.render(X, path = 'cart_pole.mp4', distance = 5, skip = 2, lookat = jnp.array([0, 0, 0]))

    if xml_path == 'xml/unitree_go2/scene_mjx.xml':
        home = dyn.model.keyframe('home')
        x0 = x0.at[:dyn.nq].set(home.qpos)
        
        X = unroll(dyn, x0, U)
        render_go(dyn, X, path = 'go.mp4')

    '''
    Warmstart
    '''
    fig, ax = plt.subplots(1, 2)

    cost_hist = []

    for i in range(100):
        key, sub_key = jax.random.split(key)
        ax[0].plot(U, color = 'blue', alpha = 0.5)
        X, U, J = opt(x0, U, key)
        print(i, J)
        cost_hist.append(J)

    ax[0].plot(U, color = 'red')
    ax[1].plot(cost_hist)
    fig.savefig(name + '/warmstart.png')

    xt = x0.copy()
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    cost_hist = []
    for t in range(steps):

        key, sub_key = jax.random.split(key)
        _, U, J = opt(xt, U, key)

        ut = U[0]

        xt = dyn.step(xt, ut)
        X = X.at[t+1].set(xt)
        cost_hist.append(J)

        print(t, J, xt, ut)

        ## shift over the plan by one action
        U = jnp.roll(U, shift = -1, axis = 0)
        U = U.at[-1].set(jnp.zeros(dyn.control_dim))

    times = jnp.linspace(0.0, steps * dt, steps)

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Horizon Empowerment')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Empowerment (nats)')

    # cost_hist = jax.random.normal(key, (steps))
    ax.plot(times, cost_hist)
    ax.set_xlim(0.0, steps * dt)
    fig.savefig(name + '/cost.png')

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Trajectory')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta Dot')
    ax.plot(X[:, 0], X[:, 1])
    ax.scatter(X[0, 0], X[0, 1], color = 'green', label = 'Start')
    ax.scatter(X[-1, 0], X[-1, 1], color = 'red', label = 'End')
    ax.legend()
    fig.savefig(name + '/trajectory.png')

    # dyn.render(X, path = name + '/single.mp4', skip = 5)
    # dyn.render(X, path = name + '/single.mp4', skip = 2)
    dyn.render(X, path = name + '/test.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
    # dyn.render(X, path = name + '/cart_pole.mp4', distance = 5, skip = 2, lookat = jnp.array([0, 0, 0]))
    # render_go(dyn, X, path = name + '/go.mp4')
