import os
os.environ['MUJOCO_PY_FORCE_FORK'] = 'False'

import mujoco
from mujoco import mjx
import jax
from jax import numpy as jnp
from jax import Array
import imageio

import matplotlib.pyplot as plt

# Enable higher precision (critical for second derivatives)
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

class Dynamics:
    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.mjx_model = mjx.put_model(self.model)

        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.state_dim = self.nq + self.nv
        self.control_dim = self.mjx_model.nu

        ## jit the jax step function
        self.mjx_step = jax.jit(mjx.step)
        self.J = jax.jit(jax.jacfwd(self.step, argnums = (0, 1)))

    def step(self, xt: Array, ut: Array):

        qpos = xt[:self.mjx_model.nq]
        qvel = xt[self.mjx_model.nq:]
        data = mjx.make_data(self.model).replace(qpos = qpos, qvel = qvel, ctrl = ut)
        data = self.mjx_step(self.mjx_model, data)
        return jnp.concatenate([data.qpos, data.qvel])
    
    def render(self, X: Array, path: str):

        renderer = mujoco.Renderer(self.model, height = 720, width = 1280)

        # Create a free camera
        camera = mujoco.MjvCamera()
        camera.lookat = jnp.array([0.0, 0.0, 1.2])  # Point the camera is looking at (x, y, z)
        camera.distance = 3.0  # Distance from the lookat point
        camera.azimuth = 90.0  # Horizontal angle (degrees, 0 = looking along +x)
        camera.elevation = 10.0  # Vertical angle (degrees, -90 = straight down)

        data = mujoco.MjData(self.model)
        writer = imageio.get_writer(path, fps = 60)
        # frames = []

        for t in range(X.shape[0]):

            data.qpos, data.qvel = split_state(X[t], self.nq)
            mujoco.mj_forward(self.model, data)
            
            renderer.update_scene(data, camera = camera)
            writer.append_data(renderer.render())
            # frames.append(renderer.render())

        renderer.close()
        writer.close()

        # print(frames)
        return

if __name__ == '__main__':

    dyn = Dynamics(path = 'xml/pendulum.xml')
    xt = jnp.zeros(dyn.state_dim)
    ut = jnp.zeros(dyn.control_dim)
    xt = xt.at[0].set(3.1)

    T = 500
    X = jnp.zeros((T, dyn.state_dim))

    for t in range(T):
        xt = dyn.step(xt, ut)
        print(xt)
        X = X.at[t].set(xt)

    dyn.render(X, path = 'pendulum.mp4')