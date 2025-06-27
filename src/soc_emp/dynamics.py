from typing import Optional

import mujoco
from mujoco import mjx
from mujoco.mjx import Data
import jax
# Enable higher precision (critical for second derivatives)
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

from jax import numpy as jnp
from jax import Array
import imageio

from soc_emp.utils import split_state, get_state

class Dynamics:
    def __init__(self, path: Optional[str] = None, string: Optional[str] = None):

        assert (path is not None) != (string is not None)

        if path is not None:
            self.model = mujoco.MjModel.from_xml_path(path)
        elif string is not None:
            self.model = mujoco.MjModel.from_xml_string(string)

        self.mjx_model = mjx.put_model(self.model)

        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.state_dim = self.nq + self.nv
        self.control_dim = self.mjx_model.nu

        ## jit the jax step function
        self.mjx_step = jax.jit(mjx.step)
        self.linearize = jax.jit(jax.jacfwd(self.step, argnums = (0, 1)))

    def init_state(self, data: Optional[Data] = None):
        
        if data is None:
            data = mjx.make_data(self.mjx_model)

        return jnp.concatenate([data.qpos, data.qvel * 0.0])

    def step(self, xt: Array, ut: Array):
        qpos, qvel = split_state(xt, self.nq)
        mjx_data = mjx.make_data(self.mjx_model).replace(qpos = qpos, qvel = qvel, ctrl = ut)
        mjx_data = self.mjx_step(self.mjx_model, mjx_data)
        return get_state(mjx_data)

    def step_data(self, data: Data, xt: Array, ut: Array):
        qpos, qvel = split_state(xt, self.nq)
        data = data.replace(qpos = qpos, qvel = qvel, ctrl = ut)
        data = self.mjx_step(self.mjx_model, data)
        return data, get_state(data)
    
    def render(
            self,
            X: Array, 
            path: str,
            lookat: Array = jnp.array([0.0, 0.0, 1.0]),
            distance: float = 3.0,
            azimuth: float = 90.0,
            elevation: float = 0.0,
            k: int = 5
            ):

        renderer = mujoco.Renderer(self.model, height = 720, width = 1280)

        # Create a free camera
        camera = mujoco.MjvCamera()
        camera.lookat = lookat  # Point the camera is looking at (x, y, z)
        camera.distance = distance  # Distance from the lookat point
        camera.azimuth = azimuth  # Horizontal angle (degrees, 0 = looking along +x)
        camera.elevation = elevation  # Vertical angle (degrees, -90 = straight down)

        data = mujoco.MjData(self.model)
        writer = imageio.get_writer(path, fps = 60)

        for t in range(0, X.shape[0], k):

            data.qpos, data.qvel = split_state(X[t], self.nq)
            mujoco.mj_forward(self.model, data)
            
            renderer.update_scene(data, camera = camera)
            writer.append_data(renderer.render())

        renderer.close()
        writer.close()
        return None