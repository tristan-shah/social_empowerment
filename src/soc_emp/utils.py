import jax
from jax import Array
from jax import numpy as jnp
from mujoco.mjx import Data

def smooth_angle_wrap(theta: float):
    return jax.lax.atan2(jax.lax.sin(theta), jax.lax.cos(theta))

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

def get_state(data: Data):
    return jnp.concatenate([data.qpos, data.qvel])

def select_output(f: callable, index: int):
    return lambda *args, **kwargs: f(*args, **kwargs)[index]