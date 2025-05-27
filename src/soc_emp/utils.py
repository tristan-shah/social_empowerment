import jax
from jax import Array

def smooth_angle_wrap(theta: float):
    return jax.lax.atan2(jax.lax.sin(theta), jax.lax.cos(theta))

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]