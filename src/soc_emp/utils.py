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

import numpy as np
import mujoco

def diff_qpos(model, q1: np.ndarray, q2: np.ndarray) -> jnp.ndarray:
    """
    Compute the difference between two MuJoCo qpos states, handling joint-specific differences.
    
    Args:
        model: MuJoCo model (mujoco.MjModel).
        q1: First qpos state (np.ndarray of shape [model.nq]).
        q2: Second qpos state (np.ndarray of shape [model.nq]).
    
    Returns:
        dq: Difference in generalized coordinates (jnp.ndarray of shape [model.nq]).
    
    Raises:
        ValueError: If q1 or q2 have incorrect shapes or types.
        RuntimeError: If an unsupported joint type is encountered.
    """
    # Validate inputs
    if not isinstance(q1, (np.ndarray, jnp.ndarray)) or not isinstance(q2, (np.ndarray, jnp.ndarray)):
        raise ValueError("q1 and q2 must be numpy or jax arrays")
    q1 = jnp.array(q1, dtype=jnp.float32)
    q2 = jnp.array(q2, dtype=jnp.float32)
    if q1.shape != (model.nq,) or q2.shape != (model.nq,):
        raise ValueError(f"q1 and q2 must have shape ({model.nq},)")

    dq = jnp.zeros(model.nq, dtype=jnp.float32)

    for jid in range(model.njnt):
        adr = model.jnt_qposadr[jid]
        jtype = model.jnt_type[jid]

        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            # Handle angle wrap for hinge joints
            dq = dq.at[adr].set((q2[adr] - q1[adr] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            # Linear difference for slide joints
            dq = dq.at[adr].set(q2[adr] - q1[adr])

        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            # Quaternion difference for ball joints
            qa = q2[adr:adr+4].reshape((4, 1))
            qb = q1[adr:adr+4].reshape((4, 1))
            # Normalize quaternions
            qa = qa / jnp.linalg.norm(qa)
            qb = qb / jnp.linalg.norm(qb)
            res = np.zeros((3, 1))  # MuJoCo's mju_subQuat may require NumPy
            mujoco.mju_subQuat(res, qa, qb)
            dq = dq.at[adr:adr+3].set(res.flatten())

        elif jtype == mujoco.mjtJoint.mjJNT_FREE:
            # Position difference
            dq = dq.at[adr:adr+3].set(q2[adr:adr+3] - q1[adr:adr+3])
            # Quaternion difference
            qa = q2[adr+3:adr+7].reshape((4, 1))
            qb = q1[adr+3:adr+7].reshape((4, 1))
            # Normalize quaternions
            qa = qa / jnp.linalg.norm(qa)
            qb = qb / jnp.linalg.norm(qb)
            res = np.zeros((3, 1))
            mujoco.mju_subQuat(res, qa, qb)
            dq = dq.at[adr+3:adr+6].set(res.flatten())

        else:
            raise RuntimeError(f"Unsupported joint type {jtype} at joint ID {jid}")

    return dq