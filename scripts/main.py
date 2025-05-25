import time

import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
from jax import numpy as jnp

# Enable higher precision (critical for second derivatives)
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

if __name__ == '__main__':
    ## jit the jax step function
    mjx_step = jax.jit(mjx.step)

    model = mujoco.MjModel.from_xml_path('xml/pendulum.xml')
    # model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml')

    data = mujoco.MjData(model)

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)

    ## define state and control input
    nq, nv, nu = mjx_model.nq, mjx_model.nv, mjx_model.nu
    xt = jnp.zeros(nq + nv)
    ut = jnp.zeros(nu)
    # ut = ut.at[0].set(1.0)
    print(xt, ut)

    mjx_data = mjx_data.replace(qpos = jnp.array([3.1]))
    mjx.get_data_into(data, model, mjx_data)


    ## simulate
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.azimuth = 90     # angle from above
        viewer.cam.elevation = -10  # tilt angle
        viewer.cam.distance = 5.0   # distance from target

        while viewer.is_running():

            mjx_data = mjx_data.replace(ctrl = ut)
            mjx_data = mjx_step(mjx_model, mjx_data)
            mjx.get_data_into(data, model, mjx_data)

            print(mjx_data.qpos, mjx_data.qvel)
            # Update the viewer
            viewer.sync()
            time.sleep(0.001)