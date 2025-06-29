import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx

from soc_emp import Dynamics
from soc_emp.empowerment import compute_F
    
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## load in xml
    xml_path = 'xml/custom/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    xt = dyn.init_state()
    xt = xt.at[0].set(3.1)
    U = jnp.zeros((300, dyn.control_dim))

    # xt = dyn.unroll(xt, U)
    F = dyn.compute_F(xt, U)
    print(F)

    # dyn.render(X, path = 'pendulum.mp4')

