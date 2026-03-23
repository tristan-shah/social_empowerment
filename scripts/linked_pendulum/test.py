import jax
from jax import numpy as jnp

from soc_emp import Dynamics

from sweep_power import make_run, set_tendon_properties

if __name__ == '__main__':
    stiffness = 3.0
    damping = 0.1
    dt = 0.01
    ## hyperparams
    steps = 2000
    state_type = 'angle'
    control_type = 'egoistic'
    horizon = 50
    alpha = 0.01
    observation_noise = 1.0

    # load dynamics
    xml_path = 'xml/custom/linked_pendulums.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    dyn = set_tendon_properties(dyn, stiffness, damping)

    run = make_run(dyn, steps, state_type, control_type, horizon, alpha, observation_noise)

    power_density = jnp.array([1.2, 1.0])

    key, subkey = jax.random.key(0)
    X = run(power_density, subkey)
