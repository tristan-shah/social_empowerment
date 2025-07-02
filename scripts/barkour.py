from soc_emp import Dynamics

import jax
from jax import numpy as jnp

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    T = 3000
    
    dyn = Dynamics(path='xml/google_barkour_vb/scene_hfield_mjx.xml')

    X = jnp.zeros((T, dyn.state_dim))

    xt = dyn.init_state()
    xt = xt.at[2].set(0.5)

    X = X.at[0].set(xt)
    U = jax.random.normal(key, (T, dyn.control_dim))

    # for t in range(T):
    #     ut = U[t]
    #     xt = dyn.step(xt, ut)
    #     X = X.at[t+1].set(xt)
    #     print(xt)

    # dyn.render(X, path = 'barkour.mp4', skip = 5)


    A, B = dyn.linearize(X[0], U[0])

    print(A)
    print(B)

    # T = 300
    # U = dyn.random_control_sequence(T)

    # X, U = dyn.forward(x0, U=U)
    # A, B = dyn.batch_linearize(X[:-1], U)

    # cost = []
    # for i in range(15):
    #     J, X, U, A, B = dyn.update(x0, X, U, A, B)
    #     print(i, J)
    #     cost.append(J)

    # dyn.render(X, path='ball.mp4', lookat=[2.5, 0.0, 0.5], elevation=-20.0, distance=5.0, skip=1)