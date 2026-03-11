from pathlib import Path
from jax import numpy as jnp

import matplotlib.pyplot as plt

if __name__ == '__main__':


    path = Path('results/flock/leader2/') 
    emp_path = path / 'empowerment_hist.npy'
    order_path = path / 'order_parameter_hist.npy'

    emp = jnp.load(emp_path)
    order = jnp.load(order_path)

    print(emp)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Empowerment (nats)')
    ax.plot(emp[:, 0], color = 'red', label = 'Dictator')
    for i in range(1, emp.shape[1]):
        ax.plot(emp[:, i], color = 'blue', alpha = 0.1)

    ax.legend()
    fig.savefig('dictator.png', dpi = 300)
    plt.show()


    # fig, ax = plt.subplots(1, 1)

    # ax.plot(order)
    # plt.show()