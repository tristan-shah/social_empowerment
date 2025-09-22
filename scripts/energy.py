import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

def energy(x):
    return 0.5 * x ** 2

def grad_energy(x):
    return 2 * x * energy(x)

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    eta = 0.1
    N = 2000

    x = jnp.linspace(-4.0, 4.0, 1000)
    y = jnp.exp(-energy(x))

    ## initial distribution p_0(x)
    z = jax.random.uniform(key, (N,)) + 2.0

    level = jnp.zeros(z.shape)

    for i in range(10):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, z.shape)
        z = z - eta * grad_energy(z) + jnp.sqrt(2 * eta) * noise

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y / y.sum())
        ax.scatter(z, level, alpha = 0.1)
        ax.hist(z, bins = 50, density = True, alpha=0.5, label="Samples")
        plt.show()