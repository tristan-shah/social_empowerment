from jax import numpy as jnp

if __name__ == '__main__':

    mat = jnp.array([
        [-3, 1, -3, -4, 0, 0, 0, 0],
        [1, 2, 2, 4, 1, 0, 0, 40],
        [2, -1, 1, 2, 0, 1, 0, 8],
        [4, -2, 1, -1, 0, 0, 1, 10]
    ])

    ## iteration 1
    row = jnp.array([2.0, -1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 8.0]) / 2.0
    col = jnp.array([-4.0, 4.0, 2.0, -1.0])
    norm = row[None, :] * col[:, None]
    mat = mat - norm
    mat = mat.at[2, :].set(row)

    ## iteration 2
    row = mat[1, :] / 4
    col = mat[:, 1]
    norm = row[None, :] * col[:, None]
    mat = mat - norm
    mat = mat.at[1, :].set(row)

    ## iteration 3
    row = mat[2, :] / 0.5
    col = mat[:, 2]
    norm = row[None, :] * col[:, None]
    mat = mat - norm
    print(mat)
    print(row)