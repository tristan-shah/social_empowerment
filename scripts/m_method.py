from jax import numpy as jnp

if __name__ == '__main__':

    M = 100
    mat = jnp.array([
        [-1, -5, -3, M, M, 0],
        [1, 2, 1, 1, 0, 6],
        [2, -1, 0, 0, 1, 8]
    ])

    mat = mat.at[0, :].set(mat[0, :] - M * mat[1, :] - M * mat[2, :])

    ## iteration 1
    row = mat[2, :] / 2
    col = mat[:, 0]
    mat = mat - row[None, :] * col[:, None]
    mat = mat.at[2, :].set(row)

    ## iteration 2
    row = mat[1, :] / 2.5
    col = mat[:, 1]
    mat = mat - row[None, :] * col[:, None]
    mat = mat.at[1, :].set(row)

    ## iteration 3
    row = mat[1, :] / 0.4
    col = mat[:, 2]
    mat = mat - row[None, :] * col[:, None]
    mat = mat.at[1, :].set(row)

    print(mat)

