from jax import numpy as jnp

if __name__ == '__main__':

    # ## phase 1
    # M = 1
    # mat = jnp.array([
    #     jnp.array([0, 0, 0, 0, 1, 1, 0]),
    #     jnp.array([1, 1, 1, 0, 1, 0, 7]),
    #     jnp.array([2, -5, 1, -1, 0, 1, 10])
    # ])

    # mat = mat.at[0, :].set(mat[0, :] - mat[1, :] - mat[2, :])

    # ## iteration 1
    # row = mat[2, :] / 2.0
    # col = mat[:, 0]
    # mat = mat - row[None, :] * col[:, None]
    # mat = mat.at[2, :].set(row)

    # ## iteration 2
    # row = mat[1, :] / 3.5
    # col = mat[:, 1]

    # mat = mat - row[None, :] * col[:, None]
    # mat = mat.at[1, :].set(row)
    # print(mat)

    mat = jnp.array([
        jnp.array([-1, -2, -1, 0, 0]),
        jnp.array([0, 1, 1/7, 1/7, 0.57]),
        jnp.array([1, 0, 0.86, -1/7, 6.43])
    ])

    ## iteration 1
    row = mat[1, :]
    col = mat[:, 1]
    mat = mat - row[None, :] * col[:, None]
    mat = mat.at[1, :].set(row)
    print(mat)
    print()

    ## iteration 2
    row = mat[2, :]
    col = mat[:, 0]
    mat = mat - row[None, :] * col[:, None]
    mat = mat.at[2, :].set(row)

    print(mat)