from scipy.optimize import linprog
import numpy as np

if __name__ == '__main__':
    c = np.array([2, 2])

    A_ub = np.array([
        [2, 5],
        [6, 5]
    ])

    b_ub = np.array([27, 16])

    A_eq = np.array([
        [0, 1]
    ])

    b_eq = np.array([4])

    print(
        linprog(-c, A_ub, b_ub, A_eq, b_eq)
    )