import sympy as sp
sp.init_printing()
import numpy as np
import matplotlib.pyplot as plt

def step(x, u, dt):
    theta, alpha = x
    new_theta = theta + dt * alpha
    new_alpha = alpha - dt * theta**3 + dt * u
    return sp.Matrix([new_theta, new_alpha])

if __name__ == '__main__':

    dt = sp.symbols('dt', positive = True)
    theta_0, alpha_0 = sp.symbols('theta_0 alpha_0')
    # Define symbolic control sequence
    N = 3
    U = sp.symbols(f'u0:{N}')

    x = sp.Matrix([theta_0, alpha_0])

    for t in range(N):
        x = step(x, U[t], dt)

    F = x.jacobian(U)
    F = F.subs({
        dt : 0.05,
        U[0]: 0.0,
        U[1]: 0.0,
        U[2]: 0.0,
        # U[3]: 0.0,
        # U[4]: 0.0,
        # U[5]: 0.0
    })

    

    S = F @ F.T

    trS = sp.trace(S)
    detS = S.det()
    # Explicit eigenvalues
    lambda1 = (trS + sp.sqrt(trS**2 - 4*detS)) / 2
    lambda2 = (trS - sp.sqrt(trS**2 - 4*detS)) / 2

    # print(
    #     sp.solve(sp.diff(lambda2, theta_0), theta_0)
    # )

    sp.pprint(sp.diff(lambda1, theta_0).subs({alpha_0: 0.0}))
    # sp.pprint(lambda1.subs({theta_0: 0.0, alpha_0: 0.0}))


    # sp.pprint(h2)
    # def evaluate_eig(theta_0_val: float, alpha_0_val: float):
    #     return h2[1].subs({theta_0: theta_0_val, alpha_0: alpha_0_val})
    
    # thetas = np.linspace(-10.0, 10.0, 50)
    # alphas = np.linspace(-10.0, 10.0, 50)
  
    # ## Initialize a 2D array for empowerment
    # empowerment = np.zeros((len(alphas), len(thetas)))

    # ## Compute empowerment for each combination of theta and alpha
    # for i, theta in enumerate(thetas):
    #     for j, alpha in enumerate(alphas):
    #         e = evaluate_eig(theta, alpha)
    #         empowerment[j, i] = e
    #         print(i, j, e)

    # ## plot the heatmap
    # fig, ax = plt.subplots(figsize=(6, 5))
    # im = ax.imshow(
    #     empowerment, 
    #     extent = [thetas[0], thetas[-1], alphas[0], alphas[-1]],
    #     origin = 'lower', 
    #     aspect = 'auto', 
    #     cmap = 'viridis')

    # ax.set_xlabel("Theta")
    # ax.set_ylabel("Alpha")
    # ax.set_title("Empowerment Heatmap")
    # fig.colorbar(im, ax=ax, label="Empowerment")
    # plt.show()