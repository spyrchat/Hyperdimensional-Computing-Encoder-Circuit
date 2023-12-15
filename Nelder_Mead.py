import numpy as np
import matplotlib.pyplot as plt

def evaluate_F_of_x_2(gamma, alpha_sp):
    a = 1
    b = 100
    return (a - gamma)**2 + b * (alpha_sp - gamma**2)**2

# Nelder-Mead parameters
NM_iter = 350
STD_EPS = 0.002
alpha_simp = 1
gamma_simp = 2 * 0.6
rho_simp = 0.5
sigma_simp = 0.5

# Initial Simplex
Simplex = [np.array([1.5, 2.0]), np.array([-1.0, 2.0]), np.array([1.0, -1.0])]

objective_ = []
STD_ = []

for iter_ in range(NM_iter):
    F_of_x = [evaluate_F_of_x_2(s[0], s[1]) for s in Simplex]

    sorted_indices = np.argsort(F_of_x)
    Simplex = np.array(Simplex)[sorted_indices]
    F_of_x = np.array(F_of_x)[sorted_indices]

    best_objective_value = F_of_x[0]
    objective_.append(best_objective_value)
    STD_.append(np.std(F_of_x))

    if np.std(F_of_x) < STD_EPS and iter_ > 100:
        break

    x_0 = np.mean(Simplex[:-1], axis=0)
    x_r = x_0 + alpha_simp * (x_0 - Simplex[-1])
    F_xr = evaluate_F_of_x_2(x_r[0], x_r[1])

    if F_of_x[0] <= F_xr < F_of_x[-2]:
        Simplex[-1] = x_r
        F_of_x[-1] = F_xr
    elif F_xr < best_objective_value:
        x_e = x_0 + gamma_simp * (x_r - x_0)
        F_xe = evaluate_F_of_x_2(x_e[0], x_e[1])
        if F_xe < F_xr :
            Simplex[-1] = x_e
            F_of_x[-1] = F_xe
        else:
            F_of_x[-1] = F_xr
            Simplex[-1] = x_r

    else:
        flag = False
        if F_xr < F_of_x[-1]:
            x_c = x_0 + rho_simp * (Simplex[-1] - x_0)
            F_c = evaluate_F_of_x_2(x_c[0], x_c[1])
            if F_c < F_xr:
                flag = True
        elif F_xr >= F_of_x[-1]:
            x_c = x_0 + rho_simp * (F_of_x[-1] - x_0)
            F_c = evaluate_F_of_x_2(x_c[0],x_c[1])
            if F_c < F_of_x[-1]:
                flag = True
            
        if flag:
            F_of_x[-1] = F_c  
            Simplex[-1,:] = x_c
        else:
            for i in range(1, len(Simplex)):
                Simplex[i,:] = Simplex[1,:] + sigma_simp * (Simplex[i,:] - Simplex[1,:])
                F_shrink = evaluate_F_of_x_2(Simplex[i,0],Simplex[i,1])
                F_of_x[i] = F_shrink
print("Final Simplex:", Simplex)
print("Best Objective:", best_objective_value)

plt.figure()
plt.plot(objective_, '.-')
plt.title("Objective Value Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(STD_, '.-')
plt.title("Standard Deviation Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Standard Deviation")
plt.grid(True)
plt.show()
