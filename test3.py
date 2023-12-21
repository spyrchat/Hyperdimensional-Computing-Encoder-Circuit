import numpy as np
import matplotlib.pyplot as plt

def evaluate_F_of_x_2(gamma, alpha_sp, a, b):
    # Rosenbrock
    # return (a - gamma)**2 + b * (alpha_sp - gamma**2)**2
   
    # Booth's Function
    return (a * gamma + 2 * b - 7)**2 + (2 * gamma + b * alpha_sp - 5)**2

# Nelder-Mead parameters

# Define the number of different values for a and b
num_values = 10  # 10 different values for each parameter

number_of_successful_tests = 0
number_of_failed_tests = 0
# Generate the values for a and b
a_values = np.linspace(0.5, 2, num_values)
b_values = np.linspace(50, 150, num_values)

# Iterate over all combinations of a and b
for i in range(1,100):
    
    a = 1 * (i+50)/(i)
    b = 100*i/(i+1)
    print(f"Running test with a = {a}, b = {b}")

    # Nelder-Mead parameters
    NM_iter = 350
    STD_EPS = 0.0002
    alpha_simp = 1
    gamma_simp = 2 * 0.6
    rho_simp = 0.5
    sigma_simp = 0.5

    # Initial Simplex
    Simplex = [np.array([1.5, 2.0]), np.array([-1.0, 2.0]), np.array([1.0, -1.0])]
    objective_ = []
    STD_ = []

    for iter_ in range(NM_iter):
        F_of_x = [evaluate_F_of_x_2(s[0], s[1], a, b) for s in Simplex]

        # ... Rest of the Nelder-Mead algorithm ...
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
        F_xr = evaluate_F_of_x_2(x_r[0], x_r[1], a, b)

        if F_of_x[0] <= F_xr < F_of_x[-2]:
            Simplex[-1] = x_r
            F_of_x[-1] = F_xr
        elif F_xr < best_objective_value:
            x_e = x_0 + gamma_simp * (x_r - x_0)
            F_xe = evaluate_F_of_x_2(x_e[0], x_e[1], a, b)
            if F_xe < F_xr:
                Simplex[-1] = x_e
                F_of_x[-1] = F_xe
            else:
                F_of_x[-1] = F_xr
                Simplex[-1] = x_r
        else:
            # Contraction
            flag = False
            if F_xr < F_of_x[-1]:
                x_c = x_0 + rho_simp * (x_r - x_0)
                F_c = evaluate_F_of_x_2(x_c[0], x_c[1], a, b)
                flag = F_c < F_of_x[-1]
            else:
                x_c = x_0 + rho_simp * (Simplex[-1] - x_0)
                F_c = evaluate_F_of_x_2(x_c[0], x_c[1], a, b)
                flag = F_c < F_xr

            if flag:
                Simplex[-1,:] = x_c
                F_of_x[-1] = F_c
            else:
                # Shrink
                for i in range(1, len(Simplex)):
                    Simplex[i,:] = Simplex[0,:] + sigma_simp * (Simplex[i,:] - Simplex[0,:])
                    F_shrink = evaluate_F_of_x_2(Simplex[i,0], Simplex[i,1], a, b)
                    F_of_x[i] = F_shrink

    # Evaluate success
    final_optimum = best_objective_value
    success = abs(0 - final_optimum) <= 0.0001
    if success:
        number_of_successful_tests += 1
    else:
        number_of_failed_tests += 1
print(f"Optimization {'successful' if success else 'unsuccessful'}")
print(f"Best objective value found: {final_optimum}\n")

print("Testing with Booth's Function :")
print("Percentage of Successful Tests :", number_of_successful_tests / (number_of_failed_tests + number_of_successful_tests) * 100)

# Plotting
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
