
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/content/Iteration#3")

import methods


# --------------------------------------------------
# 1. Six-page simulated web graph
# --------------------------------------------------

out_links = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [4],
    4: [0, 3],
    5: []
}

n = len(out_links)
alpha = 0.85

# Build adjacency matrix using A[i,j] = 1 if j links to i
A = np.zeros((n, n))

for source, destinations in out_links.items():
    for destination in destinations:
        A[destination, source] = 1

print("Six-page adjacency matrix:")
print(A)


# --------------------------------------------------
# 2. True PageRank using NumPy linear solver
#    (reference result only)
# --------------------------------------------------

S, G = methods.my_google_matrix(A, alpha)

print("\nLink transition matrix S:")
print(S)

print("\nGoogle matrix G:")
print(G)

# Reference PageRank using NumPy
M = np.eye(n) - alpha * S
b = np.full(n, (1 - alpha) / n)

true_pagerank = np.linalg.solve(M, b)

print("\nReference PageRank:")
print(true_pagerank)


# --------------------------------------------------
# 3. Test power iteration
# --------------------------------------------------

power_rank, power_steps = methods.my_power_iteration(G)

print("\nPower iteration PageRank:")
print(power_rank)

print("Iterations:", len(power_steps))
print("Sum:", power_rank.sum())

power_error = np.max(np.abs(power_rank - true_pagerank))
print("Maximum error:", power_error)


# --------------------------------------------------
# 4. Test Gaussian elimination
# --------------------------------------------------

M = np.eye(n) - alpha * S
b = np.full(n, (1 - alpha) / n)

gauss_rank = methods.my_gauss_solve(M, b)

print("\nGaussian elimination PageRank:")
print(gauss_rank)

gauss_error = np.max(np.abs(gauss_rank - true_pagerank))
print("Maximum error:", gauss_error)


# --------------------------------------------------
# 6. Compare with NumPy eigenvector
# --------------------------------------------------

eigenvalues, eigenvectors = np.linalg.eig(G)

index = np.argmin(np.abs(eigenvalues - 1))
eigen_rank = np.real(eigenvectors[:, index])

if eigen_rank.sum() < 0:
    eigen_rank = -eigen_rank

eigen_rank = eigen_rank / eigen_rank.sum()

print("\nNumPy eigenvector PageRank:")
print(eigen_rank)

eigen_error = np.max(np.abs(eigen_rank - true_pagerank))

print("Eigenvector maximum error:", eigen_error)


# --------------------------------------------------
# 7. Convergence rate and log plot
# --------------------------------------------------

x = np.ones(n) / n

actual_errors = []
iteration_sums = []

for k in range(100000):
    actual_errors.append(
        np.linalg.norm(x - true_pagerank, ord=1)
    )

    iteration_sums.append(x.sum())

    x_next = G @ x

    if np.linalg.norm(x_next - x, ord=1) < 1e-13:
        break

    x = x_next

actual_errors = np.array(actual_errors)
iteration_sums = np.array(iteration_sums)

iterations = np.arange(len(actual_errors))

# Estimate convergence slope using log error
valid = actual_errors > 1e-12

slope = np.polyfit(
    iterations[valid],
    np.log(actual_errors[valid]),
    1
)[0]

print("\nEstimated convergence slope:", slope)
print("Theoretical upper bound log(alpha):", np.log(alpha))
print("Slope satisfies bound:",
      slope <= np.log(alpha) + 1e-10)

print("Every iteration sums to 1:",
      np.allclose(iteration_sums, 1.0, atol=1e-10))

plt.figure(figsize=(8, 5))
plt.semilogy(iterations, actual_errors, marker="o")
plt.xlabel("Iteration")
plt.ylabel("L1 error")
plt.title("PageRank Power Iteration Convergence")
plt.grid(True)
plt.show()


# --------------------------------------------------
# 8. Stress test with alpha = 0.99
# --------------------------------------------------

S_stress, G_stress = methods.my_google_matrix(A, 0.99)

rank_stress, steps_stress = methods.my_power_iteration(
    G_stress,
    tol=1e-13,
    max_iter=100000
)

print("\nStress test (alpha = 0.99):")
print("Iterations:", len(steps_stress))
print("Sum of PageRank:", rank_stress.sum())


# --------------------------------------------------
# 9. Final numerical checks
# --------------------------------------------------

print("\nFINAL CHECKS")

print("Power iteration accurate:",
      power_error < 1e-10)

print("Gaussian elimination accurate:",
      gauss_error < 1e-10)

print("Every iteration sums to 1:",
      np.allclose(iteration_sums, 1.0, atol=1e-10))

print("Eigenvector accurate:",
      eigen_error < 1e-10)
