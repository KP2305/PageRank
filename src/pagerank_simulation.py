# -*- coding: utf-8 -*-
"""Six-page PageRank simulation (DS5020 Final Project, Iteration 2).

Builds the adjacency matrix A, link matrix S and Google matrix G for the
six-page example (page 5 is dangling, alpha = 0.85), then:
  1. runs power iteration, asserting at every step that x_k is a probability
     vector (Question 5);
  2. computes the eigenvector of G for eigenvalue 1;
  3. simulates a random surfer for 100,000 steps (seed 0).

Run from the repository root:  python src/pagerank_simulation.py
"""

import numpy as np

# Number of web pages
n = 6

# Damping factor
alpha = 0.85

# Define the outgoing links for each page
links = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [4],
    4: [0, 3],
    5: []
}

# Create the adjacency matrix
A = np.zeros((n, n))

for j, outgoing_pages in links.items():
    for i in outgoing_pages:
        A[i, j] = 1

print("Adjacency Matrix A:")
print(A)

# Calculate the out-degree of each page
out = A.sum(axis=0)

# Construct the link matrix S
S = np.zeros((n, n))

for j in range(n):
    if out[j] > 0:
        S[:, j] = A[:, j] / out[j]
    else:
        S[:, j] = 1 / n

print("Link Matrix S:")
print(S)

# Check column sums
print("\nColumn sums of S:")
print(S.sum(axis=0))

# Construct the Google matrix
G = alpha * S + ((1 - alpha) / n) * np.ones((n, n))

print("Google Matrix G:")
print(G)

# Check column sums
print("\nColumn sums of G:")
print(G.sum(axis=0))

# Check whether all entries are positive
print("\nIs every entry of G positive?")
print(np.all(G > 0))

# Check the minimum entry
print("\nMinimum entry of G:")
print(G.min())

# Initialize equal probability for all six pages
x = np.ones(n) / n

# Convergence tolerance
tolerance = 1e-10

# Maximum number of iterations
max_iterations = 1000

# Power iteration
for k in range(max_iterations):
    x_new = G @ x

    # Q5: verify every iterate is a probability vector
    assert (x_new >= 0).all() and abs(x_new.sum() - 1) < 1e-12

    # Calculate the difference between iterations
    difference = np.linalg.norm(x_new - x, ord=1)

    # Update the PageRank vector
    x = x_new

    # Stop if converged
    if difference < tolerance:
        break

print("All iterates were probability vectors (checked at every step).")
print("Final PageRank Scores:")
for i in range(n):
    print(f"Page {i}: {x[i]:.10f}")

print("\nNumber of iterations:", k + 1)
print("Sum of PageRank scores:", x.sum())
print("Final difference:", difference)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(G)

# Find the eigenvector corresponding to eigenvalue closest to 1
index = np.argmin(np.abs(eigenvalues - 1))

# Extract the corresponding eigenvector
v = np.real(eigenvectors[:, index])

# Normalize the eigenvector
v = v / v.sum()

print("PageRank using Eigenvector Method:")
for i in range(n):
    print(f"Page {i}: {v[i]:.10f}")

# Compare both methods
print("\nDifference between the two methods:")
print(np.linalg.norm(x - v, ord=1))

# Random Surfer Simulation

rng = np.random.default_rng(0)

num_steps = 100_000
current_page = 0

visits = np.zeros(n, dtype=int)

for _ in range(num_steps):
    current_page = rng.choice(n, p=G[:, current_page])
    visits[current_page] += 1

# Calculate visit frequencies
frequencies = visits / num_steps

print("Random Surfer Results:")
for i in range(n):
    print(f"Page {i}: {frequencies[i]:.10f}")

print("\nComparison with Power Iteration:")
for i in range(n):
    print(
        f"Page {i}: "
        f"PageRank = {x[i]:.10f}, "
        f"Visit Frequency = {frequencies[i]:.10f}"
    )
