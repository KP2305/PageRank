
import numpy as np


# Equation: d_j = sum_i A[i,j]
# Calculate the out-degree of every page.
def my_out_degrees(A):
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    return np.sum(A, axis=0)


# Equation:
# S[i,j] = A[i,j]/d_j, when d_j > 0
# S[i,j] = 1/n, when d_j = 0
# Construct the column-stochastic link matrix.
def my_link_matrix(A):
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]
    d = my_out_degrees(A)

    S = np.zeros((n, n), dtype=float)

    for j in range(n):
        if d[j] > 0:
            S[:, j] = A[:, j] / d[j]
        else:
            S[:, j] = 1.0 / n

    return S


# Equation: G = alpha*S + ((1-alpha)/n)*11^T
# Construct the Google matrix.
def my_google_matrix(A, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    S = my_link_matrix(A)
    n = S.shape[0]

    G = alpha * S + ((1 - alpha) / n) * np.ones((n, n))

    return S, G


# Equation: x_(k+1) = G*x_k
# Perform one power iteration step.
def my_pagerank_step(G, x):
    return G @ x


# Equation: x_(k+1) = G*x_k
# Stop when ||x_(k+1)-x_k||_1 < tol.
def my_power_iteration(G, tol=1e-13, max_iter=100000):
    G = np.asarray(G, dtype=float)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be square.")

    n = G.shape[0]
    x = np.ones(n) / n
    steps = []

    for _ in range(max_iter):
        x_new = my_pagerank_step(G, x)
        difference = np.sum(np.abs(x_new - x))

        steps.append(difference)
        x = x_new

        if difference < tol:
            return x, np.array(steps)

    raise RuntimeError("Power iteration did not converge.")


# Equation: Mx = b
# Solve a linear system using Gaussian elimination
# with partial pivoting and back substitution.
def my_gauss_solve(M, b):
    M = np.array(M, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    n = len(b)

    if M.shape != (n, n):
        raise ValueError("M must be square and match b.")

    for k in range(n):
        pivot_row = k + np.argmax(np.abs(M[k:, k]))

        if abs(M[pivot_row, k]) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular.")

        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] - np.dot(M[i, i + 1:], x[i + 1:])
        ) / M[i, i]

    return x


# Equation: Gx = alpha*Sx + ((1-alpha)/n)*1
# Compute PageRank using out-link lists without constructing
# a dense adjacency, link, or Google matrix.
def my_pagerank_outlinks(out_links, alpha=0.85,
                         tol=1e-12, max_iter=100000):

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    n = len(out_links)
    x = np.ones(n) / n
    steps = []

    dangling = np.array(
        [len(targets) == 0 for targets in out_links]
    )

    for _ in range(max_iter):
        link_contribution = np.zeros(n)

        for j, targets in enumerate(out_links):
            if len(targets) > 0:
                link_contribution[targets] += (
                    x[j] / len(targets)
                )

        dangling_mass = np.sum(x[dangling])

        x_new = alpha * (
            link_contribution + dangling_mass / n
        ) + (1 - alpha) / n

        difference = np.sum(np.abs(x_new - x))
        steps.append(difference)
        x = x_new

        if difference < tol:
            return x, np.array(steps)

    raise RuntimeError("Sparse PageRank did not converge.")


# Equation: report max|mine-ref| and ||mine-ref||/||ref||
# Compare our result with a reference result.
def my_report(name, mine, ref):
    mine = np.asarray(mine, dtype=float)
    ref = np.asarray(ref, dtype=float)

    abs_err = np.max(np.abs(mine - ref))
    denominator = np.linalg.norm(ref)

    if denominator == 0:
        rel_err = np.linalg.norm(mine - ref)
    else:
        rel_err = np.linalg.norm(mine - ref) / denominator

    print(
        f"{name:32s} "
        f"max|diff| = {abs_err:.2e} "
        f"rel = {rel_err:.2e}"
    )


# Short alias for compatibility with the worked example.
report = my_report
