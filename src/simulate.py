"""Data and simulation helpers for the PageRank project (Iteration 3).

Owner: Krishnananda Das.

Everything here *generates data or the known truth*; none of it is the method
under test. The Google matrix is therefore built with its own small private
function, so that the from-scratch functions in methods.py can be compared
against an independent construction.

Used by:  small_example.py, check_simulation.py, run_real.py
"""
import numpy as np

# page -> pages it links to (page j links to page i  ==>  A[i, j] = 1)
SIX_PAGE_LINKS = {0: [1, 2], 1: [2], 2: [0, 3], 3: [4], 4: [0, 3], 5: []}   # Iteration 2
FOUR_PAGE_LINKS = {0: [1, 2], 1: [2, 3], 2: [0], 3: []}                      # worked example


def adjacency_from_links(links):
    """A_ij = 1 if page j links to page i, else 0 (column j = out-links of j)."""
    n = len(links)
    A = np.zeros((n, n))
    for j, outs in links.items():
        for i in outs:
            A[i, j] = 1.0
    return A


def six_page_adjacency():
    """The Iteration 2 six-page web; page 5 is dangling."""
    return adjacency_from_links(SIX_PAGE_LINKS)


def four_page_adjacency():
    """The four-page worked example; page 3 is dangling."""
    return adjacency_from_links(FOUR_PAGE_LINKS)


def _google_matrix(A, alpha):
    """Independent construction of S and G (used only to generate truth/data).

    S_ij = A_ij / d_j (dangling column -> 1/n),  G = alpha S + (1 - alpha)/n 11^T
    """
    n = A.shape[0]
    d = A.sum(axis=0)
    S = np.where(d > 0, A / np.where(d == 0, 1, d), 1.0 / n)
    G = alpha * S + (1.0 - alpha) / n * np.ones((n, n))
    return S, G


def six_page_truth(alpha=0.85):
    """Exact PageRank of the six-page web from the linear system
    (I - alpha S) x = (1 - alpha)/n 1.   (numpy.linalg.solve: check only)
    """
    A = six_page_adjacency()
    n = A.shape[0]
    S, _ = _google_matrix(A, alpha)
    x = np.linalg.solve(np.eye(n) - alpha * S, (1.0 - alpha) / n * np.ones(n))
    return x


def simulate_surfer(n_steps=100_000, seed=0, alpha=0.85, start=0):
    """Random-surfer simulation on the six-page web (Iteration 2 simulation).

    The surfer starts on page `start` and at each step moves to page i with
    probability G[i, current page]. Returns (A, frequencies, x_true) where
    `frequencies` are the visit fractions after n_steps and `x_true` is the
    exact PageRank vector, so the two can be compared.
    """
    A = six_page_adjacency()
    n = A.shape[0]
    _, G = _google_matrix(A, alpha)
    rng = np.random.default_rng(seed)
    visits = np.zeros(n, dtype=int)
    page = start
    for _ in range(n_steps):
        page = rng.choice(n, p=G[:, page])
        visits[page] += 1
    return A, visits / n_steps, six_page_truth(alpha)


if __name__ == "__main__":
    A, freq, x_true = simulate_surfer(100_000, seed=0)
    print("Six-page adjacency matrix A:")
    print(A)
    print("\nExact PageRank (linear system):", np.round(x_true, 6))
    print("Visit frequencies (100,000 steps, seed 0):", np.round(freq, 6))
    print("max |frequency - truth| =", f"{np.abs(freq - x_true).max():.2e}")
