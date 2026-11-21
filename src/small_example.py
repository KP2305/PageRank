"""Four-page PageRank worked example, step by step (Iteration 3).

Owner: Krishnananda Das.        Run from the repository root:
    python src/small_example.py

Every step prints its intermediate result. The from-scratch functions come from
methods.py (NumPy only); exact values use fractions.Fraction; numpy.linalg.eig
is used only to CHECK the answer.

Functions expected from methods.py (signatures as in the Iteration 3 handout):
    my_out_degrees(A)               -> d                      d_j = sum_i A_ij
    my_link_matrix(A)               -> S
    my_google_matrix(A, alpha)      -> (S, G)
    my_pagerank_step(G, x)          -> G @ x
    my_power_iteration(G, tol, max_iter) -> (x, steps)        steps[k] = ||x_{k+1}-x_k||_1
    my_gauss_solve(M, b)            -> x
    report(name, mine, ref)         (or my_report)
"""
from fractions import Fraction
from math import lcm

import numpy as np

from simulate import FOUR_PAGE_LINKS, four_page_adjacency
from methods import (my_out_degrees, my_link_matrix, my_google_matrix,
                     my_pagerank_step, my_power_iteration, my_gauss_solve)
try:
    from methods import report
except ImportError:
    from methods import my_report as report

ALPHA = 0.85
ALPHA_EXACT = Fraction(17, 20)


# ---------- exact (Fraction) helpers: used only for display and exact checks ----------
def fmt(M):
    """Format a Fraction vector or matrix as text."""
    if isinstance(M[0], list):
        return "\n".join("  [" + ", ".join(f"{str(v):>9}" for v in row) + "]" for row in M)
    return "  [" + ", ".join(str(v) for v in M) + "]"


def exact_matrices(links, alpha):
    n = len(links)
    A = [[Fraction(0)] * n for _ in range(n)]
    for j, outs in links.items():
        for i in outs:
            A[i][j] = Fraction(1)
    d = [sum(A[i][j] for i in range(n)) for j in range(n)]
    S = [[A[i][j] / d[j] if d[j] > 0 else Fraction(1, n) for j in range(n)] for i in range(n)]
    G = [[alpha * S[i][j] + (1 - alpha) / n for j in range(n)] for i in range(n)]
    return A, d, S, G


def exact_matvec(G, x):
    return [sum(G[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]


def exact_gauss(M, b, verbose=True):
    """Gaussian elimination in exact fractions. The equations are first scaled by
    the lcm of all denominators so that the augmented matrix has integer entries."""
    n = len(M)
    scale = lcm(*[v.denominator for row in M for v in row], *[v.denominator for v in b])
    Mw = [[v * scale for v in row] for row in M]
    bw = [v * scale for v in b]
    if verbose:
        print(f"Multiply every equation by {scale}; augmented matrix [M | b]:")
        print(fmt([row + [bw[i]] for i, row in enumerate(Mw)]))
    for c in range(n):                         # eliminate below pivot c
        for r in range(c + 1, n):
            if Mw[r][c] != 0:
                m = Mw[r][c] / Mw[c][c]
                Mw[r] = [a - m * p for a, p in zip(Mw[r], Mw[c])]
                bw[r] -= m * bw[c]
                if verbose:
                    print(f"  R{r} <- R{r} + ({-m}) R{c}")
        if verbose and c < n - 1:
            print(f"after column {c}:")
            print(fmt([row + [bw[i]] for i, row in enumerate(Mw)]))
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):             # back substitution
        x[i] = (bw[i] - sum(Mw[i][j] * x[j] for j in range(i + 1, n))) / Mw[i][i]
    return x


def main():
    n = len(FOUR_PAGE_LINKS)
    A_ex, d_ex, S_ex, G_ex = exact_matrices(FOUR_PAGE_LINKS, ALPHA_EXACT)

    print("=== Step 1: adjacency matrix A and out-degrees d  (algorithm steps 1-2) ===")
    A = four_page_adjacency()
    d = my_out_degrees(A)
    print("A =\n", A)
    print("d =", d, "  (page 3 has d = 0: dangling)")
    report("d vs exact", d, [float(v) for v in d_ex])

    print("\n=== Step 2: link matrix S  (algorithm step 3) ===")
    S = my_link_matrix(A)
    print("S (exact) =\n" + fmt(S_ex))
    print("column sums of S =", S.sum(axis=0))
    report("S vs exact fractions", S, [[float(v) for v in r] for r in S_ex])

    print("\n=== Step 3: Google matrix G, alpha = 0.85  (algorithm step 4) ===")
    S2, G = my_google_matrix(A, ALPHA)
    print("(1 - alpha)/n =", (1 - ALPHA_EXACT) / n)
    print("G (exact) =\n" + fmt(G_ex))
    print("column sums of G =", G.sum(axis=0), "  min G =", G.min())
    report("G vs exact fractions", G, [[float(v) for v in r] for r in G_ex])

    print("\n=== Step 4: three steps of power iteration  (algorithm steps 5-6) ===")
    x = np.ones(n) / n
    x_ex = [Fraction(1, n)] * n
    xs = [x]
    print("x_0 =", x, "  exact:", fmt(x_ex))
    for k in range(1, 4):
        x = my_pagerank_step(G, x)
        x_ex = exact_matvec(G_ex, x_ex)
        xs.append(x)
        print(f"x_{k} = {np.round(x, 6)}   sum = {x.sum():.12f}")
        print(f"      exact: {fmt(x_ex)}   exact sum = {sum(x_ex)}")
        report(f"x_{k} vs exact", x, [float(v) for v in x_ex])

    print("\n=== Second-way check: Gaussian elimination, exact  (Q2) ===")
    M_ex = [[(Fraction(1) if i == j else Fraction(0)) - ALPHA_EXACT * S_ex[i][j]
             for j in range(n)] for i in range(n)]
    b_ex = [(1 - ALPHA_EXACT) / n] * n
    x_star_ex = exact_gauss(M_ex, b_ex)
    print("x* (exact) =", fmt(x_star_ex))
    print("x* (decimal) =", np.round([float(v) for v in x_star_ex], 6))
    print("sum of x* =", sum(x_star_ex))
    resid = [a - b for a, b in zip(exact_matvec(G_ex, x_star_ex), x_star_ex)]
    print("G x* - x* (exact) =", fmt(resid))
    x_star = np.array([float(v) for v in x_star_ex])

    print("\n=== My own Gaussian elimination (methods.py) vs the exact answer ===")
    x_gauss = my_gauss_solve(np.eye(n) - ALPHA * S, (1 - ALPHA) / n * np.ones(n))
    report("my_gauss_solve vs exact x*", x_gauss, x_star)

    print("\n=== Power-iteration error against x* ===")
    errs = [np.abs(xk - x_star).sum() for xk in xs]
    for k, e in enumerate(errs):
        print(f"||x_{k} - x*||_1 = {e:.4f}")
    print("ratios of successive errors:", np.round(np.array(errs[1:]) / np.array(errs[:-1]), 2),
          "  (each should be <= alpha = 0.85)")
    x_conv, steps = my_power_iteration(G, tol=1e-13, max_iter=1000)
    print(f"power iteration to tol 1e-13: {len(steps)} steps")
    report("power iteration vs exact x*", x_conv, x_star)

    print("\n=== Q4: numpy.linalg.eig (check only), normalization and sign ===")
    vals, vecs = np.linalg.eig(G)
    print("eigenvalues of G:", np.round(vals, 4))
    print("moduli:", np.round(np.abs(vals), 4))
    idx = np.argmax(vals.real)
    raw = np.real(vecs[:, idx])
    print("raw eigenvector (||v||_2 = 1):", np.round(raw, 6), "  sum =", round(raw.sum(), 6))
    v = raw / raw.sum()
    print("v / sum(v):", np.round(v, 6))
    report("eig (normalized) vs exact x*", v, x_star)
    report("eig WITHOUT normalizing vs x*", raw, x_star)
    report("eig, sign flipped, normalized", (-raw) / (-raw).sum(), x_star)


if __name__ == "__main__":
    main()
