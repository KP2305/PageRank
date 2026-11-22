"""Assumption checks on the simulated six-page graph (Iteration 3, step 6).

Owner: Krishnananda Das.        Run from the repository root:
    python src/check_assumptions_sim.py

Runs each of the eight Iteration 2 assumptions on the six-page web, where they
hold by construction, so every check should PASS. This shows what "passing"
looks like before the real data is examined. The numbers printed here are the
ones in the "Assumption Checks" table of the report.

Uses from methods.py:  my_google_matrix(A, alpha) -> (S, G),
                       my_pagerank_step(G, x) -> G @ x
numpy.linalg.eig is used only as the reference answer.
"""
import numpy as np

from simulate import SIX_PAGE_LINKS, six_page_adjacency, simulate_surfer
from methods import my_google_matrix, my_pagerank_step

ALPHA = 0.85
TOL = 1e-13          # stopping rule for the convergence check: ||x_{k+1}-x_k||_1 < TOL
RESULTS = []


def verdict(no, name, ok, detail):
    RESULTS.append(ok if ok is not None else None)
    tag = "N/A " if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {no}. {name}\n        {detail}")


def power_from(G, x0, tol=TOL, max_iter=5000):
    """Power iteration from a given start vector; returns (x, steps, max |sum-1|)."""
    x, worst = x0.copy(), 0.0
    for k in range(1, max_iter + 1):
        x_new = my_pagerank_step(G, x)
        worst = max(worst, abs(x_new.sum() - 1.0))
        done = np.abs(x_new - x).sum() < tol
        x = x_new
        if done:
            return x, k, worst
    return x, max_iter, worst


def main():
    A = six_page_adjacency()
    n = A.shape[0]
    S, G = my_google_matrix(A, ALPHA)
    out = A.sum(axis=0)

    # reference: eigenvector of G for eigenvalue 1, normalized to sum 1 (check only)
    vals, vecs = np.linalg.eig(G)
    v = np.real(vecs[:, np.argmax(vals.real)])
    v = v / v.sum()

    print(f"Six-page web: n = {n}, alpha = {ALPHA}\n")

    # 1. A is a 0/1 matrix (no repeated links)
    no_dups = all(len(t) == len(set(t)) for t in SIX_PAGE_LINKS.values())
    verdict(1, "A is a 0/1 matrix", no_dups and A.max() == 1.0 and set(np.unique(A)) <= {0.0, 1.0},
            f"repeated targets: {not no_dups}; max A_ij = {A.max():.0f}")

    # 2. dangling pages redirected uniformly: column sums of S equal 1
    dev_S = np.abs(S.sum(axis=0) - 1).max()
    verdict(2, "Dangling pages redirected uniformly", int((out == 0).sum()) == 1 and dev_S < 1e-12,
            f"dangling pages: {int((out == 0).sum())} (page {int(np.argmin(out))}); "
            f"max |column sum of S - 1| = {dev_S:.1e}")

    # 3. 0 < alpha < 1, all entries of G positive
    verdict(3, "0 < alpha < 1 and G > 0",
            0 < ALPHA < 1 and abs(G.min() - (1 - ALPHA) / n) < 1e-12,
            f"alpha = {ALPHA}; min G = {G.min():.4f}; (1-alpha)/n = {(1 - ALPHA) / n:.4f}")

    # 4. uniform teleportation: G - alpha S is the constant matrix (1-alpha)/n
    dev_T = np.abs((G - ALPHA * S) - (1 - ALPHA) / n).max()
    verdict(4, "Uniform teleportation", dev_T < 1e-12,
            f"max |(G - alpha S) - (1-alpha)/n| = {dev_T:.1e}")

    # 5. Markov property / time-homogeneity: simulated surfer matches the stationary vector
    _, freq, x_true = simulate_surfer(100_000, seed=0, alpha=ALPHA)
    gap = np.abs(freq - x_true).max()
    bound = (3 * np.sqrt(x_true * (1 - x_true) / 100_000)).max()
    verdict(5, "Markov property / time-homogeneity", gap < bound,
            f"100,000 steps, seed 0: max |visit frequency - x*| = {gap:.1e}; "
            f"3*sqrt(p(1-p)/N) <= {bound:.1e}  (rough guide: steps are correlated)")

    # 6. convergence to a unique vector, whatever the start
    starts = {"e_0": np.eye(n)[0], "e_5": np.eye(n)[5], "uniform": np.ones(n) / n,
              "skewed": np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])}
    finals, steps_used, worst = {}, {}, 0.0
    for name, x0 in starts.items():
        finals[name], steps_used[name], w = power_from(G, x0)
        worst = max(worst, w)
    pair = max(np.abs(a - b).sum() for a in finals.values() for b in finals.values())
    to_eig = max(np.abs(x - v).sum() for x in finals.values())
    verdict(6, "Convergence to a unique PageRank vector", pair < 1e-10 and to_eig < 1e-10,
            f"steps to tol {TOL:g} from e_0, e_5, uniform, skewed: "
            f"{', '.join(str(steps_used[k]) for k in starts)}; "
            f"max pairwise l1 distance = {pair:.1e}; max l1 distance to eigenvector = {to_eig:.1e}")

    # 7. the subgraph represents the web: not applicable to the six-page web
    verdict(7, "Subgraph represents the web", None,
            f"not applicable (the six-page graph is the whole web); dangling fraction = "
            f"{int((out == 0).sum())}/{n} = {(out == 0).mean():.1%}; checked on the real data")

    # 8. floating-point accuracy: sums within 1e-12 of 1
    verdict(8, "Floating-point arithmetic accurate enough",
            max(worst, dev_S, np.abs(G.sum(axis=0) - 1).max()) < 1e-12,
            f"max |1^T x_k - 1| over every step of all four runs = {worst:.1e}; "
            f"max |column sum of G - 1| = {np.abs(G.sum(axis=0) - 1).max():.1e}  (limit 1e-12)")

    mods = np.sort(np.abs(vals))[::-1]
    print(f"\nsecond-largest |eigenvalue| of G = {mods[1]:.3f}  (alpha/2 = {ALPHA / 2:.3f}; bound alpha = {ALPHA})")
    ran = [r for r in RESULTS if r is not None]
    print(f"{sum(ran)} of {len(ran)} applicable checks passed; "
          f"{RESULTS.count(None)} not applicable.")


if __name__ == "__main__":
    main()
