import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_edges(path):
    return pd.read_csv(path, sep=r"\s+", comment="#",
                       names=["from", "to"], compression="infer")


def describe(df, lines):
    nodes = np.union1d(df["from"].to_numpy(), df["to"].to_numpy())
    n_dang = len(nodes) - df["from"].nunique()
    lines += [
        "=== FULL GRAPH ===",
        f"Rows (edges): {len(df):,}",
        f"Columns: {list(df.columns)}",
        f"Column types: {df.dtypes.astype(str).to_dict()}",
        f"Missing values per column: {df.isna().sum().to_dict()}",
        f"Distinct nodes appearing in the file: {len(nodes):,}",
        f"Node ID range: {nodes.min()} to {nodes.max()}",
        f"Duplicate edges: {df.duplicated().sum():,}",
        f"Self-loops: {(df['from'] == df['to']).sum():,}",
        f"Max out-degree: {df['from'].value_counts().max():,}",
        f"Max in-degree: {df['to'].value_counts().max():,}",
        f"Dangling pages in full graph: {n_dang:,} ({n_dang / len(nodes):.1%})",
        "",
    ]
    return nodes


def select_nodes(df, nodes, n_keep, rule):
    indeg = df["to"].value_counts()
    outdeg = df["from"].value_counts()

    if rule == "first":
        return nodes[:n_keep]

    if rule == "topdeg":
        total = indeg.add(outdeg, fill_value=0)
        ids, deg = total.index.to_numpy(), total.to_numpy()
        order = np.lexsort((ids, -deg))
        return np.sort(ids[order[:n_keep]])

    if rule == "bfs":
        order = np.argsort(df["from"].to_numpy(), kind="stable")
        f = df["from"].to_numpy()[order]
        t = df["to"].to_numpy()[order]
        ids, deg = indeg.index.to_numpy(), indeg.to_numpy()
        starts = ids[np.lexsort((ids, -deg))]
        seen = set()
        for s in starts:
            if len(seen) >= n_keep:
                break
            if s in seen:
                continue
            seen.add(s)
            queue = deque([s])
            while queue and len(seen) < n_keep:
                u = queue.popleft()
                lo = np.searchsorted(f, u, "left")
                hi = np.searchsorted(f, u, "right")
                for v in np.unique(t[lo:hi]):
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
                        if len(seen) >= n_keep:
                            break
        return np.array(sorted(seen))

    raise ValueError("rule must be bfs, topdeg or first")


def build_matrix(df, keep):
    sub = df[df["from"].isin(keep) & df["to"].isin(keep)].drop_duplicates()
    j = np.searchsorted(keep, sub["from"].to_numpy())
    i = np.searchsorted(keep, sub["to"].to_numpy())
    A = np.zeros((len(keep), len(keep)), dtype=np.uint8)
    A[i, j] = 1
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/web-Stanford.txt.gz")
    ap.add_argument("--n", type=int, default=3000, help="subgraph size")
    ap.add_argument("--rule", default="bfs", choices=["bfs", "topdeg", "first"])
    args = ap.parse_args()

    lines = []
    df = load_edges(args.path)
    nodes = describe(df, lines)

    keep = select_nodes(df, nodes, args.n, args.rule)
    A = build_matrix(df, keep)
    n = A.shape[0]
    out_deg = A.sum(axis=0, dtype=np.int64)
    in_deg = A.sum(axis=1, dtype=np.int64)
    dangling = int((out_deg == 0).sum())

    lines += [
        f"=== SUBGRAPH (rule: {args.rule}, n = {n}) ===",
        f"Nodes n: {n:,}",
        f"Edges m: {int(A.sum()):,}",
        f"Largest entry of A (must be 1): {A.max()}",
        f"Dangling pages (zero columns): {dangling:,} ({dangling / n:.1%})",
        f"Pages with no in-links (zero rows): {int((in_deg == 0).sum()):,}",
        f"Max out-degree: {out_deg.max():,}   Max in-degree: {in_deg.max():,}",
        f"Mean out-degree: {out_deg.mean():.2f}",
        f"Matrix density: {A.mean():.5f}",
    ]
    text = "\n".join(lines)
    print(text)
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/snap_summary_{args.rule}.txt").write_text(text + "\n")

    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, deg, name in [(ax[0], in_deg, "In-degree"),
                         (ax[1], out_deg, "Out-degree")]:
        vals, counts = np.unique(deg[deg > 0], return_counts=True)
        a.loglog(vals, counts, "o", markersize=4)
        a.set_xlabel(name + " (degree $\\geq 1$)")
        a.set_ylabel("Number of pages")
        a.set_title(f"{name} distribution (n = {n}, {args.rule})")
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"figures/degree_distribution_{args.rule}.png", dpi=150)
    print(f"Saved figures/degree_distribution_{args.rule}.png and "
          f"docs/snap_summary_{args.rule}.txt")


if __name__ == "__main__":
    main()
