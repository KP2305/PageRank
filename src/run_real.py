
import sys
import os
import numpy as np

sys.path.insert(0, "/content/Iteration#3")

import methods


# --------------------------------------------------
# 1. Load the saved BFS subgraph
# --------------------------------------------------

data_path = "/content/Iteration#3/data/bfs_subgraph.npz"

data = np.load(data_path, allow_pickle=True)

nodes = data["nodes"]
out_links_array = data["out_links"]

n = len(nodes)

# Convert saved out-links into a list of local indices
# expected by my_pagerank_outlinks().
# Map original SNAP page IDs to local indices (0 to n-1)
node_to_index = {
    int(node): i
    for i, node in enumerate(nodes)
}

# Convert original page IDs into local indices
out_links = [
    [node_to_index[int(target)] for target in out_links_array[i]]
    for i in range(n)
]

# Verify that every destination belongs to the subgraph
assert all(
    int(target) in node_to_index
    for links in out_links_array
    for target in links
), "Some destinations are outside the extracted subgraph."

print("SNAP WEB-STANFORD REAL DATA")
print("--------------------------------")

print("Number of pages:", n)

edge_count = sum(len(v) for v in out_links)
dangling_count = sum(len(v) == 0 for v in out_links)

print("Number of edges:", edge_count)
print("Dangling pages:", dangling_count)
print("Dangling percentage:", round(100 * dangling_count / n, 2), "%")


# --------------------------------------------------
# 2. Run PageRank using sparse out-link method
# --------------------------------------------------

alpha = 0.85

pagerank, errors = methods.my_pagerank_outlinks(
    out_links,
    alpha=alpha,
    tol=1e-12,
    max_iter=100000
)

print("\nPAGERANK RESULTS")
print("--------------------------------")

print("Alpha:", alpha)
print("Number of iterations:", len(errors))
print("Sum of PageRank:", pagerank.sum())
print("Minimum PageRank:", pagerank.min())
print("Maximum PageRank:", pagerank.max())
print("Final convergence error:", errors[-1])


# --------------------------------------------------
# 3. Display top 10 ranked pages
# --------------------------------------------------

top_indices = np.argsort(pagerank)[::-1][:10]

print("\nTOP 10 PAGES")
print("--------------------------------")

for rank, idx in enumerate(top_indices, start=1):
    print(
        f"Rank {rank}: Page ID {nodes[idx]}, "
        f"PageRank = {pagerank[idx]:.10f}"
    )


# --------------------------------------------------
# 4. Verify basic numerical properties
# --------------------------------------------------

print("\nNUMERICAL CHECKS")
print("--------------------------------")

print("All scores are nonnegative:",
      np.all(pagerank >= 0))

print("PageRank sums to 1:",
      np.isclose(pagerank.sum(), 1.0, atol=1e-10))

print("All scores are finite:",
      np.all(np.isfinite(pagerank)))

print("Convergence reached:",
      errors[-1] <= 1e-12)
