from synthetic_generator import generate_sbm_graph
from anonymization.merge import agglomerative_merge
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.convert_matrix import to_numpy_array
import os
import time


# Arguments
K = 15  # Tried 5
n = 100  # Tried 100
num_blocks = 5
p_in = 0.5
p_out = 0.01
seed = 2025

filename_suffix = f"n{n}_K{K}_seed{seed}"

os.makedirs("data/results/plots", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("data/results/merge", exist_ok=True)
os.makedirs("data/results/plots/merge", exist_ok=True)

# Generate graph
G, ground_truth_assignment_vector = generate_sbm_graph(n, num_blocks, p_in, p_out, seed)

# Visualize
color_map = [ground_truth_assignment_vector[node] for node in G.nodes()]
cmap = plt.get_cmap("tab20b", len(set(color_map)))
layout = nx.spring_layout(G, seed=seed)
nx.draw(
    G,
    pos=layout,
    node_color=color_map,
    with_labels=False,
    node_size=50,
    cmap=cmap,
    edge_color="gray",
    width=0.4,
    alpha=1.0,
    edgecolors="black",
    linewidths=0.3,
)
plt.title(
    f"SBM Graph (n={n}, blocks={num_blocks}, p_in={p_in}, p_out={p_out}, seed={seed})"
)
plt.savefig(
    f"data/results/plots/merge/ground_truth_sbm_{filename_suffix}.png",
    dpi=500,
    bbox_inches="tight",
)
plt.close()
print(
    f"\n Log: Saved SBM plot to results/plots/merge/ground_truth_sbm_{filename_suffix}.png"
)


# Converting graph to adjacency matrix
A = to_numpy_array(G, dtype=int, nodelist=sorted(G.nodes()))

# Saving adjacency matrix (A) and block assignment vector (z)
np.save(f"data/synthetic/adjacency_A_{filename_suffix}.npy", A)
np.save(
    f"data/synthetic/z_assignment_{filename_suffix}.npy",
    np.array(ground_truth_assignment_vector),
)
print("\n Log: Saved adjacency matrix A and ground truth block assignment vector z")

# Run selected algorithm:
start_time = time.time()
block_assignment_vector_anonymized = agglomerative_merge(A, K, verbose=True)
end_time = time.time()
print(f"\n Log: Agglomerative merge completed in {end_time - start_time:.2f} seconds")

# Save anonymized results
np.save(
    f"data/results/merge/z_anonymized_merge_{filename_suffix}.npy",
    np.array(block_assignment_vector_anonymized),
)
print(
    f"\n Log: Anonymized block assignment vector saved to data/results/merge/z_anonymized_merge_{filename_suffix}.npy"
)

# Saving runtime
with open(f"data/results/merge/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{end_time - start_time:.2f}")
