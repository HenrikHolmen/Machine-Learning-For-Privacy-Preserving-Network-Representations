from synthetic_generator import generate_sbm_graph
from anonymization.hybrid import hybrid_merge_swap
from anonymization.utils import draw_colored_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.convert_matrix import to_numpy_array
from collections import Counter
import os
import time


# Arguments
K = 15  # Used before:
n = 100  # Used before:
num_blocks = 5  # Used before:
p_in = 0.5
p_out = 0.01
seed = 2025

filename_suffix = f"n{n}_K{K}_seed{seed}"

os.makedirs("data/results/hybrid", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("data/results/plots", exist_ok=True)
os.makedirs("data/results/plots/hybrid", exist_ok=True)

# Generate graph
G, ground_truth_assignment_vector = generate_sbm_graph(n, num_blocks, p_in, p_out, seed)

# Visualize
color_map = [ground_truth_assignment_vector[node] for node in G.nodes()]
cmap = plt.get_cmap("tab20b", len(set(color_map)))
# nx.draw_spring(G, node_color=color_map, with_labels=False, node_size=50, cmap=cmap)
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
    f"data/results/plots/hybrid/ground_truth_sbm_{filename_suffix}.png",
    dpi=500,
    bbox_inches="tight",
)
plt.close()
print(
    f"\n Log: Saved SBM plot to results/plots/hybrid/ground_truth_sbm_{filename_suffix}.png"
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
z_merge, z_hybrid, ll_trace = hybrid_merge_swap(A, K, seed=seed, verbose=True)
end_time = time.time()
block_assignment_vector_anonymized = z_hybrid
print(f"\n Log: Hybrid completed in {end_time - start_time:.2f} seconds")

# save trace
np.save(
    f"data/results/hybrid/log_likelihood_trace_{filename_suffix}.npy",
    np.array(ll_trace),
)

# Save anonymized results
np.save(
    f"data/results/hybrid/z_anonymized_hybrid_{filename_suffix}.npy",
    np.array(block_assignment_vector_anonymized),
)
print(
    f"\n Log: Anonymized block assignment vector saved to data/results/hybrid/z_anonymized_hybrid_{filename_suffix}.npy"
)

# Saving runtime
with open(f"data/results/hybrid/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{end_time - start_time:.2f}")

# Save final block sizes
final_block_sizes = [
    block_assignment_vector_anonymized.count(b)
    for b in set(block_assignment_vector_anonymized)
]
np.save(
    f"data/results/hybrid/final_block_sizes_{n}_K{K}.npy",
    np.array(final_block_sizes),
)
print(f"\n Log: Final block sizes: {Counter(block_assignment_vector_anonymized)}")

# Construct layout once
G = nx.from_numpy_array(A)
layout = nx.spring_layout(G, seed=seed)

# After merge phase
draw_colored_graph(
    G,
    z_merge,
    layout,
    "data/results/plots/hybrid",
    step=f"merge_phase_{filename_suffix}",
)

# After hybrid refinement
draw_colored_graph(
    G,
    z_hybrid,
    layout,
    "data/results/plots/hybrid",
    step=f"final_{filename_suffix}",
)

print("Log: Saved plot for merge and final hybrid steps.")
