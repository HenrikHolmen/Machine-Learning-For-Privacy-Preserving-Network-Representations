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

synthetic_base = "data/results/synthetic"
plot_dir = f"{synthetic_base}/plots/merge"
merge_output_dir = f"{synthetic_base}/merge"
adjacency_dir = "data/synthetic"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(merge_output_dir, exist_ok=True)
os.makedirs(adjacency_dir, exist_ok=True)

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
    f"SBM Graph (n={n}, blocks={num_blocks}, p_in={p_in}, p_out={p_out}, seed={seed})")
plt.savefig(f"{plot_dir}/ground_truth_sbm_{filename_suffix}.png", dpi=500, bbox_inches="tight",)
plt.close()
print(f"\n Log: Saved SBM plot to {plot_dir}/ground_truth_sbm_{filename_suffix}.png")

# Converting graph to adjacency matrix
A = to_numpy_array(G, dtype=int, nodelist=sorted(G.nodes()))

# Saving adjacency matrix (A) and block assignment vector (z)
np.save(f"{adjacency_dir}/adjacency_A_{filename_suffix}.npy", A)
np.save(f"{adjacency_dir}/z_assignment_{filename_suffix}.npy", np.array(ground_truth_assignment_vector))
print("\n Log: Saved adjacency matrix and block assignment vector z to {adjacency_dir}/")

# Run selected algorithm:
start_time = time.time()
block_assignment_vector_anonymized = agglomerative_merge(A, K, verbose=True)
end_time = time.time()
runtime = end_time - start_time
print(f"\n Log: Agglomerative merge completed in {runtime:.2f} seconds")

# Save anonymized results
np.save(f"{merge_output_dir}/z_anonymized_merge_{filename_suffix}.npy", np.array(block_assignment_vector_anonymized))
with open(f"{merge_output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")
print(f"\n[✓] Results saved to {merge_output_dir}/")
