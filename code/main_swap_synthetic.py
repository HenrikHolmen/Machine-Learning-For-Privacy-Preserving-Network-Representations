from synthetic_generator import generate_sbm_graph
from anonymization.swap import swap_refinement
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.convert_matrix import to_numpy_array
import os
import time


# Arguments
K = 15
n = 100
num_blocks = 5
p_in = 0.5
p_out = 0.1
seed = 2025

filename_suffix = f"n{n}_K{K}_seed{seed}"

swap_output_dir = "data/results/synthetic/swap"
plot_dir = "data/results/synthetic/plots/swap"
adjacency_dir = "data/synthetic"

os.makedirs(swap_output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
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
    f"SBM Graph (n={n}, blocks={num_blocks}, p_in={p_in}, p_out={p_out}, seed={seed})"
)
plt.savefig(
    f"{plot_dir}/ground_truth_sbm_{filename_suffix}.png",
    dpi=500,
    bbox_inches="tight",
)
plt.close()
print(
    f"\n Log: Saved SBM plot to {plot_dir}/ground_truth_sbm_{filename_suffix}.png")

# Converting graph to adjacency matrix
A = to_numpy_array(G, dtype=int, nodelist=sorted(G.nodes()))

# Saving adjacency matrix (A) and block assignment vector (z)
np.save(f"{adjacency_dir}/adjacency_A_{filename_suffix}.npy", A)
np.save(f"{adjacency_dir}/z_assignment_{filename_suffix}.npy", np.array(ground_truth_assignment_vector))
print(f"\n Log: Saved adjacency matrix and ground truth block assignment vector to {adjacency_dir}/")

# Run selected algorithm:
start_time = time.time()
block_assignment_vector_anonymized, ll_trace = swap_refinement(
    A, K, seed=seed, verbose=True
)
end_time = time.time()
runtime = end_time - start_time
print(f"\n Log: Swap refinement completed in {runtime:.2f} seconds")

# Save anonymized results
np.save(f"{swap_output_dir}/log_likelihood_trace_{filename_suffix}.npy", np.array(ll_trace))
np.save(f"{swap_output_dir}/z_anonymized_swap_{filename_suffix}.npy", np.array(block_assignment_vector_anonymized))
with open(f"{swap_output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")

# Save final block sizes
final_block_sizes = [block_assignment_vector_anonymized.count(b) for b in set(block_assignment_vector_anonymized)]
np.save(f"{swap_output_dir}/final_block_sizes_{filename_suffix}.npy", np.array(final_block_sizes))

print(f"\n Log: All results saved to {swap_output_dir}/")