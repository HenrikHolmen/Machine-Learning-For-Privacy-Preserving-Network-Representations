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

hybrid_output_dir = "data/results/synthetic/hybrid"
plot_dir = "data/results/synthetic/plots/hybrid"
adjacency_dir = "data/synthetic"
os.makedirs(hybrid_output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(adjacency_dir, exist_ok=True)

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

plt.savefig(f"{plot_dir}/ground_truth_sbm_{filename_suffix}.png", dpi=500, bbox_inches="tight")

plt.close()
print(f"\n Log: Saved SBM plot to {plot_dir}/ground_truth_sbm_{filename_suffix}.png")

# Converting graph to adjacency matrix
A = to_numpy_array(G, dtype=int, nodelist=sorted(G.nodes()))

# Saving adjacency matrix (A) and block assignment vector (z)
np.save(f"{adjacency_dir}/adjacency_A_{filename_suffix}.npy", A)
np.save(f"{adjacency_dir}/z_assignment_{filename_suffix}.npy", np.array(ground_truth_assignment_vector))
print(f"\n Log: Saved adjacency matrix and ground truth block assignment vector to {adjacency_dir}/")

# Run selected algorithm:
start_time = time.time()
z_merge, z_hybrid, ll_trace = hybrid_merge_swap(A, K, seed=seed, verbose=True)
end_time = time.time()
runtime = end_time - start_time
block_assignment_vector_anonymized = z_hybrid
print(f"\n Log: Hybrid completed in {runtime:.2f} seconds")


# Save anonymized results
np.save(f"{hybrid_output_dir}/log_likelihood_trace_{filename_suffix}.npy", np.array(ll_trace))
np.save(f"{hybrid_output_dir}/z_anonymized_hybrid_{filename_suffix}.npy", np.array(z_hybrid))
with open(f"{hybrid_output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")
print(f"\n Log: Anonymized block assignment vector saved to {hybrid_output_dir}/z_anonymized_hybrid_{filename_suffix}.npy")

# Save final block sizes
final_block_sizes = [z_hybrid.count(b) for b in set(z_hybrid)]
np.save(f"{hybrid_output_dir}/final_block_sizes_{filename_suffix}.npy", np.array(final_block_sizes))

print(f"\n Log: Final block sizes: {Counter(z_hybrid)}")

# Redraw the graph for intermediate + final results
G = nx.from_numpy_array(A)
layout = nx.spring_layout(G, seed=seed)
draw_colored_graph(G, z_merge, layout, plot_dir, step=f"merge_phase_{filename_suffix}")
draw_colored_graph(G, z_hybrid, layout, plot_dir, step=f"final_{filename_suffix}")

print("Log: Saved plots for merge and final hybrid steps.")