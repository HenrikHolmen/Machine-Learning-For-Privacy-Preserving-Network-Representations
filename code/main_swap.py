from synthetic_generator import generate_sbm_graph
from anonymization.swap import swap_refinement
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.convert_matrix import to_numpy_array
import os
import time


# Arguments
K = 10  # Used before: 5, 10
n = 100  # Used before: 100
num_blocks = 5  # Used before: 5
p_in = 0.5
p_out = 0.05
seed = 210659

filename_suffix = f"n{n}_K{K}_seed{seed}"

os.makedirs("data/results/swap", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)
os.makedirs("data/results/plots", exist_ok=True)
os.makedirs("data/results/plots/swap", exist_ok=True)

# Generate graph
G, ground_truth_assignment_vector = generate_sbm_graph(n, num_blocks, p_in, p_out, seed)

# Visualize
color_map = [ground_truth_assignment_vector[node] for node in G.nodes()]
cmap = plt.get_cmap("tab20b", len(set(color_map)))
nx.draw_spring(G, node_color=color_map, with_labels=False, node_size=50, cmap=cmap)
plt.title(
    f"SBM Graph (n={n}, blocks={num_blocks}, p_in={p_in}, p_out={p_out}, seed={seed})"
)
plt.savefig(
    f"data/results/plots/swap/ground_truth_sbm_{filename_suffix}.png",
    dpi=500,
    bbox_inches="tight",
)
plt.close()
print(
    f"\n Log: Saved SBM plot to results/plots/swap/ground_truth_sbm_{filename_suffix}.png"
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
block_assignment_vector_anonymized, ll_trace = swap_refinement(
    A, K, seed=seed, verbose=True
)
end_time = time.time()
print(f"\n Log: Swap refinement completed in {end_time - start_time:.2f} seconds")


# Saving log-likelihood trace
np.save(
    f"data/results/swap/log_likelihood_trace_{filename_suffix}.npy", np.array(ll_trace)
)

# Save anonymized results
np.save(
    f"data/results/swap/z_anonymized_swap_{filename_suffix}.npy",
    np.array(block_assignment_vector_anonymized),
)
print(
    f"\n Log: Anonymized block assignment vector saved to data/results/swap/z_anonymized_swap_{filename_suffix}.npy"
)

# Saving runtime
with open(f"data/results/swap/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{end_time - start_time:.2f}")

# Save final block sizes
final_block_sizes = [
    block_assignment_vector_anonymized.count(b)
    for b in set(block_assignment_vector_anonymized)
]
np.save(
    f"data/results/swap/final_block_sizes_{n}_K{K}.npy",
    np.array(final_block_sizes),
)
print(
    f"\n Log: Saved final block sizes to data/results/swap/final_block_sizes_{n}_K{K}.npy"
)
