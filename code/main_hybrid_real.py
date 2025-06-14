from anonymization.hybrid import hybrid_merge_swap
from anonymization.utils import draw_colored_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import time

# Arguments
K = 15
n = 100
seed = 2025
dataset = "facebook"  # change into "ca-hepth" or "facebook" or "enron"

filename_suffix = f"n{n}_K{K}_seed{seed}"
adjacency_path = f"data/real/{dataset}/adjacency_A_{dataset}_n{n}_seed{seed}.npy"
output_dir = f"data/results/real/{dataset}/hybrid"
plot_dir = output_dir

os.makedirs(output_dir, exist_ok=True)

# Load graph
A = np.load(adjacency_path)
G = nx.from_numpy_array(A)
layout = nx.spring_layout(G, seed=seed)

# Visualize structure raw (unlabeled, pre-anonymization) 
plt.figure(figsize=(6, 6))
nx.draw(
    G,
    pos=layout,
    node_color="lightgray",
    edge_color="gray",
    node_size=50,
    width=0.4,
    alpha=1.0,
)
plt.title(f"{dataset} Sampled Subgraph (n={n}, seed={seed})")
plt.tight_layout()
plt.savefig(f"{plot_dir}/graph_structure_{filename_suffix}.png", dpi=500)
plt.close()
print(f"Saved graph structure to {plot_dir}/graph_structure_{filename_suffix}.png")

# Run hybrid algorithm
start_time = time.time()
z_merge, z_hybrid, ll_trace = hybrid_merge_swap(A, K, seed=seed, verbose=True, dataset=dataset)
end_time = time.time()
runtime = end_time - start_time
print(f"Hybrid completed in {runtime:.2f} seconds")

# Save outputs
np.save(f"{output_dir}/z_anonymized_hybrid_{filename_suffix}.npy", np.array(z_hybrid))
np.save(f"{output_dir}/log_likelihood_trace_{filename_suffix}.npy", np.array(ll_trace))
with open(f"{output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")

# Save final block sizes
final_block_sizes = [z_hybrid.count(b) for b in set(z_hybrid)]
np.save(f"{output_dir}/final_block_sizes_{filename_suffix}.npy", np.array(final_block_sizes))
print(f"Final block sizes: {Counter(z_hybrid)}")

# Visualizations
draw_colored_graph(G, z_merge, layout, plot_dir, step=f"merge_phase_{filename_suffix}")
draw_colored_graph(G, z_hybrid, layout, plot_dir, step=f"final_{filename_suffix}")
print("Saved plots for merge and hybrid steps")