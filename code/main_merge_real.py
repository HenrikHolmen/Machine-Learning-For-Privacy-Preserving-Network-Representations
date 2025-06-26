from anonymization.merge import agglomerative_merge
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import time

# Arguments
K = 15
n = 100
seed = 210659
dataset = "ca-hepth"  # change to "ca-hepth" or "facebook" or "enron"

filename_suffix = f"n{n}_K{K}_seed{seed}"
input_path = f"data/real/{dataset}/adjacency_A_{dataset}_n{n}_seed{seed}.npy"
output_dir = f"data/results/real/{dataset}/merge"
plot_dir = output_dir  # optionally keep separate plot folder

os.makedirs(output_dir, exist_ok=True)

# Load adjacency matrix
A = np.load(input_path)
print(f"Loaded adjacency matrix from {input_path}")

# Visualize graph raw (unlabeled, pre-anonymization) 
G = nx.from_numpy_array(A)
layout = nx.spring_layout(G, seed=seed)

plt.figure(figsize=(6, 6))
nx.draw(
    G,
    pos=layout,
    node_color="lightgray",
    edge_color="gray",
    node_size=50,
    width=0.4,
    alpha=1,
)
plt.title(f"{dataset} Sampled Subgraph (n={n}, seed={seed})")
plt.tight_layout()
plt.savefig(f"{plot_dir}/graph_structure_{filename_suffix}.png", dpi=500)
plt.close()
print(f"Saved graph plot to {plot_dir}/graph_structure_{filename_suffix}.png")

# Run merge algorithm
start_time = time.time()
block_assignment_vector_anonymized = agglomerative_merge(A, K, verbose=True, seed=seed, dataset=dataset)
end_time = time.time()
runtime = end_time - start_time
print(f"Agglomerative merge completed in {runtime:.2f} seconds")

# Save anonymized results
np.save(f"{output_dir}/z_anonymized_merge_{filename_suffix}.npy", np.array(block_assignment_vector_anonymized))
with open(f"{output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")

# Save final block sizes
final_block_sizes = [block_assignment_vector_anonymized.count(b) for b in set(block_assignment_vector_anonymized)]
np.save(f"{output_dir}/final_block_sizes_{filename_suffix}.npy", np.array(final_block_sizes))

print(f"All results saved in {output_dir}/")