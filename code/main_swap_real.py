from anonymization.swap import swap_refinement
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import time

#    Parameters   
dataset = "facebook"  # "ca-hepth" or "facebook" or "enron"
n = 200
K = 5
seed = 210659
filename_suffix = f"n{n}_K{K}_seed{seed}"

#    Paths   
input_path = f"data/real/{dataset}/adjacency_A_{dataset}_n{n}_seed{seed}.npy"
output_dir = f"data/results/real/{dataset}/swap"
os.makedirs(output_dir, exist_ok=True)

#    Load adjacency matrix   
A = np.load(input_path)
print(f"\n Log Loaded adjacency matrix from: {input_path}")

#    Visualize raw graph (unlabeled, pre-anonymization) 
G = nx.from_numpy_array(A)
layout = nx.spring_layout(G, seed=seed)  # fixed layout for consistency

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
plt.savefig(f"{output_dir}/graph_structure_{filename_suffix}.png", dpi=500)
plt.close()
print(f"Saved structure plot to: {output_dir}/graph_structure_{filename_suffix}.png")

#  Run swap refinement   
start_time = time.time()
block_assignment_vector_anonymized, ll_trace = swap_refinement(A, K, seed=seed, verbose=True)
end_time = time.time()
runtime = end_time - start_time
print(f"\n Swap refinement completed in {runtime:.2f} seconds")

#    Save results   
np.save(f"{output_dir}/log_likelihood_trace_{filename_suffix}.npy", np.array(ll_trace))
np.save(f"{output_dir}/z_anonymized_swap_{filename_suffix}.npy", np.array(block_assignment_vector_anonymized))
with open(f"{output_dir}/runtime_{filename_suffix}.txt", "w") as f:
    f.write(f"{runtime:.2f}")

final_block_sizes = [block_assignment_vector_anonymized.count(b) for b in set(block_assignment_vector_anonymized)]
np.save(f"{output_dir}/final_block_sizes_{filename_suffix}.npy", np.array(final_block_sizes))
print(f"All results saved to: {output_dir}/")