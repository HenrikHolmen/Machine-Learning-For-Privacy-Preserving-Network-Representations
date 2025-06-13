import os
import networkx as nx
import numpy as np
from random import random, seed


def forest_fire_sampling(
    G, target_size=target_size, fw_prob=fw_prob, seed_value=seed_value
):
    """
    Forest Fire sampling algorithm to extract a representative subgraph.

    Parameters:
    - G (networkx.Graph): Input graph (undirected).
    - target_size (int): Number of nodes in the sampled subgraph.
    - fw_prob (float): Forward burning probability (0 < fw_prob < 1).
    - seed_value (int): Random seed for reproducibility.

    Returns:
    - sampled_subgraph (networkx.Graph): Subgraph induced by sampled nodes.
    - sampled_nodes (list): List of node IDs in the sampled graph.
    """
    seed(seed_value)
    np.random.seed(seed_value)
    visited = set()
    nodes = list(G.nodes())
    start_node = np.random.choice(nodes)
    frontier = [start_node]
    visited.add(start_node)

    while len(visited) < target_size and frontier:
        current = frontier.pop()
        neighbors = list(set(G.neighbors(current)) - visited)
        np.random.shuffle(neighbors)
        num_to_burn = np.random.geometric(1 - fw_prob)
        to_burn = neighbors[:num_to_burn]
        for nbr in to_burn:
            if len(visited) >= target_size:
                break
            if nbr not in visited:
                visited.add(nbr)
                frontier.append(nbr)

    sampled_nodes = list(visited)
    sampled_subgraph = G.subgraph(sampled_nodes).copy()
    return sampled_subgraph, sampled_nodes


if __name__ == "__main__":
    dataset_name = "ca-hepth"  # change to "enron" or "facebook" when needed
    input_path = f"real_data/datasets/{dataset_name}/processed_data.txt"
    output_dir = f"real_data/datasets/{dataset_name}"

    # Load raw or processed graph
    G_full = nx.read_edgelist(input_path, nodetype=int)
    print(
        f"[INFO] Loaded graph with {G_full.number_of_nodes()} nodes and {G_full.number_of_edges()} edges."
    )

    # Parameters
    target_size = 100
    seed_value = 210659
    fw_prob = 0.7

    # Run Forest Fire sampling
    G_sampled, sampled_nodes = forest_fire_sampling(
        G_full, target_size=target_size, fw_prob=fw_prob, seed_value=seed_value
    )
    print(f"[INFO] Sampled {len(sampled_nodes)} nodes.")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    nx.write_edgelist(
        G_sampled,
        f"{output_dir}/sampled_edges_n{target_size}_seed{seed_value}.txt",
        data=False,
    )

    with open(
        f"{output_dir}/sampled_nodes_n{target_size}_seed{seed_value}.txt", "w"
    ) as f:
        for node in sampled_nodes:
            f.write(f"{node}\n")

    print(f"[INFO] Saved sampled edges and node list to {output_dir}")
