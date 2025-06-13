import networkx as nx
import numpy as np


def generate_sbm_graph(n, num_blocks, p_in, p_out, seed=None):
    """
    Arguments:
        n (int): Total number of nodes.
        num_blocks (int): Number of ground-truth blocks in the synthetic graph - useful for testing if the algorithms preserve it.
        p_in (float): Intra-block connection probability (B_diagonal).
        p_out (float): Inter-block connection probability (B_off-diagonal).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        G (networkx.Graph): Generated undirected graph.
        z (list): List of ground truth, block assignments (z) (length n).
    """

    if seed is not None:
        np.random.seed(seed)

    block_size = [n // num_blocks] * num_blocks
    for index in range(n % num_blocks):
        block_size[index] += 1  # Distribute remaining blocks into the first few blocks

    probs = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(p_in)
            else:
                row.append(p_out)
        probs.append(row)

    G = nx.stochastic_block_model(block_size, probs, seed=seed)

    z = []
    for block_id, size in enumerate(block_size):
        z.extend([block_id] * size)

    return G, z
