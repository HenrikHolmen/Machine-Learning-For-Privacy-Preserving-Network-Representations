import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from collections import defaultdict


def compute_block_probability_matrix(A, z, num_blocks=None):
    """
    Computes the block probability matrix (B) given the current assignment vector z.

    Parameters:
        A (np.ndarray): Adjacency matrix
        z (list): Block assignment vector
        num_blocks (int): If known, force number of blocks

    Returns:
        B (np.ndarray): Block probability matrix
        N (np.ndarray): Block size (num_blocks x 1)
    """

    n = len(z)
    if num_blocks is None:
        num_blocks = max(z) + 1

    B = np.zeros((num_blocks, num_blocks))
    counts = np.zeros((num_blocks, num_blocks))

    for i in range(n):
        for j in range(n):
            l, m = z[i], z[j]
            B[l][m] += A[i][
                j
            ]  # accumulates the number of edges between block l and m (N_lm^+)
            counts[l][m] += 1  # counts the total number of dyads N_lm^+ + N_lm^-

    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.divide(
            B, counts, out=np.zeros_like(B), where=counts != 0
        )  # and this is there its divide Eq 3.13 in report

    return B, counts


def compute_log_likelihood(A, z, B):
    """
    Computes (scalar form) log-likelihood as described in report Equation 3.5:
        \log P(\mathbf{A}|z, \mathbf{B}) = \sum_{i<j} \mathbf{A}_{ij} \log (\mathbf{B}_{z_{i}z_{j}}) + (1-\mathbf{A}_{ij}) \log (1-\mathbf{B}_{z_{i}z_{j}})

    Parameters:
        A (np.ndarray): Adjacency matrix
        z (list): Block assignment vector
        B (np.ndarray): Block probability matrix

    Returns:
        total_log_likelihood (float): Total log-likelihood
    """

    total_log_likelihood = 0.0
    n = len(z)

    for i in range(n):
        for j in range(n):
            if i == j:  # If node i and node j are in the same block
                continue
            p_ij = B[z[i]][z[j]]  # Block assignments -> l,m
            a_ij = A[i, j]

            # Maintain numerical stability convention 0* \log(0) = 0  (Report page 17)
            if p_ij > 0:
                total_log_likelihood += a_ij * np.log(p_ij)
            if p_ij < 1:
                total_log_likelihood += (1 - a_ij) * np.log(1 - p_ij)

    return total_log_likelihood


def find_best_merge(A, block_assignment_vector, candidate_pairs):
    """
    Finds the best merge (with minimal Δ log-likelihood) among candidate block pairs.

    Parameters:
        A (np.ndarray): Adjacency matrix (n x n)
        block_assignment_vector (list): Current block assignment vector z of length n
        candidate_pairs (list of tuples): List of (block_l, block_m) pairs to consider merging

    Returns:
        best_pair (tuple): The (block_l, block_m) pair that yields the lowest Δ log-likelihood
        min_delta (float): The corresponding change in log-likelihood
    """

    min_delta = float("inf")
    best_pair = None

    for block_l, block_m in candidate_pairs:
        z_candidate = block_assignment_vector.copy()
        for i in range(len(block_assignment_vector)):
            if z_candidate[i] == block_m:
                z_candidate[i] = block_l  # simulate merging block m into block l

        B_candidate, _ = compute_block_probability_matrix(A, z_candidate)
        B_current, _ = compute_block_probability_matrix(A, block_assignment_vector)

        total_log_likelihood_candidate = compute_log_likelihood(
            A, z_candidate, B_candidate
        )
        total_log_likelihood_current = compute_log_likelihood(
            A, block_assignment_vector, B_current
        )

        delta_log_likelihood = (
            total_log_likelihood_candidate - total_log_likelihood_current
        )

        if delta_log_likelihood < min_delta:
            min_delta = delta_log_likelihood
            best_pair = (block_l, block_m)

    return best_pair, min_delta


def draw_colored_graph(G, z, layout, save_path, step=None):
    """
    Draw and save a graph with nodes colored by block assignment.

    Parameters:
    - G (networkx.Graph): The input graph.
    - z (list): Block assignment vector for each node.
    - layout (dict): Fixed layout for node positions (e.g., spring layout).
    - save_path (str): Directory where to save the figure.
    - step (int): Optional step number to include in filename.
    """
    os.makedirs(save_path, exist_ok=True)

    color_map = [z[node] for node in G.nodes()]
    num_blocks = len(set(color_map))
    cmap = plt.get_cmap(
        "tab20b", num_blocks
    )  # Using the tab20b so there is 20 different colormaps, but also different shaps
    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        pos=layout,
        node_color=color_map,
        with_labels=False,
        node_size=50,
        cmap=cmap,
        edge_color="black",
        width=0.6,
        alpha=1.0,
    )

    filename = f"step_{step}.png" if step is not None else "snapshot.png"
    plt.title(f"Merge Step {step}" if step is not None else "Graph Snapshot")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=500)
    plt.close()
