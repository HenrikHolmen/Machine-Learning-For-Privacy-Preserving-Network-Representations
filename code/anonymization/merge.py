from anonymization.utils import (
    compute_log_likelihood,
    compute_block_probability_matrix,
    find_best_merge,
    draw_colored_graph,
)
from collections import defaultdict
from itertools import combinations
import numpy as np
import networkx as nx
import os


def agglomerative_merge(A, K, verbose=False, seed=None, dataset=None):
    """
    Runs the Agglomerative Merge Algorithm to enforce K-anonymity.

    Parameters:
        A (np.ndarray): Adjacency matrix (n x n)
        K (int): Minimum block size (K-constrain)
        verbose (bool): If True, print debug information

    Return:
        block_assignment_vector_anonymized (list): Final block assignment vector z of length n, after anonymization.
    """
    G = nx.from_numpy_array(A)
    layout = nx.spring_layout(G, seed=seed)
    num_nodes = A.shape[0]
    block_assignment_vector = list(
        range(num_nodes)
    )  # Initialize each node in its own singleton block

    if verbose:
        print(f"Initialized {num_nodes} singleton blocks.")

    if dataset is not None:
        snapshot_dir = f"data/results/real/{dataset}/merge/snapshots/n{num_nodes}_K{K}"
        merge_output_dir = f"data/results/real/{dataset}/merge"
    else:
        snapshot_dir = f"data/results/synthetic/plots/merge/snapshots/n{num_nodes}_K{K}"
        merge_output_dir = "data/results/synthetic/merge"
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(merge_output_dir, exist_ok=True)

    merge_step = 0  # Tracking how many merges are performed
    log_likelihood_trace = []  # Keep tract on how the log-likelihood changes over time
    B_init, _ = compute_block_probability_matrix(A, block_assignment_vector)
    initial_log_likelihood = compute_log_likelihood(A, block_assignment_vector, B_init)
    log_likelihood_trace.append(initial_log_likelihood)

    # MAIN MERGE LOOP
    while True:
        merge_step += 1

        # Log current total log-likelihood (before merge)

        block_to_nodes = defaultdict(list)
        for node, block in enumerate(block_assignment_vector):
            block_to_nodes[block].append(node)

        small_blocks = [
            block for block, nodes in block_to_nodes.items() if len(nodes) < K
        ]
        if len(small_blocks) < 2:
            break  # No valid merges left

        candidate_pairs = list(combinations(small_blocks, 2))
        best_pair, min_delta = find_best_merge(
            A, block_assignment_vector, candidate_pairs
        )

        if best_pair is None:
            break  # No valid merge found

        block_l_star, block_m_star = best_pair
        for i in range(num_nodes):
            if (
                block_assignment_vector[i] == block_m_star
            ):  # if node i is in block m* we merge the node(s) in m* into new block l*
                block_assignment_vector[i] = block_l_star

        B_current, _ = compute_block_probability_matrix(A, block_assignment_vector)
        current_log_likelihood = compute_log_likelihood(
            A, block_assignment_vector, B_current
        )
        log_likelihood_trace.append(current_log_likelihood)

        if merge_step % 5 == 0 and merge_step > 0:
            draw_colored_graph(
                G, block_assignment_vector, layout, snapshot_dir, step=merge_step
            )

        if verbose:
            print(
                f"Merge blocks {block_l_star} and {block_m_star} with Δ log-likelihood = {min_delta:.4f}"
            )

    # FALLBACK STEP: Force-merge remaining small blocks
    block_to_nodes = defaultdict(list)
    for node, block in enumerate(block_assignment_vector):
        block_to_nodes[block].append(node)

    remaining_small_blocks = [
        block for block, nodes in block_to_nodes.items() if len(nodes) < K
    ]

    for small_block in remaining_small_blocks:
        candidate_blocks = [
            block
            for block in block_to_nodes
            if block != small_block and len(block_to_nodes[block]) >= K
        ]
        candidate_pairs = [
            (target_block, small_block) for target_block in candidate_blocks
        ]

        best_pair, min_delta = find_best_merge(
            A, block_assignment_vector, candidate_pairs
        )

        if best_pair is not None:
            block_l_star, block_m_star = best_pair
            for i in range(num_nodes):
                if block_assignment_vector[i] == block_m_star:
                    block_assignment_vector[i] = block_l_star
            merge_step += 1

            if verbose:
                print(
                    f"Force-merged small block {block_m_star} into {block_l_star} with Δ log-likelihood = {min_delta:.4f}"
                )

    draw_colored_graph(G, block_assignment_vector, layout, snapshot_dir, step="final")
    # Update log-likelihood after fallback merges
    B_post, _ = compute_block_probability_matrix(A, block_assignment_vector)
    current_log_likelihood = compute_log_likelihood(A, block_assignment_vector, B_post)
    log_likelihood_trace.append(current_log_likelihood)

    # save final log-likelihood trace
    np.save(f"{merge_output_dir}/log_likelihood_trace_{num_nodes}_K{K}.npy", np.array(log_likelihood_trace))

    # save final block sizes
    final_block_sizes = [
        block_assignment_vector.count(b) for b in set(block_assignment_vector)
    ]
    np.save(f"{merge_output_dir}/final_block_sizes_n{num_nodes}_K{K}_seed{seed}.npy", np.array(final_block_sizes))

    # save number of resulting blocks
    with open(f"{merge_output_dir}/final_num_blocks_{num_nodes}_K{K}.txt", "w") as f:
        f.write(str(len(set(block_assignment_vector))))

    # Save total number of merge steps
    with open(f"{merge_output_dir}/total_merge_steps_{num_nodes}_K{K}.txt", "w") as f:
        f.write(str(merge_step))

    block_assignment_vector_anonymized = block_assignment_vector
    return block_assignment_vector_anonymized
