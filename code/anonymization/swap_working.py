from anonymization.utils import compute_block_probability_matrix
from anonymization.utils import compute_log_likelihood_full as compute_log_likelihood
# from anonymization.utils import compute_log_likelihood_optimized as compute_log_likelihood
# from anonymization.utils import compute_log_likelihood_delta
import numpy as np
import random
from collections import defaultdict, Counter


def initialize_k_anonymous_partition(n, K, seed=None):
    """
    Initializes a valid K-anonymous assignment vector.
    """
    if seed is not None:
        random.seed(seed)

    z = [-1] * n
    nodes = list(range(n))
    random.shuffle(nodes)

    num_blocks = n // K
    r = n % K

    block_id = 0
    idx = 0

    for _ in range(num_blocks):
        for _ in range(K):
            z[nodes[idx]] = block_id
            idx += 1
        block_id += 1

    for i in range(r):
        z[nodes[idx]] = i % num_blocks
        idx += 1

    return z


def swap_refinement(A, K, seed=None, verbose=False, z_init=None):
    """
    Performs swap-based refinement of a K-anonymous block assignment.

    If z_init is None, initializes a K-anonymous partition from scratch (standalone mode).
    Otherwise, refines the provided z_init (hybrid mode).
    """
    n = A.shape[0]

    if z_init is not None:
        z = z_init.copy()
        if verbose:
            print("[Swap] Using provided initial block assignment (Hybrid mode)")
    else:
        z = initialize_k_anonymous_partition(n, K, seed=seed)
        if verbose:
            print("[Swap] Initialized K-anonymous partition (Standalone mode)")
            print("[Swap] Initial block size distribution:", Counter(z))

    B, _ = compute_block_probability_matrix(A, z)
    current_ll = compute_log_likelihood(A, z, B)
    log_likelihood_trace = [current_ll]

    if verbose:
        print(f"[Swap] Initial log-likelihood: {current_ll:.4f}")

    iteration = 0
    while True:
        improved = False
        eligible_swaps = 0
        nodes = list(range(n))
        random.shuffle(nodes)

        for i in nodes:
            for j in nodes:
                if i == j or z[i] == z[j]:
                    continue

                block_i, block_j = z[i], z[j]
                block_counts = defaultdict(int)
                for u in z:
                    block_counts[u] += 1

                size_i = block_counts[block_i]
                size_j = block_counts[block_j]
                # Check if the swap would violate K-anonymity
                # After the swap, both blocks still have same size (swap keeps sizes unchanged)
                # So only disallow if either block is already smaller than K (should never happen)
                if size_i < K or size_j < K:
                    continue  # Invalid state (shouldn't happen), or skip to be safe

                # Try the swap
                z[i], z[j] = block_j, block_i
                eligible_swaps += 1

                B_new, _ = compute_block_probability_matrix(A, z)
                new_ll = compute_log_likelihood(A, z, B_new)

                if new_ll > current_ll:
                    current_ll = new_ll
                    improved = True
                    if verbose:
                        print(
                            f"[Swap] Iter {iteration}: Swapped {i}<->{j} | LL improved to {new_ll:.4f}"
                        )
                        log_likelihood_trace.append(new_ll)
                    break  # greedy accept
                else:
                    z[i], z[j] = block_i, block_j  # revert
                    log_likelihood_trace.append(new_ll)

            if improved:
                break  # exit outer loop too

        if verbose:
            print(
                f"[Swap] Iteration {iteration} done | Eligible swaps tried: {eligible_swaps}"
            )
            if not improved:
                print(f"[Swap] No improving swaps found. Converged.")

        if not improved:
            break

        iteration += 1

    return z, log_likelihood_trace
