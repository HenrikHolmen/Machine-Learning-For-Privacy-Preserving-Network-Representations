from anonymization.merge import agglomerative_merge
from anonymization.swap import swap_refinement


def hybrid_merge_swap(A, K, seed=None, verbose=False):
    """
    Hybrid algorithm: merge first, then refine using swaps.
    """

    if verbose:
        print("\n[Hybrid] Phase 1: Running Agglomerative Merge...")

    z_merge = agglomerative_merge(A, K, verbose=verbose, seed=seed)

    if verbose:
        print("\n[Hybrid] Phase 2: Refining using Swap-Based Refinement...")

    z_hybrid, log_likelihood_trace = swap_refinement(
        A, K, seed=seed, verbose=verbose, z_init=z_merge
    )

    return z_merge, z_hybrid, log_likelihood_trace
