from anonymization.merge import agglomerative_merge
from anonymization.swap import swap_refinement


def hybrid_merge_swap(A, K, seed=None, verbose=False, dataset=None):
    """
    Hybrid algorithm: merge first, then refine using swaps.

    Returns:
        z_merge (list): Block assignment after merge
        z_hybrid (list): Final block assignment after refinement
        log_likelihood_trace (list): LL trace during swap
    """

    if verbose:
        print("\n[Hybrid] Phase 1: Running Agglomerative Merge...")

    z_merge = agglomerative_merge(A, K, verbose=verbose, seed=seed, dataset=dataset)

    if verbose:
        print("\n[Hybrid] Phase 2: Refining using Swap-Based Refinement...")

    z_hybrid, log_likelihood_trace = swap_refinement(
        A, K, seed=seed, verbose=verbose, z_init=z_merge
    )

    return z_merge, z_hybrid, log_likelihood_trace
