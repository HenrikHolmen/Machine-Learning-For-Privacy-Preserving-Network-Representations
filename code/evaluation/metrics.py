import numpy as np
import networkx as nx
from scipy.linalg import eigvals


# def compute_degree_distribution(A):
#     G = nx.from_numpy_array(A)
#     degrees = [d for _, d in G.degree()]
#     hist = np.bincount(degrees)
#     return hist / hist.sum()


def compute_clustering_coefficient(A):
    G = nx.from_numpy_array(A)
    return nx.average_clustering(G)


def compute_shortest_path_stats(A):
    G = nx.from_numpy_array(A)
    if nx.is_connected(G):
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        all_lengths = [
            l for lengths_dict in lengths.values() for l in lengths_dict.values()
        ]
        return np.mean(all_lengths), np.std(all_lengths)
    else:
        return None, None  # Graph is disconnected


def compute_assortativity(A):
    G = nx.from_numpy_array(A)
    return nx.degree_assortativity_coefficient(G)


def compute_modularity(A, z):
    G = nx.from_numpy_array(A)
    communities = {}
    for node, block in enumerate(z):
        communities.setdefault(block, []).append(node)
    partition = list(communities.values())
    return nx.community.modularity(G, partition)


def compute_spectral_properties(A):
    eigenvalues = eigvals(A)
    return np.sort(np.real(eigenvalues))[::-1]  # descending order


def compute_all_utilities(A, z_true, z_anonymized):
    """
    Returns a dictionary of all relevant utility metrics comparing true versus anonymized.
    """

    results = {}

    # # Degree distribution - NOTE Found it was unchanged since A is fixed.
    # degree_distribution_true = compute_degree_distribution(A)
    # degree_distribution_anonymized = compute_degree_distribution(A)
    # results["degree_distribution_difference"] = np.linalg.norm(
    #     degree_distribution_true - degree_distribution_anonymized
    # )

    # Clustering coefficient
    results["clustering_coefficient"] = compute_clustering_coefficient(A)

    # Shortest paths
    mean_sp, std_sp = compute_shortest_path_stats(A)
    results["shortest_path_mean"] = mean_sp
    results["shortest_path_std"] = std_sp

    # Assortativity
    results["assortativity"] = compute_assortativity(A)

    # Modularity (requires block assignments)
    results["modularity_true"] = compute_modularity(A, z_true)
    results["modularity_anonymized"] = compute_modularity(A, z_anonymized)

    # Spectral properties (just logs top eigenvalues - which can be visualized separately)
    results["spectral_eigenvalues"] = compute_spectral_properties(A)[
        :10
    ]  # The top 10 eigenvalues

    return results
