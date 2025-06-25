import numpy as np
import networkx as nx
from scipy.linalg import eigvals


def compute_degree_distribution(A):
    G = nx.from_numpy_array(A)
    degrees = [d for _, d in G.degree()]
    hist = np.bincount(degrees)
    return hist / hist.sum() if hist.sum() > 0 else hist

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

def compute_all_utilities(A_hat, z_anonymized, A_original=None, z_true=None):
    """
    Computes structural utility metrics on the anonymized graph (A_hat).
    If A_original and z_true are provided, also compares modularity and degree distribution.
    """
    results = {}

    if A_original is not None:
        deg_true = compute_degree_distribution(A_original)
        deg_anonymized = compute_degree_distribution(A_hat)

        # Pad to same length
        max_len = max(len(deg_true), len(deg_anonymized))
        deg_true = np.pad(deg_true, (0, max_len - len(deg_true)))
        deg_anonymized = np.pad(deg_anonymized, (0, max_len - len(deg_anonymized)))

        results["degree_distribution_difference"] = np.linalg.norm(deg_true - deg_anonymized)
   

    # Clustering coefficient
    results["clustering_coefficient"] = compute_clustering_coefficient(A_hat)

    # Shortest path stats
    mean_sp, std_sp = compute_shortest_path_stats(A_hat)
    results["shortest_path_mean"] = mean_sp
    results["shortest_path_std"] = std_sp

    # Assortativity
    results["assortativity"] = compute_assortativity(A_hat)

    # Modularity (anonymized always)
    results["modularity_anonymized"] = compute_modularity(A_hat, z_anonymized)

    # Modularity (true, if ground truth provided)
    if z_true is not None and A_original is not None:
        results["modularity_true"] = compute_modularity(A_original, z_true)

    # Spectral properties
    results["spectral_eigenvalues"] = compute_spectral_properties(A_hat)[:10]

    return results
