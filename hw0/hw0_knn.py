"""
K Nearest Neighbors
Liam Strand

>>> data_NF = np.asarray([
...     [1, 0],
...     [0, 1],
...     [-1, 0],
...     [0, -1]])
>>> query_QF = np.asarray([
...     [0.9, 0],
...     [0, -0.9]])

Example Test K=1
----------------
# Find the single nearest neighbor for each query vector
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
>>> neighb_QKF.shape
(2, 1, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[1., 0.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.]])

Example Test K=3
----------------
# Now find 3 nearest neighbors for the same queries
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
>>> neighb_QKF.shape
(2, 3, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.],
       [ 1.,  0.],
       [-1.,  0.]])
"""
import numpy as np

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    """ Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    """

    # Extract dimensions
    N, F = data_NF.shape
    Q, _ = query_QF.shape

    def process_query(q):
        # Euclidian distance is the norm of the vector difference
        eucl_dist = lambda a: np.linalg.norm(q - a)

        # Calculate the distances from all examples to query
        dists_N = np.array(list(map(eucl_dist, data_NF)))
        # faster: np.sum(np.square(data_NF - q), axis=1) 
        ranking_N = np.argsort(dists_N, kind="stable")

        # Return the K examples nearest to the query
        return data_NF[ranking_N][:K]

    # Process all the queries and collect them into an array of the correct type
    neighb_QKF = np.array(list(map(process_query, query_QF)), dtype="float64")

    return neighb_QKF

