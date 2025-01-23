import numpy as np
from sklearn.metrics import pairwise_distances
import kmedoids as km

EPS = np.finfo(float).eps


def clara(X, n_clusters=10, metric="euclidean", n_sampling=100,
               n_sampling_iter=5, method="pam"):
    best_score = np.inf
    for iter in range(n_sampling_iter):
        if n_sampling >= X.shape[0]:
            sample_idxs = np.arange(X.shape[0])
        else:
            sample_idxs = np.random.choice(
                np.arange(X.shape[0]), size=n_sampling, replace=False
            )
        pam = km.KMedoids(n_clusters=n_clusters,
        metric="precomputed", method=method)
        
        K = pairwise_distances(X[sample_idxs], metric=metric)
        
        pam.fit(K)
        cluster_centers = X[sample_idxs][pam.medoid_indices_]
        dist = pairwise_distances(X, cluster_centers, metric=metric)
        inertia = np.sum(dist.min(1))

        if inertia < best_score:
            best_score = inertia
            best_medoids_idxs = pam.medoid_indices_
            best_sample_idxs = sample_idxs

    return best_sample_idxs[best_medoids_idxs]


def kmc2(X, K=10, distance="euclidean", chain_length=100):
    index = np.random.choice(X.shape[0], 1)[0]

    medoids = [index]

    for k in range(K-1):
        index_x = np.random.choice(X.shape[0], 1)[0]
        Dx = pairwise_distances(X[[index_x]], X[medoids], metric=distance).min()

        for j in range(chain_length):
            index_y = np.random.choice(X.shape[0], 1)[0]
            Dy = pairwise_distances(X[[index_y]], X[medoids], metric=distance).min()

            U = np.random.random(1)[0]
            if Dy / (Dx + EPS) > U:
                index_x = index_y
                Dx = Dy
        medoids.append(index_x)
    return medoids


def kmeans_pp(X, K=10, distance="euclidean"):
    index = np.random.choice(X.shape[0], 1)[0]
    min_dist = np.full((X.shape[0],), np.inf)

    medoids = [index]
    
    for k in range(K-1):
        dist = pairwise_distances(X[[medoids[-1]]], X, metric=distance).ravel()
        min_dist = np.minimum(min_dist, dist)
        probas = np.zeros(min_dist.shape[0])
        probas += min_dist
        sum_ = probas.sum()
        if sum_ > 0.:
            probas /= sum_
        else:
            probas = None
        medoids.append(np.random.choice(X.shape[0], 1, p=probas)[0])
    return medoids


def kmeans_pp_ls(X, K=10, distance="euclidean", Z=10):
    # Kmeans++
    index = np.random.choice(X.shape[0], 1)[0]
    min_dist = np.full((X.shape[0],), np.inf)

    medoids = [index]

    dist_to_medoids = []
    
    for k in range(K-1):
        dist = pairwise_distances(X[[medoids[-1]]], X, metric=distance).ravel()
        min_dist = np.minimum(min_dist, dist)
        probas = np.zeros(min_dist.shape[0])
        probas += min_dist
        sum_ = probas.sum()
        if sum_ > 0.:
            probas /= sum_
        else:
            probas = None
        medoids.append(np.random.choice(X.shape[0], 1, p=probas)[0])
        dist_to_medoids.append(dist)

    dist = pairwise_distances(X[[medoids[-1]]], X, metric=distance).ravel()
    min_dist = np.minimum(min_dist, dist)
    dist_to_medoids.append(dist)

    # Local search
    for _ in range(Z):
        probas = np.zeros(min_dist.shape[0])
        probas += min_dist
        sum_ = probas.sum()
        if sum_ > 0.:
            probas /= sum_
        else:
            probas = None
        medoids_pot = np.random.choice(X.shape[0], 1, p=probas)[0]
        dist = pairwise_distances(X[[medoids_pot]], X, metric=distance).ravel()

        best_potential = min_dist.sum()
        index_to_remove = None
        best_min_dist = min_dist
        for k in range(K):
            min_dist_k = np.stack(dist_to_medoids[:k]+dist_to_medoids[k+1:], 0).min(0)
            min_dist_k = np.minimum(min_dist_k, dist)
            potential_p = min_dist_k.sum()
            if potential_p < best_potential:
                index_to_remove = k
                best_potential = potential_p
                best_min_dist = min_dist_k
        if index_to_remove is not None:
            dist_to_medoids.pop(index_to_remove)
            dist_to_medoids.append(dist)
            medoids.pop(index_to_remove)
            medoids.append(medoids_pot)
            min_dist = best_min_dist
    return medoids
    