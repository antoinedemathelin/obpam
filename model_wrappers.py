import numpy as np
import kmedoids as km
import onebatch as obk
from sklearn.metrics import pairwise_distances
import time 
import gc
import models as mod

try:
    from banditpam import KMedoids
except:
    pass


def fast_clara(X, K, distance, seed, n_sampling_iter=5):
    np.random.seed(seed)
    t0 = time.time()
    medoids = mod.clara(X, n_clusters=K, metric=distance, n_sampling=int(80 + 4*K),
                         n_sampling_iter=n_sampling_iter, method="fasterpam")
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))
    

def clara(X, K, distance, seed, n_sampling_iter=5):
    np.random.seed(seed)
    t0 = time.time()
    medoids = mod.clara(X, n_clusters=K, metric=distance, n_sampling=int(80 + 4*K),
                         n_sampling_iter=n_sampling_iter, method="pam")
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def kmc2(X, K, distance, seed, chain_length=100):
    np.random.seed(seed)
    t0 = time.time()
    medoids = mod.kmc2(X, K=K, distance=distance, chain_length=chain_length)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def kmeans_pp(X, K, distance, seed):
    np.random.seed(seed)
    t0 = time.time()
    medoids = mod.kmeans_pp(X, K=K, distance=distance)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def kmeans_pp_ls(X, K, distance, seed, Z=10):
    np.random.seed(seed)
    t0 = time.time()
    medoids = mod.kmeans_pp_ls(X, K=K, distance=distance, Z=Z)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def random_model(X, K, distance, seed):
    np.random.seed(seed)
    t0 = time.time()
    medoids = np.random.choice(X.shape[0], K, replace=False)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))
    

def one_batch(X, K, distance, seed, batch_size="auto", verbose=0):
    """
    Wrapper for our new implementation.
    """
    t0 = time.time()
    if batch_size == "auto":
        batch_size = int(100 * np.log(X.shape[0] * K))
    batch_size = min(X.shape[0], batch_size)
    print("Batch Size", batch_size)
    medoids = obk.one_batch_pam(X=X, K=K, distance=distance, batch_size=batch_size, verbose=verbose)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def one_batch_nniw(X, K, distance, seed, batch_size="auto", verbose=0):
    """
    Wrapper for our new implementation.
    """
    t0 = time.time()
    if batch_size == "auto":
        batch_size = int(100 * np.log(X.shape[0] * K))
    batch_size = min(X.shape[0], batch_size)
    print("Batch Size", batch_size)
    medoids = obk.one_batch_pam(X=X, K=K, distance=distance, batch_size=batch_size, verbose=verbose, weight="nniw")
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def one_batch_bias(X, K, distance, seed, batch_size="auto", verbose=0):
    """
    Wrapper for our new implementation.
    """
    t0 = time.time()
    if batch_size == "auto":
        batch_size = int(100 * np.log(X.shape[0] * K))
    batch_size = min(X.shape[0], batch_size)
    print("Batch Size", batch_size)
    medoids = obk.one_batch_pam(X=X, K=K, distance=distance, batch_size=batch_size, verbose=verbose, weight=None)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def one_batch_lwcs(X, K, distance, seed, batch_size="auto", verbose=0):
    """
    Wrapper for our new implementation.
    """
    t0 = time.time()
    if batch_size == "auto":
        batch_size = int(100 * np.log(X.shape[0] * K))
    batch_size = min(X.shape[0], batch_size)
    print("Batch Size", batch_size)
    medoids = obk.one_batch_pam(X=X, K=K, distance=distance, batch_size=batch_size, verbose=verbose, weight="lwcs")
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def banditpam(X, K, distance, seed, max_iter=0):
    """
    Wrapper for fitting a KMedoids object with BanditPAM algorithm.
    See: https://banditpam.readthedocs.io/en/latest/
    """
    np.random.seed(seed)
    t0 = time.time()
    bpam = KMedoids(n_medoids = K, algorithm="BanditPAM", max_iter=max_iter)
    bpam.fit(X, distance)
    medoids = bpam.medoids.reshape(K)
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))


def pam(X, K, distance, seed, method="pam"):
    np.random.seed(seed)
    t0 = time.time()
    m = km.KMedoids(n_clusters = K, metric="precomputed", method=method)
    K = pairwise_distances(X, metric=distance)
    m.fit(K)
    medoids = m.medoid_indices_
    t1 = time.time()
    elapsed_time = t1 - t0
    np.save("./medoids.npy", np.array(medoids))
    np.save("./time.npy", np.array([elapsed_time]))