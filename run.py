import numpy as np
from sklearn.metrics import pairwise_distances
import gc
import multiprocessing
from tqdm import tqdm
import csv
import os
from data_loaders import load_data


def update_results(container, keys, values):
    """
    Updates log dictionary <container> with the new <key>:<values> pairs.
    """
    for key, value in zip(keys, values):
        container[key].append(value)
        
def save_results(results,out_name):
    field_names = list(results.keys())
    with open(out_name,'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)  
        if os.stat(out_name).st_size == 0:
            writer.writeheader()
        num_rows = len(results[list(results.keys())[0]])
        rows = [dict((k, results[k][i]) for k in field_names) for i in range(num_rows)]
        writer.writerows(rows)
        f.close()


def run(dataset, seeds, K_list, sample_sizes, metric='euclidean', method=None, out_name='other_runs', timeout=600, save=False, params={}):
    """
    Performs multiple runs of KMedoids algorithms with given configurations.
    Keyword arguments:
    
    :param dataset str: Dataset to compute medoids on.
    :param seeds list: list of seeds.
    :param K_list list: List with values for the number of medoids to compute.
    :param sample_sizes list: Sizes for different runs.
    :param metric str: Metric used for distance computations. Follows sklearn.metrics.pairwise_distances naming, defaults to 'euclidean'
    :param method callable: {name: function} function to compute kmedoids, defaults to None
    :param out_name str: Name for output csv file, defaults to 'other_runs'
    :param save bool: Whether to save or not a file, defaults to False
    :param params dict: Parameters for the method, defaults to {}
    """
    results = dict(method=[], n=[], K=[], objective=[], time=[], index=[], seed=[], data=[], params=[])
    subsample = np.empty(())
    
    func = method
    name = method.__name__
    print("METHOD - {}".format(name))
    kmax = np.inf

    if sample_sizes is None:
        sample_sizes = [None]
    
    for n in tqdm(sample_sizes):
        for seed in seeds:
            del(subsample)
            gc.collect()
            subsample, shuffled_idxs = load_data(dataset=dataset, seed=seed, n=n)

            print(params)
            
            for K in tqdm(K_list):
                if K >= kmax:
                    update_results(results, results.keys(), [name, n, K, np.inf, timeout, [], seed, dataset, str(params)])
                else:
                    process = multiprocessing.Process(target=func, name="running-"+name,
                                                      args=(subsample, K, metric, seed), kwargs=params)
                    process.start()
                    process.join(timeout)

                    if process.is_alive():
                        kmax = K
                        process.terminate()
                        process.join()
                        update_results(results, results.keys(), [name, n, K, np.inf, timeout, [], seed, dataset, str(params)])
                    else:
                        medoid_index = list(np.load("./medoids.npy").astype(np.int32))
                        time_medoid = np.load("./time.npy")[0]
                        os.remove("./medoids.npy")
                        os.remove("./time.npy")
                        obj = pairwise_distances(subsample, subsample[medoid_index],metric=metric).min(axis=1).mean()
                        update_results(results, results.keys(),
                                       [name, n, K, obj, time_medoid, shuffled_idxs[medoid_index], seed, dataset, str(params)])
                gc.collect()
            if save:
                save_results(results, out_name)
                results = {key:[] for key in results.keys()}
    return True