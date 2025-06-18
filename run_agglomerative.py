#!/usr/bin/env python

"""
Omnibenchmark-izes Marek Gagolewski's https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_agglomerative.py

Takes the true number of clusters into account and outputs a 2D matrix with as many columns as ks tested,
being true number of clusters `k` and tested range `k plusminus 2`

linkage_matrix computations are based on the codenby Mathew Kallada, Andreas Mueller (License: BSD 3 clause)
https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
"""

import argparse
import os, sys
import warnings

import numpy as np
import sklearn.cluster
import scipy.cluster.hierarchy
from prng import set_seed

VALID_LINKAGES = ['average', 'complete', 'ward']

def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)

    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")

    return(data)

def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)

    ##data.reset_index(drop=True,inplace=True)

    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    return(data)

## this maps the ks to their true offset to the truth, e.g.:
# >>> generate_k_range(5)
# {'k-2': 3, 'k-1': 4, 'k': 5, 'k+1': 6, 'k+2': 7}
# >>> generate_k_range(1)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 2, 'k+2': 3}
# >>> generate_k_range(2)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 3, 'k+2': 4}
## k is the true k
def generate_k_range(k):
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))

    # ids = ['k-2', 'k-1', 'k', 'k+1', 'k+2']
    ids = list(range(0,5))
    assert(len(ids) == len(Ks))

    k_ids_dict = dict.fromkeys(ids, 0)
    for i in range(len(ids)):
        key = ids[i]

        k_ids_dict[key] = Ks[i]
    return(k_ids_dict)

def do_agglomerative(X, Ks, linkage, seed=None):
    c = sklearn.cluster.AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None,
        compute_full_tree=True,
        linkage=linkage
    )

    # there is not randomness in the algorithm, but just to be consistent with other methods
    # (and perhaps uncover race conditions etc)
    with set_seed(seed):
        c.fit(X)

    # See https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    # create the counts of samples under each node::
    counts = np.zeros(c.children_.shape[0])
    n_samples = len(c.labels_)
    for i, merge in enumerate(c.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([c.children_, c.distances_,
                                      counts]).astype(float)

    fn = scipy.cluster.hierarchy.cut_tree

    # Function to compute clusters for a given k value with proper 1-based indexing
    def compute_clusters_for_k(k):
        labels = fn(linkage_matrix, n_clusters=k)
        return labels[:, -1] + 1  # Convert from 0-based to 1-based

    # Use dict comprehension to compute clusters for each k value
    res = {k_id: compute_clusters_for_k(Ks[k_id]) for k_id in sorted(Ks.keys())}

    return np.array([res[key] for key in sorted(res.keys())]).T

def main():
    parser = argparse.ArgumentParser(description='clustbench sklearn agglomerative runner')

    parser.add_argument('--data.matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data.true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store data files.')
    parser.add_argument('--name', type=str, help='name of the dataset', default='clustbench')
    parser.add_argument('--linkage', type=str,
                        help='linkage',
                        required = True)
    parser.add_argument('--seed', type=int,
                        help='random seed for reproducibility',
                        required = False, default=123)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.linkage not in VALID_LINKAGES:
        raise ValueError(f"Invalid method `{args.linkage}`")

    truth = load_labels(getattr(args, 'data.true_labels'))
    k = int(max(truth)) # true number of clusters
    Ks = generate_k_range(k)

    data = getattr(args, 'data.matrix')
    curr = do_agglomerative(X= load_dataset(data), Ks = Ks, linkage = args.linkage, seed = args.seed)

    name = args.name

    header=['k=%s'%s for s in Ks.values()]

    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(os.path.join(args.output_dir, f"{name}_ks_range.labels.gz"),
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header))

if __name__ == "__main__":
    main()
