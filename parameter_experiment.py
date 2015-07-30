'''
Run the alignment parameter search experiment.
'''

import os
import numpy as np
import align_dataset
import functools
import tinydb
import joblib
import multiprocessing

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'
# Path to the tinydb file
DB_PATH = 'results/results.json'
# Number of parameter settings to try
N_TRIALS = 10000


def result_writer(table, queue):
    '''
    Waits for results to appear on a queue, and when they do, writes them out

    Parameters
    ----------
    table : tinydb.database.Table
        TinyDB table instance to save results to

    queue : multiprocessing.Queue
        Queue where results will appear
    '''
    while True:
        # Check for an item on the queue
        item = queue.get()
        # Store in the tinydb table
        table.insert(item)


def experiment_wrapper(seed, param_sampler, data, queue):
    '''
    Run alignment over the dataset and save the result.

    Parameters
    ----------
    seed : int
        Seed for numpy RNG.

    param_sampler : dict of functions
        Dictionary which maps parameter names to functions to sample values for
        those parameters.

    data : list of dict of np.ndarray
        Collection aligned/unaligned MIDI pairs

    queue : multiprocessing.Queue
        Queue instance where results will be put
    '''
    # Use a different seed for each call to ensure that processes are
    # sampling different values from param_sampler
    np.random.seed(seed)
    # Call the sample function for each param name in the param sampler dict
    # to create a dict which maps param names to sampled values
    params = dict((name, sample()) for (name, sample) in param_sampler.items())
    # Get the results dictionary for this parameter setting
    results = align_dataset.align_dataset(params, data)
    # Store this result
    queue.put({'params': params, 'results': results})


if __name__ == '__main__':

    param_sampler = {
        # Use chroma or CQT for feature representation
        'feature': functools.partial(np.random.choice, ['chroma', 'gram']),
        # Beat sync, or don't
        'beat_sync': functools.partial(np.random.choice, [0, 1]),
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': functools.partial(np.random.choice, [None, np.inf, 1, 2]),
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': functools.partial(np.random.choice, [0, 1]),
        # Which distance metric to use for distance matrix
        'metric': functools.partial(np.random.choice,
                   ['euclidean', 'sqeuclidean', 'cosine']),
        # DTW additive penalty
        'add_pen': functools.partial(np.random.uniform, 0, 3),
        # DTW end point tolerance
        'gully': functools.partial(np.random.uniform, 0, 1),
        # Whether to constrain the path to within the tolerance
        'band_mask': functools.partial(np.random.choice, [0, 1])}

    # Load in the easy corrupted dataset
    data = align_dataset.load_dataset(CORRUPTED_PATH)
    # Check that the results database directory exists
    if not os.path.exists(os.path.split(DB_PATH)[0]):
        os.makedirs(os.path.split(DB_PATH)[0])
    # Load the database
    db = tinydb.TinyDB(DB_PATH)
    # Choose the table
    table = db.table('parameter_experiment')

    # Create multiprocessing Queue instance
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    # Start a process which waits for results and writes them
    writer_process = multiprocessing.Process(
        target=result_writer, args=(table, queue))
    writer_process.start()

    joblib.Parallel(n_jobs=10, verbose=51)(joblib.delayed(experiment_wrapper)(
        seed, param_sampler, data, queue) for seed in range(N_TRIALS))
