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


def experiment_wrapper(param_sampler, data, db, lock):
    '''
    Run alignment over the dataset and save the result.

    Parameters
    ----------
    param_sampler : dict of functions
        Dictionary which maps parameter names to functions to sample values for
        those parameters.

    data : list of dict of np.ndarray
        Collection aligned/unaligned MIDI pairs

    db : tinydb.database.Table
        TinyDB table instance to save results to

    lock : multiprocessing.Lock
        Lock instance which prevents simultaneous writing to tinydb
    '''
    # Call the sample function for each param name in the param sampler dict
    # to create a dict which maps param names to sampled values
    params = dict((name, sample()) for (name, sample) in param_sampler.items())
    # Get the results dictionary for this parameter setting
    results = align_dataset.align_dataset(params, data)
    # Acquire lock for database
    lock.acquire()
    # Store this result
    db.insert({'params': params, 'results': results})
    # Acquire lock for database
    lock.release()


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
    # Database writing lock
    lock = multiprocessing.Lock()

    joblib.Parallel(n_jobs=10, verbose=51, backend='threading')(
        joblib.delayed(experiment_wrapper)(
            param_sampler, data, table, lock) for _ in range(N_TRIALS))
