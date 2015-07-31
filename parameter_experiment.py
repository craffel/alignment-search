'''
Run the alignment parameter search experiment.
'''

import os
import numpy as np
import align_dataset
import functools
try:
    import ujson as json
except ImportError:
    import json

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'
# Path where results should be written
OUTPUT_PATH = 'results/parameter_experiment'
# Number of parameter settings to try
N_TRIALS = 1000


def experiment_wrapper(param_sampler, data, output_path):
    '''
    Run alignment over the dataset and save the result.

    Parameters
    ----------
    param_sampler : dict of functions
        Dictionary which maps parameter names to functions to sample values for
        those parameters.

    data : list of dict of np.ndarray
        Collection aligned/unaligned MIDI pairs

    output_path : str
        Where to write the results .json file
    '''
    # Call the sample function for each param name in the param sampler dict
    # to create a dict which maps param names to sampled values
    params = dict((name, sample()) for (name, sample) in param_sampler.items())
    # Get the results dictionary for this parameter setting
    results = align_dataset.align_dataset(params, data)
    # ujson can't handle infs, so we need to replace them manually:
    if params['norm'] == np.inf:
        params['norm'] = str(np.inf)
    # Convert params dict to a string of the form
    # param1_name_param1_value_param2_name_param2_value...
    param_string = "_".join(
        '{}_{}'.format(name, value) if type(value) != float else
        '{}_{:.3f}'.format(name, value) for name, value in params.items())
    # Construct a path where the .json results file will be written
    output_filename = os.path.join(output_path, "{}.json".format(param_string))
    # Store this result
    try:
        with open(output_filename, 'wb') as f:
            json.dump({'params': params, 'results': results}, f)
    # Ignore "OverflowError"s raised by ujson; they correspond to inf/NaN
    except OverflowError:
        pass


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
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for _ in range(N_TRIALS):
        experiment_wrapper(param_sampler, data, OUTPUT_PATH)
