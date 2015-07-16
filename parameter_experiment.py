'''
Run the alignment parameter search experiment.
'''

import spearmint.main
import os
import numpy as np
import argparse
import align_dataset

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'


def main(job_id, params):
    # Spearmint requires (I think) all params to be passed as at least
    # 1-dimensional arrays.  So, get the first entry to flatten.
    for key, value in params.items():
        params[key] = value[0]
    # Load in the dataset
    data = align_dataset.load_dataset(CORRUPTED_PATH)
    # Compute results for this parameter setting and retrieve mean errors
    mean_errors = [r['mean_error']
                   for r in align_dataset.align_dataset(params, data)]
    # TODO: Is there a way to write out results?
    return np.mean(mean_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a MIDI alignment parameter search experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', action='store', type=int, default=0,
                        help='Random seed for spearmint.')
    seed = parser.parse_args().seed

    space = {
        # Use chroma or CQT for feature representation
        'feature': {'type': 'ENUM', 'size': 1, 'options': ['chroma', 'gram']},
        # Beat sync, or don't
        'beat_sync': {'type': 'ENUM', 'size': 1, 'options': [True, False]},
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': {'type': 'ENUM', 'size': 1, 'options': [None, np.inf, 1, 2]},
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': {'type': 'ENUM', 'size': 1, 'options': [True, False]},
        # Which distance metric to use for distance matrix
        'metric': {'type': 'ENUM', 'size': 1,
                   'options': ['euclidean', 'sqeuclidean', 'cosine']},
        # DTW additive penalty
        'add_pen': {'type': 'FLOAT', 'size': 1, 'min': 0, 'max': 2.},
        # DTW end point tolerance
        'gully': {'type': 'FLOAT', 'size': 1, 'min': 0, 'max': 1.},
        # Whether to constrain the path to within the tolerance
        'band_mask': {'type': 'ENUM', 'size': 1, 'options': [True, False]}}

    # Set up spearmint options dict
    options = {'language': 'PYTHON',
               'main-file': os.path.basename(__file__),
               'experiment-name': 'alignment_search_seed_{}'.format(seed),
               'likelihood': 'NOISELESS',
               'variables': space,
               'grid-seed': seed}

    spearmint.main.main(options, os.getcwd())
