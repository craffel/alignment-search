'''
Run the alignment parameter search experiment.
'''

import simple_spearmint
import numpy as np
import argparse
import align_dataset
import os
import db_utils

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'
# How many total trials of hyperparameter optimization should we run?
N_TRIALS = 100
# How many randomly selected hyperparameter settings shuld we start with?
INITIAL_RANDOM = 100
# Where should experiment results be output?
OUTPUT_PATH = 'results/parameter_experiment_gp'
# Where do the results from the random parameter search live?
RANDOM_RESULTS_PATH = 'results/parameter_experiment_random'

if __name__ == '__main__':
    # Retrieve the seed from the command line
    parser = argparse.ArgumentParser(
        description='Run a MIDI alignment parameter search experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', action='store', type=int, default=0,
                        help='Random seed')
    seed = parser.parse_args().seed
    np.random.seed(seed)

    space = {
        # Use chroma or CQT for feature representation
        'feature': {'type': 'enum', 'options': ['chroma', 'gram']},
        # Beat sync, or don't
        'beat_sync': {'type': 'enum', 'options': [True, False]},
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': {'type': 'enum', 'options': [None, np.inf, 1, 2]},
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': {'type': 'enum', 'options': [True, False]},
        # Which distance metric to use for distance matrix
        'metric': {'type': 'enum',
                   'options': ['euclidean', 'sqeuclidean', 'cosine']},
        # DTW additive penalty
        'add_pen': {'type': 'float', 'min': 0, 'max': 3},
        # DTW end point tolerance
        'gully': {'type': 'float', 'min': 0, 'max': 1},
        # Whether to constrain the path to within the tolerance
        'band_mask': {'type': 'enum', 'options': [True, False]}}

    # Check that the results database directory exists
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Initialize apsis experiment
    experiment = simple_spearmint.SimpleSpearmint(space, noiseless=True)

    # Load in all random parameter search results
    random_params, random_objectives = db_utils.get_experiment_results(
        os.path.join(RANDOM_RESULTS_PATH, '*.json'))
    # Seed the GP optimizer with the INITIAL_RANDOM best results and
    # INITIAL_RANDOM random results
    best_indices = np.argsort(random_objectives)[:INITIAL_RANDOM]
    # Set subtraction avoids randomly choosing best objective trials
    random_indices = np.random.choice(
        [n for n in range(len(random_params)) if n not in best_indices],
        INITIAL_RANDOM, False)
    # Seed the GP optimizer with random parameter search results
    for n in np.append(best_indices, random_indices):
        # Replace 'inf' with actual values
        params = dict((k, v) if v != 'inf' else (k, np.inf)
                      for k, v in random_params[n].items())
        experiment.update(params, random_objectives[n])

    # Load in the alignment dataset
    data = align_dataset.load_dataset(CORRUPTED_PATH)

    for _ in range(N_TRIALS):
        # Retrieve GP-based parameter suggestion
        candidate_params = experiment.suggest()
        # Get results for these parameters
        result = align_dataset.align_dataset(candidate_params, data)
        # Write results out
        db_utils.dump_result(candidate_params, result, OUTPUT_PATH)
        # Update optimizer
        objective = np.mean(result['mean_errors'])
        experiment.update(candidate_params, objective)
