'''
Run the alignment parameter search experiment.
'''

import apsis.models.parameter_definition
import apsis.assistants.experiment_assistant
import apsis.utilities.randomization
import apsis.models.candidate
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
INITIAL_RANDOM = 1000
# Where should experiment results be output?
OUTPUT_PATH = 'results/parameter_experiment_gp'
# Where do the results from the random parameter search live?
RANDOM_RESULTS_PATH = 'results/parameter_experiment_random'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a MIDI alignment parameter search experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', action='store', type=int, default=0,
                        help='Random seed')
    seed = parser.parse_args().seed

    ListParam = apsis.models.parameter_definition.NominalParamDef
    FloatParam = apsis.models.parameter_definition.MinMaxNumericParamDef

    space = {
        # Use chroma or CQT for feature representation
        'feature': ListParam(['chroma', 'gram']),
        # Beat sync, or don't
        'beat_sync': ListParam([True, False]),
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': ListParam([None, np.inf, 1, 2]),
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': ListParam([True, False]),
        # Which distance metric to use for distance matrix
        'metric': ListParam(['euclidean', 'sqeuclidean', 'cosine']),
        # DTW additive penalty
        'add_pen': FloatParam(0, 3),
        # DTW end point tolerance
        'gully': FloatParam(0, 1),
        # Whether to constrain the path to within the tolerance
        'band_mask': ListParam([True, False])}

    # Get random state from apsis
    random_state = apsis.utilities.randomization.check_random_state(seed)

    # Check that the results database directory exists
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Initialize apsis experiment
    expmnt = apsis.assistants.experiment_assistant.BasicExperimentAssistant(
        'parameter_experiment_gp_seed_{}'.format(seed),
        optimizer="BayOpt", param_defs=space,
        optimizer_arguments={'random_state': random_state,
                             'initial_random_runs': 0})

    # Load in all random parameter search results
    random_params, random_objectives = db_utils.get_experiment_results(
        os.path.join(RANDOM_RESULTS_PATH, '*.json'))
    # Seed the GP optimizer with random parameter search results
    for n in np.random.choice(len(random_params), INITIAL_RANDOM, False):
        # Replace 'inf' with actual values
        params = dict((k, v) if v != 'inf' else (k, np.inf)
                      for k, v in random_params[n].items())
        candidate = apsis.models.candidate.Candidate(params)
        candidate.result = random_objectives[n]
        expmnt.update(candidate)

    # Load in the alignment dataset
    data = align_dataset.load_dataset(CORRUPTED_PATH)

    for _ in range(N_TRIALS):
        # Retrieve GP-based parameter suggestion
        candidate = expmnt.get_next_candidate()
        # Get results for these parameters
        params = dict(candidate.params)
        result = align_dataset.align_dataset(params, data)
        # Write results out
        db_utils.dump_result(params, result, OUTPUT_PATH)
        # Update optimizer
        candidate.result = np.mean(result['mean_errors'])
        expmnt.update(candidate)
