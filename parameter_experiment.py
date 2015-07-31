'''
Run the alignment parameter search experiment.
'''

import apsis.models.parameter_definition
import apsis.assistants.experiment_assistant
import apsis.utilities.randomization
import numpy as np
import argparse
import align_dataset
import os

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'
# How many total trials of hyperparameter optimization should we run?
TRIALS = 100
# How many randomly selected hyperparameter settings shuld we start with?
INITIAL_RANDOM = 3
# Where should experiment results be output?
RESULTS_PATH = 'results'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a MIDI alignment parameter search experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', action='store', type=int, default=0,
                        help='Random seed for spearmint.')
    seed = parser.parse_args().seed

    ListParam = apsis.models.parameter_definition.NominalParamDef
    FloatParam = apsis.models.parameter_definition.MinMaxNumericParamDef

    space = {
        # Use chroma or CQT for feature representation
        'feature': ListParam(['chroma', 'gram']),
        # Beat sync, or don't
        # 'beat_sync': ListParam([True, False]),
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': ListParam([None, np.inf, 1, 2]),
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': ListParam([True, False]),
        # Which distance metric to use for distance matrix
        'metric': ListParam(['euclidean', 'sqeuclidean', 'cosine']),
        # DTW additive penalty
        'add_pen': FloatParam(0, 4),
        # DTW end point tolerance
        'gully': FloatParam(0, 1),
        # Whether to constrain the path to within the tolerance
        'band_mask': ListParam([True, False])}

    # Get random state from apsis
    random_state = apsis.utilities.randomization.check_random_state(seed)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    expmnt = apsis.assistants.experiment_assistant.BasicExperimentAssistant(
        'parameter_experiment_seed_{}'.format(seed),
        optimizer="BayOpt", param_defs=space,
        optimizer_arguments={'random_state': random_state,
                             'initial_random_runs': INITIAL_RANDOM},
        write_directory_base=RESULTS_PATH)

    data = align_dataset.load_dataset(CORRUPTED_PATH)

    for _ in range(TRIALS):
        candidate = expmnt.get_next_candidate()
        params = dict(candidate.params)
        params['beat_sync'] = True
        result = align_dataset.align_dataset(params, data)
        mean_errors = [r['mean_error'] for r in result]
        candidate.result = np.mean(mean_errors)
        print candidate.params
        print candidate.result
        print
        expmnt.update(candidate)
