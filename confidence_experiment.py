'''
Run the confidence score search experiment.
'''

import numpy as np
import db_utils
import scipy.stats
import joblib
import align_dataset
try:
    import ujson as json
except ImportError:
    import json
import glob
import os

# Path to corrupted datasets, created by create_data.py
CORRUPTED_HARD = 'data/corrupted_hard/*.npz'
# Path to json results for parameter experiment
PARAMETER_RESULTS_GLOB = 'results/parameter_experiment_gp/*.json'
# Path where confidence experiment results should be written
OUTPUT_RESULTS_PATH = 'results/confidence_experiment/'


def check_trials(best_errors, parameter_trials):
    """Compute confidence score effectiveness a set of parameter settings.

    Parameters
    ----------
    best_errors : np.ndarray
        Array of the per-song errors resulting from the best aligner
    parameter_trials: list
        List of hyperparameter search trials on the "easy" dataset
    """
    # Load in corrupted MIDI datasets
    hard_dataset = align_dataset.load_dataset(CORRUPTED_HARD)
    # Grab objective values for each trial
    objectives = [np.mean(r['results']['mean_errors'])
                  for r in parameter_trials]
    # Sort the results settings by their objective value
    parameter_trials = [parameter_trials[n] for n in np.argsort(objectives)]
    for trial in parameter_trials:
        easy_results = trial['results']
        # Retrieve the errors for each song
        easy_errors = np.array(easy_results['mean_errors'])
        # Run a paired difference test of the errors, i.e. test whether the
        # distribution of differences between best_errors[n] and
        # easy_errors[n] is significantly different from 0 under a t-test
        _, r_score = scipy.stats.ttest_1samp(best_errors - easy_errors, 0)
        # When best_errors = easy_errors, the r_score will be NaN
        if np.isnan(r_score):
            r_score = 1.
        # Replace 'norm' param with numeric infinity if it's 'inf'
        params = trial['params']
        if params['norm'] == str(np.inf):
            params['norm'] = np.inf
        # Align the hard dataset using these params and retrieve errors
        hard_results = align_dataset.align_dataset(params, hard_dataset)
        hard_errors = np.array(hard_results['mean_errors'])
        # Create results dict, storing the r_csore and errors, plus stuff below
        result = dict(r_score=r_score,
                      easy_errors=easy_errors.tolist(),
                      hard_errors=hard_errors.tolist())
        # Try all combinations of score normalization
        for include_pen in [0, 1]:
            for length_normalize in [0, 1]:
                for mean_normalize in [0, 1]:
                    for results, name in zip([hard_results, easy_results],
                                             ['hard', 'easy']):
                        # Retrieve the score with or without penalties included
                        if include_pen:
                            scores = np.array(results['raw_scores'])
                            name += '_penalty'
                        else:
                            scores = np.array(results['raw_scores_no_penalty'])
                            name += '_no_penalty'
                        # Optionally normalize by path length
                        if length_normalize:
                            scores /= np.array(results['path_lengths'])
                            name += '_len_norm'
                        # Optionally normalize by distance matrix mean
                        if mean_normalize:
                            scores /= np.array(
                                results['distance_matrix_means'])
                            name += '_mean_norm'
                        # Store the scores
                        result[name + '_scores'] = scores.tolist()
        # Write out this result
        db_utils.dump_result(params, result, OUTPUT_RESULTS_PATH)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_RESULTS_PATH):
        os.makedirs(OUTPUT_RESULTS_PATH)
    # Load in all parameter search experiment results
    parameter_trials = []
    for result_file in glob.glob(PARAMETER_RESULTS_GLOB):
        with open(result_file) as f:
            parameter_trials.append(json.load(f))
    # Grab objective values for each trial
    objectives = [np.mean(r['results']['mean_errors'])
                  for r in parameter_trials]
    best_errors = np.array(
        parameter_trials[np.argmin(objectives)]['results']['mean_errors'])
    # Split up the parameter trials into 10 roughly equally sized divisions
    split_parameter_trials = [parameter_trials[n::10] for n in range(10)]

    # Run check_trials for all splits in parallel
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(check_trials)(best_errors, trials)
        for trials in split_parameter_trials)
