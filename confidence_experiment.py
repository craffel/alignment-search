'''
Run the confidence score search experiment.
'''

import numpy as np
import spearmint_utils
import scipy.stats
import pymongo
import joblib
import align_dataset

# Path to corrupted datasets, created by create_data.py
CORRUPTED_EASY = 'data/corrupted_easy/*.npz'
CORRUPTED_HARD = 'data/corrupted_hard/*.npz'


def check_one_seed(best_errors, seed, database, collection):
    """Compute confidence score effectiveness for one parameter search seed.

    Parameters
    ----------
    best_errors : np.ndarray
        Array of the per-song errors resulting from the best aligner
    seed : int
        Parameter search random seed to retrieve results from
    database : str
        Which database in the mongodb to use
    collection : str
        Which collection in the mongodb to use
    """
    # Load a pymongo client which will be used to write results
    client = pymongo.MongoClient()
    # Load in corrupted MIDI datasets
    easy_dataset = align_dataset.load_dataset(CORRUPTED_EASY)
    hard_dataset = align_dataset.load_dataset(CORRUPTED_HARD)
    # Load in all parameter search experiment results for this seed
    param_settings, objectives = spearmint_utils.get_experiment_results(
        'alignment_search_seed_{}'.format(seed))
    # Sort the parameter settings by their objective value
    param_settings = [param_settings[n] for n in np.argsort(objectives)]
    for params in param_settings:
        # Get the results of aligning the easy dataset with these params
        easy_results = align_dataset.align_dataset(params, easy_dataset)
        # Retrieve the errors for each song
        easy_errors = np.array([r['mean_error'] for r in easy_results])
        # Run a paired difference test of the errors, i.e. test whether the
        # distribution of differences between best_errors[n] and
        # easy_errors[n] is significantly different from 0 under a t-test
        _, r_score = scipy.stats.ttest_1samp(best_errors - easy_errors, 0)
        # Align the hard dataset using these params and retrieve errors
        hard_results = align_dataset.align_dataset(params, hard_dataset)
        hard_errors = np.array([r['mean_error'] for r in hard_results])
        # Create results dict, storing the r_csore and errors, plus stuff below
        result = dict(r_score=r_score,
                      easy_errors=easy_errors.tolist(),
                      hard_errors=hard_errors.tolist(),
                      **params)
        # Try all combinations of score normalization
        for include_pen in [0, 1]:
            for length_normalize in [0, 1]:
                for mean_normalize in [0, 1]:
                    # Construct a name for this normalization scheme
                    name = ''
                    # Retrieve the score with or without penalties included
                    if include_pen:
                        scores = np.array(
                            [r['raw_score'] for r in hard_results])
                        name += 'raw'
                    else:
                        scores = np.array(
                            [r['raw_score_no_penalty'] for r in hard_results])
                        name += 'no_penalty'
                    # Optionally normalize by path length
                    if length_normalize:
                        scores /= np.array(
                            [r['path_length'] for r in hard_results])
                        name += '_len_norm'
                    # Optionally normalize by distance matrix mean
                    if mean_normalize:
                        scores /= np.array(
                            [r['distance_matrix_mean'] for r in hard_results])
                        name += '_mean_norm'
                    # Compute rank correlation coefficient
                    rank_corr = scipy.stats.spearmanr(hard_errors, scores)
                    # Store the rank correlation coefficient
                    result[name + '_rank_corr'] = rank_corr
                    # Store the scores
                    result[name + '_scores'] = scores.tolist()
        # Write resultt to database
        client[database][collection].insert(result)
    # Close the mongodb connection
    client.close()

if __name__ == '__main__':
    # Retrieve the best parameter settings/objectives for each seed used
    best_params = []
    best_objectives = []
    for n in range(10):
        p, o = spearmint_utils.get_best_result(
            'alignment_search_seed_{}'.format(n))
        best_params.append(p)
        best_objectives.append(o)
    # Find the best result among the best results for all seeds
    best_params = best_params[np.argmin(best_objectives)]
    easy_dataset = align_dataset.load_dataset(CORRUPTED_EASY)
    # Get the mean error on the easy dataset for this result
    best_results = align_dataset.align_dataset(best_params, easy_dataset)
    best_errors = [r['mean_error'] for r in best_results]

    # Run check_one_seed for all seeds in parallel
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(check_one_seed)(
            best_errors, seed, 'confidence_experiment', 'results')
        for seed in range(10))
