import glob
import tabulate
import numpy as np
import numba
import ujson as json
import collections


@numba.jit(nopython=True)
def kendall_ignore_ties(x, y):
    """ Compute Kendall's rank correlation coefficient, ignoring ties.

    Parameters
    ----------
    x, y : np.ndarray
        Samples from two different random variables.

    Returns
    -------
    tau : float
        Kendall's rank correlation coefficient, ignoring ties.
    """
    # Accumulate numerator and denominator as we go
    numer = 0.
    denom = 0.
    # Compare all pairs
    for i in xrange(1, x.shape[0]):
        for j in xrange(0, i):
            # When either pair is equal, ignore
            if x[i] == x[j] or y[i] == y[j]:
                continue
            else:
                # Add one when both x[i] < x[j] and y[i] < y[j]
                # or x[i] > x[j] and y[i] > y[j],
                # otherwise add -1
                numer += np.sign(x[i] - x[j])*np.sign(y[i] - y[j])
                # Add 1 more sample considered
                denom += 1
    # Divide to compute tau
    return numer/denom

if __name__ == '__main__':

    # Load in all confidence experiment results
    results = []
    for result_file in glob.glob('results/confidence_experiment/*.json'):
        with open(result_file) as f:
            results.append(json.load(f))

    # Create a list which stores the performance of each aligner tried
    aligner_performance = []
    for result in results:
        # Retrieve the parameters, to report later
        params = result['params']
        # Retrieve the result, for less verbosity
        result = result['results']
        # Only consider aligners with small r-score
        if result['r_score'] > .05:
            # Combine the errors for the easy and hard datasets
            errors = np.array(result['hard_errors'] + result['easy_errors'])
            # Retrieve the reported aligner scores for all reported errors
            scores = dict((k, v) for k, v in result.items() if 'scores' in k)
            # Combine hard and easy scores
            scores = dict((k, np.array(v + scores[k.replace('hard', 'easy')]))
                          for k, v in scores.items() if 'hard' in k)
            # Compute rank correlation coefficients for all scores
            rank_corrs = dict((k, kendall_ignore_ties(errors, s))
                              for k, s in scores.items())
            # Find the name and score of the best-correlating score
            best_name = max(rank_corrs, key=rank_corrs.get)
            best_score = rank_corrs[best_name]
            # Store the performance of this aligner
            aligner_performance.append(collections.OrderedDict([
                ('hard_error', np.mean(result['hard_errors'])),
                ('easy_error', np.mean(result['easy_errors'])),
                ('r_score', result['r_score']),
                ('best_name', best_name),
                ('best_score', best_score),
                ('params', params)]))
    # Sort aligners by their best_score, descendin (
    aligner_performance.sort(key=lambda x: x['best_score'], reverse=True)
    # Print table of all aligners
    print tabulate.tabulate(
        aligner_performance[:20],
        headers=dict((k, k) for k in aligner_performance[0]))
