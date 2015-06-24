'''
Run the alignment parameter search experiment.
'''

import hyperopt
from hyperopt import hp
import scipy.spatial
import glob
import numpy as np
import scipy.stats
import librosa
import djitw
import create_data

# Path to corrupted dataset, created by create_data.py
CORRUPTED_PATH = 'data/corrupted_easy/*.npz'
# Load in all npz's, casted to dict to force full loading
data = [dict(np.load(d)) for d in glob.glob(CORRUPTED_PATH)]


def objective(params):
    '''
    Perform alignment of all corrupted MIDIs in the database given the supplied
    parameters and compute the mean alignment error across all examples

    Parameters
    ----------
    params : dict
        Dictionary of alignment parameters.

    Returns
    -------
    result : dict
        Dictionary reporting the results of the alignment
    '''
    def post_process_features(gram, beats):
        '''
        Apply processing to a feature matrix given the supplied param values

        Parameters
        ----------
        gram : np.ndarray
            Feature matrix, shape (n_features, n_samples)
        beats : np.ndarray
            Indices of beat locations in gram

        Returns
        -------
        gram : np.ndarray
            Feature matrix, shape (n_samples, n_features), post-processed
            according to the values in `params`
        '''
        # Convert to chroma
        if params['feature'] == 'chroma':
            gram = librosa.feature.chroma_cqt(
                C=gram, fmin=librosa.midi_to_hz(create_data.NOTE_START))
        # Beat-synchronize the feature matrix
        if params['beat_sync']:
            gram = librosa.feature.sync(gram, beats, pad=False)
        # Compute log magnitude
        if params['log']:
            gram = librosa.logamplitude(gram, ref_power=gram.max())
        # Normalize the feature vectors
        gram = librosa.util.normalize(gram, norm=params['norm'])
        # Standardize the feature vectors
        if params['standardize']:
            gram = scipy.stats.mstats.zscore(gram, axis=1)
        # Transpose it to (n_samples, n_features) and return it
        return gram.T
    # Pre-allocate array for storing the mean error for each corrupted MIDI
    mean_errors = np.zeros(len(data))
    for n, d in enumerate(data):
        # Post proces the chosen feature matrices
        orig_gram = post_process_features(
            d['orig_gram'], d['orig_beat_frames'])
        corrupted_gram = post_process_features(
            d['corrupted_gram'], d['corrupted_beat_frames'])
        # Compute a distance matrix according to the supplied metric
        distance_matrix = scipy.spatial.distance.cdist(
            orig_gram, corrupted_gram, params['metric'])
        # Get DTW path and score
        add_pen = np.percentile(distance_matrix, params['add_pen'])
        p, q, score = djitw.dtw(distance_matrix, params['gully'], add_pen)
        if params['beat_sync']:
            # If we are beat syncing, we have to compare against beat times
            # so we index adjusted_times by the beat indices
            adjusted_times = d['adjusted_times'][d['orig_beat_frames']]
            corrupted_times = d['corrupted_beat_times']
        else:
            corrupted_times = d['corrupted_times']
            adjusted_times = d['adjusted_times']
        # Compute the error, clipped to within .5 seconds
        error = np.clip(
            corrupted_times[q] - adjusted_times[p], -.5, .5)
        # Compute the mean error for this MIDI
        mean_errors[n] = np.mean(np.abs(error))
    return {'status': hyperopt.STATUS_OK,
            'loss': np.mean(mean_errors)}


if __name__ == '__main__':
    space = {
        # Use chroma or CQT for feature representation
        'feature': hp.choice('feature', ['chroma', 'gram']),
        # Beat sync, or don't
        'beat_sync': hp.choice('beat_sync', [True, False]),
        # Don't normalize, max-norm, L1-norm, or L2 norm
        'norm': hp.choice('norm', [None, 'inf', 1, 2]),
        # Whether or not to z-score (standardize) the feature dimensions
        'standardize': hp.choice('standardize', [True, False]),
        # Compute the log magnitude of the features
        'log': hp.choice('log', [True, False]),
        # Which distance metric to use for distance matrix
        'metric': hp.choice('metric', ['euclidean', 'sqeuclidean', 'cosine']),
        # DTW additive penalty
        'add_pen': hp.randint('add_pen', 101),
        # DTW end point tolerance
        'gully': hp.uniform('gully', 0., 1.)}

    # Run hyperopt
    print hyperopt.fmin(
        objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
