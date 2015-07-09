'''
Run the alignment parameter search experiment.
'''

import spearmint.main
import os
import scipy.spatial
import glob
import numpy as np
import scipy.stats
import librosa
import djitw
import create_data
import argparse

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
        # Set any NaN values to the largest distance
        distance_matrix[np.isnan(distance_matrix)] = np.nanmax(distance_matrix)
        # Compute a band mask or set to None for no mask
        if params['band_mask']:
            mask = np.zeros(distance_matrix.shape, dtype=np.bool)
            djitw.band_mask(1 - params['gully'], mask)
        else:
            mask = None
        # Get DTW path and score
        add_pen = params['add_pen']*np.median(distance_matrix)
        p, q, score = djitw.dtw(
            distance_matrix, params['gully'], add_pen, mask=mask)
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
    return np.mean(mean_errors)


def main(job_id, params):
    # Spearmint requires (I think) all params to be passed as at least
    # 1-dimensional arrays.  So, get the first entry to flatten.
    for key, value in params.items():
        params[key] = value[0]
    return objective(params)


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
