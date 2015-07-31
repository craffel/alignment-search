'''
Code for aligning an entire dataset
'''

import glob
import scipy.spatial
import librosa
import os
import numpy as np
import create_data
import djitw
import collections


def load_dataset(file_glob):
    """Load in a collection of feature files created by create_data.py.

    Parameters
    ----------
    file_glob : str
        Glob string for .npz files to load.

    Returns
    -------
    data : list of dict
        Loaded dataset, sorted by filename.
    """
    # Load in all npz's, casted to dict to force full loading
    return [dict(feature_file=os.path.abspath(d), **np.load(d))
            for d in sorted(glob.glob(file_glob))]


def align_dataset(params, data):
    '''
    Perform alignment of all corrupted MIDIs in the database given the supplied
    parameters and compute the mean alignment error across all examples

    Parameters
    ----------
    params : dict
        Dictionary of alignment parameters.

    data : list of dict
        Collection of things to align, loaded via load_dataset.

    Returns
    -------
    results : list of dict
        List of dicts reporting the results for each alignment
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
    # List for storing the results of each alignment
    results = collections.defaultdict(list)
    for n, d in enumerate(data):
        # Post proces the chosen feature matrices
        orig_gram = post_process_features(
            d['orig_gram'], d['orig_beat_frames'])
        corrupted_gram = post_process_features(
            d['corrupted_gram'], d['corrupted_beat_frames'])
        # Compute a distance matrix according to the supplied metric
        distance_matrix = scipy.spatial.distance.cdist(
            orig_gram, corrupted_gram, params['metric'])
        # Set any Nan/inf values to the largest distance
        distance_matrix[np.logical_not(np.isfinite(distance_matrix))] = np.max(
            distance_matrix[np.isfinite(distance_matrix)])
        # Compute a band mask or set to None for no mask
        if params['band_mask']:
            mask = np.zeros(distance_matrix.shape, dtype=np.bool)
            djitw.band_mask(1 - params['gully'], mask)
        else:
            mask = None
        # Get DTW path and score
        add_pen = params['add_pen']*np.median(distance_matrix)
        p, q, score = djitw.dtw(
            distance_matrix, params['gully'], add_pen, mask=mask, inplace=0)
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
        mean_error = np.mean(np.abs(error))
        # If the mean error is NaN or inf for some reason, set it to max (.5)
        if not np.isfinite(mean_error):
            mean_error = .5
        results['mean_errors'].append(mean_error)
        results['raw_scores'].append(score)
        results['raw_scores_no_penalty'].append(distance_matrix[p, q].sum())
        results['path_lengths'].append(p.shape[0])
        results['distance_matrix_means'].append(np.mean(
            distance_matrix[p.min():p.max() + 1, q.min():q.max() + 1]))
        results['feature_files'].append(os.path.basename(d['feature_file']))
    return results
