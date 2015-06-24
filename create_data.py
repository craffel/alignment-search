'''
Creates .npz archives of corrupted MIDI file features.
'''
import numpy as np
import pretty_midi
import librosa
import corrupt_midi
import tempfile
import subprocess
import os
import itertools
import sys
import argparse
import glob
import traceback
import joblib

FS = 22050
NOTE_START = 36
N_NOTES = 48


def extract_cqt(audio_data):
    '''
    CQT routine with default parameters filled in, and some post-processing.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio data to compute CQT of

    Returns
    -------
    cqt : np.ndarray
        CQT of the supplied audio data.
    frame_times : np.ndarray
        Times, in seconds, of each frame in the CQT
    '''
    # Compute CQT
    cqt = librosa.cqt(audio_data, sr=FS, fmin=librosa.midi_to_hz(NOTE_START),
                      n_bins=N_NOTES, tuning=0.)
    # Compute the time of each frame
    times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=FS)
    # Use float32 for the cqt to save space/memory
    cqt = cqt.astype(np.float32)
    return cqt, times


def fast_fluidsynth(midi_object):
    '''
    Faster fluidsynth synthesis using the command-line program instead of
    pyfluidsynth.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        PrettyMIDI object

    Returns
    -------
    midi_audio : np.ndarray
        Synthesized audio
    '''
    # Write out temp mid file
    temp_mid = tempfile.NamedTemporaryFile()
    midi_object.write(temp_mid.name)
    # Get path to temporary .wav file
    temp_wav = tempfile.NamedTemporaryFile()
    # Get path to default pretty_midi SF2
    sf2_path = os.path.join(os.path.dirname(pretty_midi.__file__),
                            pretty_midi.DEFAULT_SF2)
    # Make system call to fluidsynth
    with open(os.devnull, 'w') as devnull:
        subprocess.check_output(
            ['fluidsynth', '--fast-render={}'.format(temp_wav.name),
             '-r', str(FS), sf2_path, temp_mid.name], stderr=devnull)
    # Load from temp wav file
    audio, _ = librosa.load(temp_wav.name, sr=FS)
    # Close/delete temp files
    temp_mid.close()
    temp_wav.close()
    return audio


def extract_features(midi_object):
    '''
    Main feature extraction routine for a MIDI file

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        PrettyMIDI object to extract features from

    Returns
    -------
    features : dict
        Dictionary of features
    '''
    # Synthesize the midi object
    midi_audio = fast_fluidsynth(midi_object)
    # Compute constant-Q transform
    gram, times = extract_cqt(midi_audio)
    # Compute chromagrapm from constant-Q
    chroma = librosa.feature.chroma_cqt(
        C=gram, fmin=librosa.midi_to_hz(NOTE_START))
    # Estimate the tempo from the MIDI data
    tempo = midi_object.estimate_tempo()
    # Usually, estimate_tempo gives tempos around 200 bpm, which is usually
    # double time, which we want.  Sometimes, it's not, so we double it.
    while tempo < 160:
        tempo *= 2
    # Estimate the beats, forcing the tempo to be near the MIDI tempo
    beat_frames = librosa.beat.beat_track(midi_audio, bpm=tempo)[1]
    beat_times = librosa.frames_to_time(beat_frames, sr=FS)

    return {'times': times, 'gram': gram, 'chroma': chroma, 'beat_frames':
            beat_frames, 'beat_times': beat_times}


def process_one_file(midi_filename, output_path):
    '''
    Create features and diagnostics dict for original and corrupted MIDI file

    Parameters
    ----------
    midi_filename : str
       Path to a MIDI file to corrupt.
    output_path : str
        Base path to write out .npz/.mid

    Returns
    -------
    features : dict
        Features of original and corrupted MIDI, with diagnostics
    '''
    try:
        # Load in and extract features/diagnostic information for the file
        midi_object = pretty_midi.PrettyMIDI(midi_filename)
        orig_features = extract_features(midi_object)
        # Prepend keys with 'orig'
        orig_features = dict(
            ('orig_{}'.format(k), v) for (k, v) in orig_features.iteritems())
        # Corrupt MIDI object (in place)
        adjusted_times, diagnostics = corrupt_midi.corrupt_midi(
            midi_object, orig_features['orig_times'])
        # Get features for corrupted MIDI
        corrupted_features = extract_features(midi_object)
        corrupted_features = dict(('corrupted_{}'.format(k), v)
                                for (k, v) in corrupted_features.iteritems())
        # Combine features, diagnostics into one fat dict
        data = dict(i for i in itertools.chain(
            orig_features.iteritems(), [('adjusted_times', adjusted_times)],
            diagnostics.iteritems(), corrupted_features.iteritems()))
        data['original_file'] = os.path.abspath(midi_filename)
        corrupted_filename = os.path.abspath(os.path.join(
            output_path, os.path.basename(midi_filename)))
        midi_object.write(corrupted_filename)
        data['corrupted_file'] = corrupted_filename
        # Write out the npz
        output_npz = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(midi_filename))[0] + '.npz')
        np.savez_compressed(output_npz, **data)
    except Exception:
        print "Error parsing {}:".format(midi_filename)
        traceback.print_exc()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Create a dataset of corrupted MIDI information.')
    parser.add_argument('midi_glob', action='store',
                        help='Glob to MIDI files (e.g. data/mid/*/*.mid)')
    parser.add_argument('output_path', action='store',
                        help='Where to output .npz files')
    parameters = vars(parser.parse_args(sys.argv[1:]))
    # Create the output directory if it doesn't exist
    if not os.path.exists(parameters['output_path']):
        os.makedirs(parameters['output_path'])
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(process_one_file)(midi_file, parameters['output_path'])
        for midi_file in glob.glob(parameters['midi_glob']))
