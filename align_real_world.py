""" Align all of the real-world data. """
import djitw
import numpy as np
import pretty_midi
import librosa
import create_data
import json
import scipy.spatial
import os
import joblib

GULLY = .96
REAL_WORLD_PATH = 'data/real_world'


def process_one_file(audio_file, midi_file, output_midi_file, pair_file,
                     diagnostics_file):
    """
    Wrapper routine for loading in audio/MIDI data, aligning, and writing
    out the result.

    Parameters
    ----------
    audio_file, midi_file, output_midi_file, pair_file, diagnostics_file : str
        Paths to the audio file to align, MIDI file to align, and paths where
        to write the aligned MIDI, the synthesized pair file, and the DTW
        diagnostics file.
    """
    # Load in the audio data
    audio_data, _ = librosa.load(audio_file, sr=create_data.FS)
    # Compute the log-magnitude CQT of the data
    audio_cqt, audio_times = create_data.extract_cqt(audio_data)
    audio_cqt = librosa.logamplitude(audio_cqt, ref_power=audio_cqt.max()).T
    # Load and synthesize MIDI data
    midi_object = pretty_midi.PrettyMIDI(midi_file)
    midi_audio = midi_object.fluidsynth(fs=create_data.FS)
    # Compute log-magnitude CQT
    midi_cqt, midi_times = create_data.extract_cqt(midi_audio)
    midi_cqt = librosa.logamplitude(midi_cqt, ref_power=midi_cqt.max()).T
    # Compute cosine distance matrix
    distance_matrix = scipy.spatial.distance.cdist(
        midi_cqt, audio_cqt, 'cosine')
    # Get lowest cost path
    p, q, score = djitw.dtw(
        distance_matrix, GULLY, np.median(distance_matrix), inplace=False)
    # Normalize by path length
    score = score/len(p)
    # Normalize by distance matrix submatrix within path
    score = score/distance_matrix[p.min():p.max(), q.min():q.max()].mean()
    # Adjust the MIDI file
    midi_object.adjust_times(midi_times[p], audio_times[q])
    # Write the result
    midi_object.write(output_midi_file)
    # Synthesize aligned MIDI
    midi_audio_aligned = midi_object.fluidsynth(fs=create_data.FS)
    # Adjust to the same size as audio
    if midi_audio_aligned.shape[0] > audio_data.shape[0]:
        midi_audio_aligned = midi_audio_aligned[:audio_data.shape[0]]
    else:
        trim_amount = audio_data.shape[0] - midi_audio_aligned.shape[0]
        midi_audio_aligned = np.append(midi_audio_aligned,
                                       np.zeros(trim_amount))
    # Stack one in each channel
    librosa.output.write_wav(
        pair_file, np.array([midi_audio_aligned, audio_data]), create_data.FS)
    # Write out diagnostics
    with open(diagnostics_file, 'wb') as f:
        json.dump({'p': list(p), 'q': list(q), 'score': score}, f)

if __name__ == '__main__':
    # Utility function for getting lists of all files of a certain type
    def get_file_list(extension):
        return [os.path.join(REAL_WORLD_PATH, '{}{}'.format(n, extension))
                for n in range(1000)]
    # Construct lists of each type of files
    audios = get_file_list('.mp3')
    mids = get_file_list('.mid')
    out_mids = get_file_list('-aligned.mid')
    pairs = get_file_list('-pair.wav')
    diags = get_file_list('-diagnostics.js')
    # Process each file from each list in parallel
    joblib.Parallel(n_jobs=20, verbose=51)(
        joblib.delayed(process_one_file)(audio, mid, output_mid, pair, diag)
        for (audio, mid, output_mid, pair, diag)
        in zip(audios, mids, out_mids, pairs, diags))
