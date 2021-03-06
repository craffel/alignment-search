'''
Methods for corrupting a MIDI file in real-world-ish ways.
'''
import numpy as np


def warp_time(original_times, std):
    """
    Computes a random smooth time offset.

    Parameters
    ----------
    original_times : np.ndarray
        Array of original times, to be warped
    std : float
        Standard deviation of smooth noise.

    Returns
    -------
    warp_offset : np.ndarray
        Smooth time warping offset, to be applied by addition
    """
    N = original_times.shape[0]
    # Invert a random spectra, with most energy concentrated
    # on low frequencies via exponential decay
    warp_offset = np.fft.irfft(
        N*std*np.random.randn(N)*np.exp(-np.arange(N)))[:N]
    return warp_offset


def crop_time(midi_object, original_times, start, end):
    """
    Crop times out of a MIDI object

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI object, notes will be modified to allow correct cropping
        with adjust_times.
    original_times : np.ndarray
        Array of original times, to be cropped
    start : float
        Start time of the crop
    end : float
        End time of the crop

    Returns
    -------
    crop_offset : np.ndarray
        Time offset to be applied by addition
    """
    for inst in midi_object.instruments:
        # Remove all notes within the interval we're cropping out
        inst.notes = [note for note in inst.notes if not (
            note.start > start and note.start < end and
            note.end > start and note.end < end)]
        for note in inst.notes:
            # If the note starts before the interval and ends within the
            # interval truncate it so that it ends at the start of the interval
            if note.start < start and note.end > start and note.end < end:
                note.end = start
            # If the note starts within the interval and ends after the
            # interval move the start to the end of the interval.
            elif note.start > start and note.start < end and note.end > end:
                note.start = end
        # Move all events within the interval to the start
        for events in [inst.control_changes, inst.pitch_bends]:
            for event in events:
                if event.time > start and event.time < end:
                    event.time = start
    # The crop offset is just the difference in timing,
    time_offset = np.zeros(original_times.shape[0])
    # applied after the interval starts.
    time_offset[original_times >= start] -= (end - start)
    return time_offset


def corrupt_instruments(midi_object, probability):
    '''
    Randomly adjust the program numbers of instruments by +/-1.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI object; program numbers will be randomly adjusted.
    probability : float
        Probability \in [0, 1] that the program number will be adjusted.
    '''
    for instrument in midi_object.instruments:
        # Ignore drum instruments; changing their prog is futile
        if not instrument.is_drum:
            # Use the applied probability
            if np.random.rand() < probability:
                # Randomly add or subtract one
                new_prog = instrument.program + np.random.choice([-1, 1])
                # Handle edge cases
                if new_prog == -1:
                    new_prog = 1
                elif new_prog == 128:
                    new_prog = 127
                # Overwrite the program number
                instrument.program = new_prog


def remove_instruments(midi_object, probability):
    '''
    Randomly remove instruments from a MIDI object.
    Will never allow there to be zero instruments.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI object; instruments will be randomly removed
    probability : float
        Probability of removing an instrument.
    '''
    # Pick a random subset of the instruments
    random_insts = [inst for inst in midi_object.instruments
                    if np.random.rand() > probability]
    # Don't allow there to be 0 instruments
    if len(random_insts) == 0:
        midi_object.instruments = [np.random.choice(midi_object.instruments)]
    else:
        midi_object.instruments = random_insts


def corrupt_velocity(midi_object, std):
    '''
    Randomly corrupt the velocity of all notes.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI object; velocity will be randomly adjusted.
    std : float
        Velocities will be multiplied by N(1, std)
    '''
    for instrument in midi_object.instruments:
        for note in instrument.notes:
            # Compute new velocity by scaling by N(1, std)
            new_velocity = note.velocity*(np.random.randn()*std + 1)
            # Clip to the range [0, 127], convert to int, and save
            note.velocity = int(np.clip(new_velocity, 0, 127))


def corrupt_midi(midi_object, original_times, warp_std,
                 start_crop_prob, end_crop_prob,
                 middle_crop_prob, remove_inst_prob,
                 change_inst_prob, velocity_std):
    '''
    Apply a series of corruptions to a MIDI object.

    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        MIDI object, will be corrupted in place.
    original_times : np.ndarray
        Array of original sampled times.
    warp_std : float
        Standard deviation of random smooth noise offsets.
    start_crop_prob : float
        Probability of cutting out the first 10% of the MIDI object.
    end_crop_prob : float
        Probability of cutting out the final 10% of the MIDI object.
    middle_crop_prob : float
        Probability of cutting out 1% of the MIDI object somewhere.
    remove_inst_prob : float
        Probability of removing instruments.
    change_inst_prob : float
        Probability of randomly adjusting instrument program numbers by +/-1.
    velocity_std : float
        Standard deviation of multiplicative scales to apply to velocities.

    Returns
    -------
    adjusted_times : np.ndarray
        `original_times` adjusted by the cropping
    diagnostics : dict
        Diagnostics about the corruptions applied
    '''
    # Store all keyword arguments as diagnostics
    diagnostics = dict((k, v) for (k, v) in locals().iteritems()
                       if isinstance(v, (int, long, float)))
    # Smoothly warp times
    warp_offset = warp_time(original_times, warp_std)
    # Start with no cropping offset, as it will depend on the probabilities
    crop_offset = np.zeros(original_times.shape[0])
    # Store whether we are cropping out the beginning
    diagnostics['crop_start'] = np.random.rand() < start_crop_prob
    if diagnostics['crop_start']:
        # Crop out the first 10%
        end_time = .1*original_times[-1]
        crop_offset += crop_time(midi_object, original_times, 0, end_time)
    diagnostics['crop_end'] = np.random.rand() < end_crop_prob
    if diagnostics['crop_end']:
        # Crop out the last 10%
        start_time = .9*original_times[-1]
        crop_offset += crop_time(
            midi_object, original_times, start_time, original_times[-1])
    diagnostics['crop_middle'] = np.random.rand() < middle_crop_prob
    if diagnostics['crop_middle']:
        # Randomly crop out 1% from somewhere in the middle
        rand = np.random.rand()
        offset = original_times[-1]*(rand*.8 + .1)
        crop_offset += crop_time(
            midi_object, original_times, offset,
            offset + .01*original_times[-1])
    # Store the number of instruments originally, and after optionally removing
    diagnostics['n_instruments_before'] = len(midi_object.instruments)
    # Randomly remove instruments
    remove_instruments(midi_object, remove_inst_prob)
    diagnostics['n_instruments_after'] = len(midi_object.instruments)
    # Corrupt their program numbers
    corrupt_instruments(midi_object, change_inst_prob)
    # Adjust velocity randomly
    corrupt_velocity(midi_object, velocity_std)
    # Apply the time warps computed above
    adjusted_times = original_times + warp_offset + crop_offset
    midi_object.adjust_times(original_times, adjusted_times)
    return adjusted_times, diagnostics
