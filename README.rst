alignment-search
================

Code for searching for good MIDI aliginment parameters.  Incomplete dependency list:

* `pretty_midi <https://github.com/craffel/pretty-midi>`_
* `djitw <https://github.com/craffel/djitw>`_
* `librosa <https://github.com/bmcfee/librosa>`_
* `joblib <https://github.com/joblib/joblib>`_
* `craffel/spearmint/main_args <https://github.com/craffel/Spearmint/tree/main_args>`_
* Scipy/numpy

Tentative general procedure:

#. Collect 1000 non-corrupt MIDI files
#. Create a corrupted version for each MIDI file, which reflects common minor issues found in audio-to-audio alignment (see create_data.py and corrupt_midi.py)
#. Run GP-SMBO hyperparameter optimization over the standard DTW-based MIDI alignment scheme to choose the alignment architecture which best turns the lightly-corrupted MIDI files back into the original files (see experiment.py)
#. Create a second corrupted version for each MIDI file, which reflects quite major corruptions which we don't expect an alignment scheme to fix in all cases
#. Run hyperparameter optimization again, and either jointly optimize the alignment performance and a confidence score, OR optimize alignment performance and then find the score calculation scheme which produces the highest Spearman rank coefficient between the alignment score and the error of each alignment, over all highly-performing alignment architectures
#. Collect real-world audio/MIDI alignment pairs and run the best-performing alignment architecture on them, manually annotate whether it was successful or not, and find the ROC-AUC score for the confidence measure
