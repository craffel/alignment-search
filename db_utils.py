""" Utility functions to retrieve results from tinydb """

import numpy as np
try:
    import ujson as json
except ImportError:
    import json
import glob


def get_experiment_results(results_glob):
    """Get a list of all results of an experiment from the tinydb table.

    Parameters
    ----------
    results_glob : str
        Glob to results .json files

    Returns
    -------
    params : list
        List of dicts, where each entry is the dict of parameter name/value
        associations for the corresponding objective value in objectives
    objectives : list
        List of float, where each entry is the objective value for the
        corresponding parameter settings in params
    """
    results = []
    for result_file in glob.glob(results_glob):
        with open(result_file) as f:
            results.append(json.load(f))
    params = [result['params'] for result in results]
    objectives = [np.mean(r['results']['mean_errors']) for r in results]
    return params, objectives


def get_best_result(results_glob):
    """Get the parameters and objective corresponding to the best result for an
    experiment.

    Parameters
    ----------
    results_glob : str
        Glob to results .json files

    Returns
    -------
    params : dict
        dict of parameter name/value associations for the best objective value
    objective : float
        Best achived objective value, corresponding to the parameter settings
        in params
    """
    # This function is just a wrapper around using argmin
    params, objectives = get_experiment_results(results_glob)
    best_result = np.argmin(objectives)
    return params[best_result], objectives[best_result]
