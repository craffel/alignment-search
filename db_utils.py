""" Utility functions to retrieve results from tinydb """

import numpy as np
try:
    import ujson as json
except ImportError:
    import json
import glob
import os


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


def dump_result(params, results, output_path):
    """Writes out a single result .json file in output_path.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values
    results : dict
        Dictionary of an alignment result
    output_path : str
        Where to write out the json file
    """
    # ujson can't handle infs, so we need to replace them manually:
    if params['norm'] == np.inf:
        params['norm'] = str(np.inf)
    # Convert params dict to a string of the form
    # param1_name_param1_value_param2_name_param2_value...
    param_string = "_".join(
        '{}_{}'.format(name, value) if type(value) != float else
        '{}_{:.3f}'.format(name, value) for name, value in params.items())
    # Construct a path where the .json results file will be written
    output_filename = os.path.join(output_path, "{}.json".format(param_string))
    # Store this result
    try:
        with open(output_filename, 'wb') as f:
            json.dump({'params': params, 'results': results}, f)
    # Ignore "OverflowError"s raised by ujson; they correspond to inf/NaN
    except OverflowError:
        pass
