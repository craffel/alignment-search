""" Utility functions to retrieve results from tinydb """

import numpy as np


def get_experiment_results(table):
    """Get a list of all results of an experiment from the tinydb table.

    Parameters
    ----------
    table : tinydb.database.Table
        TinyDB table instance to save results to

    Returns
    -------
    params : list
        List of dicts, where each entry is the dict of parameter name/value
        associations for the corresponding objective value in objectives
    objectives : list
        List of float, where each entry is the objective value for the
        corresponding parameter settings in params
    """
    params = [result['params'] for result in table.all()]
    objectives = [np.mean([alignment['mean_error']
                           for alignment in result['results']])
                  for result in table.all()]
    return params, objectives


def get_best_result(experiment_name):
    """Get the parameters and objective corresponding to the best result for an
    experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (database collection)

    Returns
    -------
    params : dict
        dict of parameter name/value associations for the best objective value
    objective : float
        Best achived objective value, corresponding to the parameter settings
        in params
    """
    # This function is just a wrapper around using argmin
    params, objectives = get_experiment_results(experiment_name)
    best_result = np.argmin(objectives)
    return params[best_result], objectives[best_result]
