""" Utility functions to do useful things with Spearmint """

import pymongo
import spearmint.utils.compression


def get_experiment_results(experiment_name):
    """Get a list of all results of an experiment from the mongo database.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (database collection)

    Returns
    -------
    params : list
        List of dicts, where each entry is the dict of parameter name/value
        associations for the corresponding objective value in objectives
    objectives : list
        List of float, where each entry is the objective value for the
        corresponding parameter settings in params
    """
    # Retrieve the mongo collection for this experiment
    client = pymongo.MongoClient('localhost', 27017)
    db = client['spearmint']
    collection = db['{}.jobs'.format(experiment_name)]
    params, objectives = [], []
    # Retrieve all results from the collection
    for result in collection.find():
        # Only retrieve completed experiments, which are the ones which have an
        # objective value in 'values'
        if 'values' in result:
            # Populate a dict of parameters for this result
            result_params_out = {}
            for key, result_params in result['params'].items():
                # When the parameter settings entry ('values') is a dict, this
                # means it's a compressed array, so decompress item
                if type(result_params['values']) == dict:
                    result_params_out[key] = \
                        spearmint.utils.compression.decompress_array(
                            result_params['values'])[0]
                else:
                    # In both cases, params are stored as a 1-entry array
                    result_params_out[key] = result_params['values'][0]
            # Add in this result
            params.append(result_params_out)
            objectives.append(result['values']['main'])
    return params, objectives
