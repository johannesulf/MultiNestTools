import os
import pickle
import numbers
import numpy as np
from functools import partial


def var_param_dict(name, fit=True, prior='uniform', default=None, tex=None,
                   **kwargs):
    """Create a dictionary that describes a single parameter of the fit.

    Parameters
    ----------
    fit : boolean
        Whether the parameter is varied in the fit.
    prior : string
        The prior used in the fit.
    default : float
        The value used if the parameter is kept fixed.
    tex : string
        LaTeX string describing the parameter.
    **kwargs : dict
        Any additional keywords passed to the prior.
    Returns
    -------
    dict
        Dictionary that describes the parameter of the fit.
    """
    return {'name': name, 'fit': fit, 'prior': prior, 'tex': tex,
            'default': default, **kwargs}


def check_var_param_list(var_param_list):
    try:
        assert isinstance(var_param_list, list)
        for var_param in var_param_list:
            assert isinstance(var_param, dict)
            assert 'name' in var_param
            assert isinstance(var_param['name'], str)
            assert 'fit' in var_param
            assert isinstance(var_param['fit'], bool)
    except AssertionError:
        raise RuntimeError('The fit parameters are not a list of ' +
                           'dictionaries with all required keywords.')


def write_var_param_list(var_param_list, directory):

    check_var_param_list(var_param_list)

    fstream = open(os.path.join(directory, 'var_param_list.pkl'), 'wb')
    pickle.dump(var_param_list, fstream)
    fstream.close()


def read_var_param_list(directory):

    return pickle.load(
        open(os.path.join(directory, 'var_param_list.pkl'), 'rb'))


def _prior(var_param_list, cube, n_dim, n_params):

    i = 0

    for var_param in var_param_list:

        if not var_param['fit']:
            continue

        if var_param['prior'] == 'uniform':
            cube[i] = (
                cube[i] * (var_param['max'] - var_param['min']) +
                var_param['min'])

        i += 1


def prior(var_param_list):
    check_var_param_list(var_param_list)
    return partial(_prior, var_param_list)


def vector_to_full_vector(vector, var_param_list):

    full_vector = np.zeros(len(var_param_list)) * np.nan

    # First, fill the values that were fitted.
    for i, var_param in enumerate(var_param_list):
        if var_param['fit']:
            full_vector[i] = vector[
                np.sum([param['fit'] for param in var_param_list[:i]],
                       dtype=np.int)]

    # Next, fill all values that were not fitted.
    for i, var_param in enumerate(var_param_list):
        if not var_param['fit']:
            if isinstance(var_param['default'], numbers.Number):
                full_vector[i] = var_param['default']
            else:
                name_of_default = var_param['default']
                for k in range(len(var_param_list)):
                    if var_param_list[k]['name'] == name_of_default:
                        break
                full_vector[i] = full_vector[k]

    return full_vector


def vector_to_param_dict(vector, var_param_list):

    full_vector = vector_to_full_vector(vector, var_param_list)

    param_dict = {}
    for i, var_param in enumerate(var_param_list):
        param_dict[var_param['name']] = full_vector[i]

    return param_dict


def read_posterior(directory, format='vector', weight='equal',
                   n_samples='max'):

    if weight == 'unequal':
        ev = np.genfromtxt(os.path.join(directory, 'ev.dat'))
        vectors_ev = ev[:, :-3]
        weights_ev = (ev[:, -3] + ev[:, -2])
        vol_min = ev[-1, -2]

        live = np.genfromtxt(os.path.join(directory, 'phys_live.points'))
        vectors_live = live[:, :-2]
        weights_live = np.repeat(vol_min, len(live)) + live[:, -2]

        vectors = np.concatenate([vectors_ev, vectors_live])
        weights = np.concatenate([weights_ev, weights_live])

        weights = weights - np.amax(weights)
        weights = np.exp(weights)

    else:
        vectors = np.genfromtxt(os.path.join(
            directory, 'post_equal_weights.dat'))[:, :-1]

    if isinstance(n_samples, int):
        vectors = vectors[np.random.randint(len(vectors), size=n_samples)]

    if format == 'vector':
        output = vectors
    elif format == 'param_dict':
        var_param_list = read_var_param_list(directory)
        output = [vector_to_param_dict(vector, var_param_list) for vector in
                  vectors]

    if weight == 'unequal':
        return output, weights
    else:
        return output


def read_best_fit(directory, format='vector'):

    live = np.genfromtxt(os.path.join(directory, 'phys_live.points'))
    vectors_live = live[:, :-2]
    vector = vectors_live[np.argmax(live[:, -2])]

    if format == 'vector':
        output = vector
    elif format == 'param_dict':
        var_param_list = read_var_param_list(directory)
        output = vector_to_param_dict(vector, var_param_list)

    return output


def tex_labels_of_fit(var_param_list):
    labels = np.array([var_param['tex'] for var_param in var_param_list])
    return labels[[var_param['fit'] for var_param in var_param_list]]
