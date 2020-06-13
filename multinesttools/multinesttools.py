import os
import inspect
import corner
import pickle
import numbers
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


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


def read_posterior(directory, format=np.ndarray, equal_weight=True,
                   n_samples='max', logl=False):

    if equal_weight:
        vectors = np.genfromtxt(os.path.join(
            directory, 'post_equal_weights.dat'))
    else:
        ev = np.genfromtxt(os.path.join(directory, 'ev.dat'))
        vectors_ev = ev[:, :-2]
        weights_ev = (ev[:, -3] + ev[:, -2])
        vol_min = ev[-1, -2]

        live = np.genfromtxt(os.path.join(directory, 'phys_live.points'))
        vectors_live = live[:, :-1]
        weights_live = np.repeat(vol_min, len(live)) + live[:, -2]

        vectors = np.concatenate([vectors_ev, vectors_live])
        weights = np.concatenate([weights_ev, weights_live])

        weights = weights - np.amax(weights)
        weights = np.exp(weights)

    if isinstance(n_samples, int):
        vectors = vectors[np.random.randint(len(vectors), size=n_samples)]
    else:
        if not n_samples == 'max':
            raise RuntimeError('Cannot understand number of samples!' +
                               ' Received {}.'.format(n_samples))

    if not inspect.isclass(format):
        raise RuntimeError('format must be a class.')

    if format == np.ndarray:
        output = vectors
        if not logl:
            output = output[:, :-1]
    elif format == dict:
        var_param_list = read_var_param_list(directory)
        var_param_list.append(var_param_dict('log L', fit=True))
        output = [vector_to_param_dict(vector, var_param_list) for vector in
                  vectors]
    else:
        raise RuntimeError('Unkown output format. Received {}.'.format(
            format.__name__))

    if equal_weight:
        return output
    else:
        return output, weights


def read_best_fit(directory, format=np.ndarray):

    live = np.genfromtxt(os.path.join(directory, 'phys_live.points'))
    vectors_live = live[:, :-2]
    vector = vectors_live[np.argmax(live[:, -2])]

    if not inspect.isclass(format):
        raise RuntimeError('format must be a class.')

    if format == np.ndarray:
        output = vector
    elif format == dict:
        var_param_list = read_var_param_list(directory)
        output = vector_to_param_dict(vector, var_param_list)
    else:
        raise RuntimeError('Unkown output format. Received {}.'.format(
            format.__name__))

    return output


def read_max_log_likelihood(directory):

    live = np.genfromtxt(os.path.join(directory, 'phys_live.points'))
    return np.amax(live[:, -2])


def read_log_evidence(directory, ins=True):
    with open(os.path.join(directory, 'stats.dat')) as fstream:
        first_line = fstream.readline()
        second_line = fstream.readline()
        if ins:
            line = second_line
        else:
            line = first_line
        log_ev = float(line.split(":")[1].split("+/-")[0])
        fstream.close()
    return log_ev


def tex_labels_of_fit(var_param_list):
    labels = np.array([var_param['tex'] for var_param in var_param_list])
    return labels[[var_param['fit'] for var_param in var_param_list]]


def make_corner_plot(directory, equal_weight=False, truths=None):
    var_param_list = read_var_param_list(directory)
    labels = tex_labels_of_fit(var_param_list)
    if equal_weight:
        samples = read_posterior(directory, equal_weight=True)
        weights = np.ones(len(samples))
    else:
        samples, weights = read_posterior(directory, equal_weight=False)

    ndim = len(labels)
    fig, axes = plt.subplots(ndim, ndim, figsize=(7.0, 7.0))
    corner.corner(np.transpose(np.transpose(samples)),
                  weights=weights, plot_datapoints=False, plot_density=False,
                  labels=labels, color='royalblue', show_titles=False,
                  levels=(0.68, 0.95), bins=20, truths=truths,
                  range=np.ones(ndim) * 0.99, fill_contours=True, fig=fig,
                  hist_kwargs={'color': 'gold', 'histtype': 'stepfilled',
                               'edgecolor': 'black', 'linewidth': 0.5},
                  max_n_ticks=3, contour_kwargs={'linewidths': 0.5,
                                                 'colors': 'black'})

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for yi in range(ndim):
        for xi in range(yi + 1):
            ax = axes[yi, xi]
            ax.tick_params(axis='x', labelsize=8, rotation=90)
            ax.xaxis.set_label_coords(0.5, -0.5)
            ax.tick_params(axis='y', labelsize=8, rotation=0)
            ax.yaxis.set_label_coords(-0.5, 0.5)

    for yi in range(ndim):
        ax = axes[yi, yi]
        ax.tick_params(axis='y', which='both', left=False, right=False)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(directory, 'posterior.pdf'))
    plt.savefig(os.path.join(directory, 'posterior.png'), dpi=300)
    plt.close()
