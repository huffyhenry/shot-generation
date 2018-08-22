import os
import pickle

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
import pandas as pd
import pystan


def hdi(vector, width=0.5):
    x = sorted(vector)
    size = len(vector)
    left = int((1.0 - width)/2 * size)
    right = int((width + (1.0 - width)/2) * size)
    return x[left], x[right]


def fit(model_file, data, force_compile=False, **kwargs):
    """ Compile the model unless cached and run the sampler. """
    if not force_compile and os.path.exists("cache/%s.pickle" % model_file):
        model = pickle.load(open("cache/%s.pickle" % model_file, 'rb'))
    else:
        model = pystan.StanModel(model_file)
        pickle.dump(model, open("cache/%s.pickle" % model_file, 'wb'))

    return model.sampling(data, **kwargs)


def graph(samples, team_map, attack=True):
    """ Quick visualisation of the team coefficient samples. """
    def weibull_means(shape, scales):
        """ Return the mean of the Weibull distro, vectorized over scale. """
        return ss.gamma(1.0 + 1.0/shape)*scales

    plt.style.use('Solarize_Light2')

    # Extract the samples
    generation = samples['generation']
    prevention = samples['prevention']
    conversion = samples['conversion']
    obstruction = samples['obstruction']

    # Get the time dependence parameters
    quantity_k = np.mean(samples['quantity_k'])
    quality_k = np.mean(samples['quality_k'])

    labels = list(team_map.keys())  # Magically in the correct order

    # Select quantities to plot and make them interpretable
    if attack:
        xsamples = weibull_means(quantity_k, np.exp(generation + np.mean(prevention)))
        ysamples = ss.expit(conversion + np.mean(obstruction) + quality_k*xsamples)
        xlabel = "Mean time to shot when drawing vs avg opposition"
        ylabel = "Expected conversion of a shot taken when drawing vs avg opposition"
    else:
        xsamples = weibull_means(quantity_k, np.exp(prevention + np.mean(generation)))
        ysamples = ss.expit(obstruction + np.mean(conversion) + quality_k*xsamples)
        xlabel = "Mean time to conceding a shot when drawing vs avg opposition"
        ylabel = "Expected conversion of a shot conceded when drawing vs avg opposition"

    xmeans = np.apply_along_axis(np.mean, 0, xsamples)
    ymeans = np.apply_along_axis(np.mean, 0, ysamples)
    xerr = np.apply_along_axis(hdi, 0, xsamples)
    yerr = np.apply_along_axis(hdi, 0, ysamples)

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = plt.subplot(111)
    ax.errorbar(xmeans, ymeans, xerr=xerr, yerr=yerr, elinewidth=0.25, fmt='.')
    labels = [ax.text(xmeans[i], ymeans[i], labels[i], size=6, color='gray')
              for i in range(len(labels))]
    adjust_text(labels, arrowprops={'arrowstyle': '-', 'color': 'gray'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def wrangle(filepath):
    """ Prepare data for Stan. """
    shots = pd.read_csv(filepath, index_col=None)

    team_map = {
        name: idx + 1
        for idx, name in enumerate(set(shots.team) | set(shots.oppo))
    }

    stan_data = (
        shots
        .assign(
            team=lambda df: list(map(team_map.__getitem__, df.team)),
            oppo=lambda df: list(map(team_map.__getitem__, df.oppo))
        )
        [['team', 'oppo', 'wait', 'time', 'home', 'neutral', 'goal', 'state']]
        .to_dict(orient='list')
    )
    stan_data['n_teams'] = len(team_map)
    stan_data['n_shots'] = len(shots)

    return stan_data, team_map


def run():
    stan_data, team_map = wrangle("data/shots.csv")
    return fit("shotgen.stan", stan_data, chains=7, iter=2000), team_map
