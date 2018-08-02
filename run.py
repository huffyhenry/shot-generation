import os
import pickle

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
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


def graph(samples, team_map):
    """ Quick visualisation of the team coefficient samples. """
    plt.style.use('Solarize_Light2')

    labels = list(team_map.keys())  # Magically in the correct order

    # Extract samples and convert them to the time to shot interpretation
    generation = 1.0 / (np.mean(samples['prevention']) * samples['generation'])
    prevention = 1.0 / (np.mean(samples['generation']) * samples['prevention'])

    gen_means = np.apply_along_axis(np.mean, 0, generation)
    pre_means = np.apply_along_axis(np.mean, 0, prevention)
    gen_bounds = np.apply_along_axis(hdi, 0, generation)
    pre_bounds = np.apply_along_axis(hdi, 0, prevention)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.errorbar(gen_means, pre_means,
                xerr=gen_bounds, yerr=pre_bounds,
                elinewidth=0.25, fmt='.')
    labels = [ax.text(gen_means[i], pre_means[i], labels[i], size=6, color='gray')
              for i in range(len(labels))]
    adjust_text(labels, arrowprops={'arrowstyle': '-', 'color': 'gray'})
    ax.set_xlabel("Expected time to shot taken when drawing vs avg opposition")
    ax.set_ylabel("Expected time to shot conceded when drawing vs avg opposition")
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
        [['team', 'oppo', 'wait', 'time', 'home', 'neutral', 'shot', 'goal',
          'own_goal', 'penalty', 'state']]
        .to_dict(orient='list')
    )
    stan_data['n_teams'] = len(team_map)
    stan_data['n_shots'] = len(shots)

    return stan_data, team_map


def run():
    stan_data, team_map = wrangle("data/shots.csv")
    return fit("shotgen.stan", stan_data, chains=7, iter=4000), team_map
