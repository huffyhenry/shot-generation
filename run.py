import os
import pickle

import pandas as pd
import pystan


def fit(model_file, data, force_compile=False, **kwargs):
    """ Compile the model unless cached and run the sampler. """
    if not force_compile and os.path.exists("cache/%s.pickle" % model_file):
        model = pickle.load(open("cache/%s.pickle" % model_file, 'rb'))
    else:
        model = pystan.StanModel(model_file)
        pickle.dump(model, open("cache/%s.pickle" % model_file, 'wb'))

    return model.sampling(data, **kwargs)


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
        [['team', 'oppo', 'wait', 'time', 'home', 'goal', 'scored', 'conceded']]
        .to_dict(orient='list')
    )
    stan_data['n_teams'] = len(team_map)
    stan_data['n_shots'] = len(shots)

    return stan_data, team_map


def run(model_file):
    stan_data, team_map = wrangle("data/shots2018.csv")
    return fit(model_file, stan_data, chains=4, iter=500), team_map
