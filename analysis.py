""" Code to help with analysis of fitted models. """
from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss

plt.style.use('Solarize_Light2')
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['font.family'] = "Merriweather"


def AIC(samples):
    """ Akaike Information Criterion. After Wikipedia. """
    LL = sum(np.apply_along_axis(np.mean, 0, samples['logLik']))
    k = np.mean(samples['n_params'])  # Should be a constant
    return -2.0*(LL - k)


def WAIC(samples):
    """
    Widely Applicable Information Criterion.
    See R. McElreath "Statistical Rethinking", Section 6.4, CRC 2016.
    """
    lppd = sum(np.log(np.apply_along_axis(np.mean, 0, np.exp(samples['logLik']))))
    pWAIC = sum(np.apply_along_axis(np.var, 0, samples['logLik']))
    return -2.0*(lppd - pWAIC)


def hdi(vector, width=0.5):
    """ Highest Density Interval. """
    x = sorted(vector)
    size = len(vector)
    left = int((1.0 - width)/2 * size)
    right = int((width + (1.0 - width)/2) * size)
    return x[left], x[right]


def team_profiles(gen_samples, conv_samples, teammap):
    """
    Visualise the fit of the generation and coversion models as a pair
    of scatter plots.
    """
    def weibull_means(shape, scales):
        """ Return the mean of the Weibull distro, vectorized over scale. """
        return ss.gamma(1.0 + 1.0/shape)*scales

    generation = gen_samples['generation']
    prevention = gen_samples['prevention']
    conversion = conv_samples['conversion']
    obstruction = conv_samples['obstruction']
    k = gen_samples['k']  # The Weibull exponent
    ls = list(teammap.keys())  # Team labels, same for both models

    plt.figure(figsize=(10, 6), tight_layout=True)

    # Attack
    xs = np.apply_along_axis(np.mean, 0, weibull_means(np.mean(k), 1.0/np.exp(generation)))
    ys = np.apply_along_axis(np.mean, 0, ss.expit(conversion))
    ax = plt.subplot(121)
    ax.scatter(xs, ys)
    ax.set_xlabel("Expected minutes to next shot taken at 0:0 vs avg opposition")
    ax.set_ylabel("Expected conversion of a shot taken at 0:0 vs avg opposition")
    gen_labels = [ax.text(xs[i], ys[i], ls[i]) for i in range(len(ls))]
    adjust_text(gen_labels, arrowprops={'arrowstyle': '-', 'color': 'gray'})

    # Shot conversion
    xs = weibull_means(np.mean(k), 1.0 / np.exp(prevention + np.mean(generation)))
    xs = np.apply_along_axis(np.mean, 0, xs)
    ys = np.apply_along_axis(np.mean, 0, ss.expit(obstruction + np.mean(conversion)))

    ax = plt.subplot(122)
    ax.scatter(xs, ys)
    ax.set_xlabel("Expected minutes to next shot conceded at 0:0 to avg opposition")
    ax.set_ylabel("Expected conversion of a shot taken at 0:0 by avg opposition")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    conv_labels = [ax.text(xs[i], ys[i], ls[i]) for i in range(len(ls))]
    adjust_text(conv_labels, arrowprops={'arrowstyle': '-', 'color': 'gray'})

    plt.show()


def score_bars(samples, figsize=(6, 3)):
    # Hardcode the score order for simplicity
    # The last estimate, 'other D', ommitted due to large CIs.
    scores = ['0:1', '0:2', '1:0', '1:1', '1:2', '2:0', '2:1', '2:2',
              'other W', 'other L']
    summaries = []

    plt.figure(tight_layout=True, figsize=figsize)
    ax = plt.subplot()

    for idx, score in enumerate(scores):
        draws = np.exp(samples['score_raw'][:, idx])
        left, right = hdi(draws, width=0.95)
        mean = np.mean(draws)
        summaries.append((score, mean, abs(left-mean), abs(right-mean)))

    summaries.sort(key=lambda x: x[1])
    scores, means, left, right = zip(*summaries)

    ax.axvline(1, linestyle=':', alpha=0.40)
    ax.errorbar(means, range(len(scores)), xerr=[left, right], fmt=".")
    ax.set_xlabel("Shooting rate multiplier relative to 0:0 (with 95% CIs)")
    ax.set_ylabel("Current score")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(scores)

    plt.show()
