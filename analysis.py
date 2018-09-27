""" Code to help with analysis of fitted models. """
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss


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


def gen_scatter(samples, team_map, cis=True):
    """ Quick visualisation of the team coefficient samples. """
    def weibull_means(shape, scales):
        """ Return the mean of the Weibull distro, vectorized over scale. """
        return ss.gamma(1.0 + 1.0/shape)*scales

    plt.style.use('Solarize_Light2')

    # Extract the samples
    generation, prevention = samples['generation'], samples['prevention']
    k = np.mean(samples['k'])

    labels = list(team_map.keys())  # Magically in the correct order

    # Select quantities to plot and make them interpretable
    # Nb. mean(prevention) = 0
    xsamples = weibull_means(k, 1.0/np.exp(generation))
    ysamples = weibull_means(k, 1.0/np.exp(prevention + np.mean(generation)))
    xlabel = "Expected time to next shot taken at 0:0 vs avg opposition"
    ylabel = "Expected time to next shot conceded at 0:0 to avg opposition"

    xmeans = np.apply_along_axis(np.mean, 0, xsamples)
    ymeans = np.apply_along_axis(np.mean, 0, ysamples)

    # Plot
    plt.figure(figsize=(8, 8), tight_layout=True)
    ax = plt.subplot(111)
    if cis:
        xerr = np.apply_along_axis(hdi, 0, xsamples)
        yerr = np.apply_along_axis(hdi, 0, ysamples)
        ax.errorbar(xmeans, ymeans, xerr=xerr, yerr=yerr, elinewidth=0.25, fmt='.')
    else:
        ax.scatter(xmeans, ymeans)

    labels = [ax.text(xmeans[i], ymeans[i], labels[i], size=6, color='gray')
              for i in range(len(labels))]
    adjust_text(labels, arrowprops={'arrowstyle': '-', 'color': 'gray'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def score_matrix(samples, max_goals=2):
    m = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
                m[i, j] = np.mean(samples['score'][:, i, j])

    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(m.T, origin='upper')
    plt.show()
