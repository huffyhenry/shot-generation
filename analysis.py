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


def team_scatter(samples, team_map, attack=True):
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
        # Nb. mean(prevention) = mean(obstruction) = 0
        xsamples = weibull_means(quantity_k, 1.0/np.exp(generation))
        ysamples = ss.expit(conversion + quality_k*xsamples)
        xlabel = "Expected time to next shot taken at 0:0 vs avg opposition"
        ylabel = "Expected conversion of a shot taken at 0:0 against avg opposition"
    else:
        xsamples = weibull_means(quantity_k, 1.0/np.exp(prevention + np.mean(generation)))
        ysamples = ss.expit(obstruction + np.mean(conversion) + quality_k*xsamples)
        xlabel = "Expected time to next shot conceded at 0:0 vs avg opposition"
        ylabel = "Expected conversion of a shot conceded at 0:0 to avg opposition"

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
