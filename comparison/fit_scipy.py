"""scipy.optimize.curve_fit: the comparison's binned χ² fit (curve_fit is a
least-squares fitter; it has no likelihood or unbinned mode)."""

import sys

import numpy as np
from scipy.optimize import curve_fit

from common import BINNED_SEED, emit, read_hist, timed

x, y = read_hist(sys.argv[1])


def peak(x, A, mu, sigma, c0, c1):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + np.exp(c0 + c1 * x)


def fit():
    return curve_fit(peak, x, y, p0=list(BINNED_SEED.values()), sigma=np.sqrt(y),
                     absolute_sigma=True, full_output=True)


(popt, pcov, info, _, ier), ms, _ = timed(200, fit)
chi2 = float((((y - peak(x, *popt)) / np.sqrt(y)) ** 2).sum())
emit("scipy", "binned_chi2", popt, np.sqrt(np.diag(pcov)), chi2, info["nfev"], ier in (1, 2, 3, 4), ms, 200)
