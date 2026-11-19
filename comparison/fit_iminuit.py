"""iminuit (C++ Minuit2 behind a Python API): the comparison's binned χ²,
binned Poisson likelihood and extended unbinned fits."""

import sys

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, LeastSquares

from common import BINNED_SEED, HI, UNBINNED, UNBINNED_SEED, emit, read_events, read_hist, timed

d = sys.argv[1]
x, y = read_hist(d)


def peak(x, A, mu, sigma, c0, c1):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + np.exp(c0 + c1 * x)


def fit(cost, seed, limits=()):
    m = Minuit(cost, **seed)
    for name in limits:
        m.limits[name] = (0.01, None)
    m.migrad()
    m.hesse()
    return m


def report(problem, m, ms, repeats):
    emit("iminuit", problem, m.values, m.errors, m.fval, m.nfcn, m.valid, ms, repeats)


chi2 = LeastSquares(x, y, np.sqrt(y), peak)
m, ms, _ = timed(200, lambda: fit(chi2, BINNED_SEED))
report("binned_chi2", m, ms, 200)


# iminuit's BinnedNLL integrates the model over each bin; ROOT's "L" fit (and
# the others here) evaluate it at the bin center — so a hand-written cost.
def poisson(A, mu, sigma, c0, c1):
    f = np.maximum(peak(x, A, mu, sigma, c0, c1), 1e-300)
    t = f - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / f), 0.0)
    return 2.0 * t.sum()


poisson.errordef = Minuit.LEAST_SQUARES
m, ms, _ = timed(200, lambda: fit(poisson, BINNED_SEED))
report("binned_likelihood", m, ms, 200)


def mixture(x, Ns, mu, sigma, Nb, lam):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    e = lam * np.exp(-lam * x) / (1 - np.exp(-lam * HI))
    return Ns + Nb, Ns * g + Nb * e


for name, scale, repeats in UNBINNED:
    events = read_events(d, name)
    nll = ExtendedUnbinnedNLL(events, mixture)
    seed = dict(UNBINNED_SEED, Ns=UNBINNED_SEED["Ns"] * scale, Nb=UNBINNED_SEED["Nb"] * scale)
    m, ms, _ = timed(repeats, lambda: fit(nll, seed, limits=("sigma", "lam")))
    report(f"unbinned_{name}", m, ms, repeats)
