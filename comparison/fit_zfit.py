"""zfit (TensorFlow): the comparison's extended unbinned fits — zfit's home
ground. The first fit includes TensorFlow's graph compilation (`first_ms`)."""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings

warnings.filterwarnings("ignore")
import zfit  # noqa: E402

from common import HI, LO, UNBINNED, UNBINNED_SEED, emit, read_events, timed  # noqa: E402

obs = zfit.Space("x", limits=(LO, HI))
for name, scale, repeats in UNBINNED:
    events = read_events(sys.argv[1], name)
    data = zfit.Data.from_numpy(obs=obs, array=events)
    seed = dict(UNBINNED_SEED, Ns=UNBINNED_SEED["Ns"] * scale, Nb=UNBINNED_SEED["Nb"] * scale)
    mu = zfit.Parameter(f"mu_{name}", seed["mu"])
    sigma = zfit.Parameter(f"sigma_{name}", seed["sigma"], 0.01, 10)
    # zfit's Exponential is exp(lam·x): lam = −λ.
    lam = zfit.Parameter(f"lam_{name}", -seed["lam"], -10, -0.01)
    ns = zfit.Parameter(f"Ns_{name}", seed["Ns"])
    nb = zfit.Parameter(f"Nb_{name}", seed["Nb"])
    sig = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=ns)
    bkg = zfit.pdf.Exponential(lam=lam, obs=obs, extended=nb)
    model = zfit.pdf.SumPDF([sig, bkg])
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    params = [ns, mu, sigma, nb, lam]

    def fit():
        for p, v in zip(params, [seed["Ns"], seed["mu"], seed["sigma"], seed["Nb"], -seed["lam"]]):
            p.set_value(v)
        result = minimizer.minimize(nll)
        result.hesse()
        return result

    result, ms, first = timed(repeats + 1, fit)
    values = [result.params[p]["value"] for p in params]
    errors = [result.params[p]["hesse"]["error"] for p in params]
    values[4], errors[4] = -values[4], errors[4]  # report λ = −lam
    emit("zfit", f"unbinned_{name}", values, errors, result.fmin, result.info.get("n_eval", 0),
         result.valid, ms, repeats, first_ms=first)
