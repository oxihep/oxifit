"""Reference values for tests/templates.rs from pyhf (numpy backend, iminuit
optimizer at a tight tolerance). Run:  python pyhf_templates.py"""
import json

import numpy as np
import pyhf

pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-7))

A = {
    "channels": [{"name": "SR", "samples": [
        {"name": "signal", "data": [4.0, 9.0, 6.0],
         "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
        {"name": "background", "data": [60.0, 45.0, 30.0], "modifiers": [
            {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.08, "lo": 0.93}},
            {"name": "jes", "type": "histosys",
             "data": {"hi_data": [63.0, 47.0, 30.5], "lo_data": [57.5, 43.0, 29.0]}},
            {"name": "mc_stat", "type": "staterror", "data": [3.0, 2.5, 2.0]},
        ]},
    ]}],
    "observations": [{"name": "SR", "data": [66.0, 58.0, 35.0]}],
    "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0",
}

B = {
    "channels": [
        {"name": "SR", "samples": [
            {"name": "signal", "data": [3.0, 6.0],
             "modifiers": [{"name": "mu", "type": "normfactor", "data": None},
                           {"name": "lumi", "type": "lumi", "data": None}]},
            {"name": "background", "data": [20.0, 15.0], "modifiers": [
                {"name": "unc", "type": "shapesys", "data": [3.0, 2.0]},
                {"name": "xsec", "type": "normsys", "data": {"hi": 1.1, "lo": 0.85}},
                {"name": "lumi", "type": "lumi", "data": None}]},
        ]},
        {"name": "CR", "samples": [
            {"name": "background", "data": [100.0, 80.0], "modifiers": [
                {"name": "xsec", "type": "normsys", "data": {"hi": 1.1, "lo": 0.85}},
                {"name": "lumi", "type": "lumi", "data": None}]},
        ]},
    ],
    "observations": [{"name": "SR", "data": [27.0, 24.0]}, {"name": "CR", "data": [95.0, 88.0]}],
    "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": [
        {"name": "lumi", "auxdata": [1.0], "sigmas": [0.02], "bounds": [[0.5, 1.5]], "inits": [1.0]}]}}],
    "version": "1.0.0",
}

out = {}
for label, spec in [("A", A), ("B", B)]:
    ws = pyhf.Workspace(spec)
    model = ws.model()
    data = ws.data(model)
    best, unc = pyhf.infer.mle.fit(data, model, return_uncertainties=True).T
    names = model.config.par_names
    obs, exp = pyhf.infer.hypotest(1.0, data, model, test_stat="qtilde", return_expected_set=True)
    lim, lim_exp = pyhf.infer.intervals.upper_limits.upper_limit(data, model, level=0.05, rtol=1e-8)
    # pyhf's HESSE errors run ~1 % small here; the inverse of a direct
    # finite-difference Hessian of its own -2 ln L at the best fit is the
    # reference the test uses.
    x0 = np.array(best, dtype=float)
    f = lambda x: float(-2 * model.logpdf(x, data)[0])
    n = len(x0)
    h = np.array([1e-4 if "[" in nm else 1e-3 for nm in names])
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h[i] * h[j])
    exact = np.sqrt(np.diag(np.linalg.inv(H / 2)))
    out[label] = dict(
        names=list(names), best=[float(v) for v in best], errors=[float(v) for v in unc],
        errors_exact=[float(v) for v in exact],
        cls=float(obs), cls_expected=[float(v) for v in exp],
        limit=float(lim), limit_expected=[float(v) for v in lim_exp],
    )
print(json.dumps(out, indent=1))
