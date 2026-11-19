"""Reference values for tests/inference.rs, from pyhf (numpy backend,
iminuit optimizer at a tight tolerance). Run:  python pyhf_reference.py"""
import json

import pyhf

pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-7))


def model(data):
    spec = {
        "channels": [
            {
                "name": "ch",
                "samples": [
                    {"name": "sig", "data": [5.0, 10.0, 3.0],
                     "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
                    {"name": "bkg", "data": [50.0, 52.0, 40.0],
                     "modifiers": [{"name": "lumi", "type": "lumi", "data": None}]},
                ],
            }
        ],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": [
            {"name": "lumi", "auxdata": [1.0], "sigmas": [0.1], "bounds": [[0.5, 1.5]], "inits": [1.0]},
        ]}}],
        "observations": [{"name": "ch", "data": data}],
        "version": "1.0.0",
    }
    ws = pyhf.Workspace(spec)
    m = ws.model()
    return m, ws.data(m)


out = {}
m, d = model([55.0, 62.0, 44.0])
for mu in [0.5, 1.0, 2.0]:
    obs, exp = pyhf.infer.hypotest(mu, d, m, test_stat="qtilde", return_expected_set=True)
    _, (clsb, clb) = pyhf.infer.hypotest(mu, d, m, test_stat="qtilde", return_tail_probs=True)
    out[f"cls_{mu}"] = dict(cls=float(obs), clsb=float(clsb), clb=float(clb),
                            expected=[float(e) for e in exp])
obs, exp = pyhf.infer.intervals.upper_limits.upper_limit(d, m, level=0.05, rtol=1e-8)
out["limit"] = dict(observed=float(obs), expected=[float(e) for e in exp])
m, d = model([62.0, 75.0, 47.0])
p0, p0_exp = pyhf.infer.hypotest(0.0, d, m, test_stat="q0", return_expected=True)
out["discovery"] = dict(p0=float(p0), p0_expected=float(p0_exp))
print(json.dumps(out, indent=1))
