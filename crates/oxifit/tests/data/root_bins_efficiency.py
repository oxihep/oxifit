"""Reference values for tests/bins_efficiency.rs from ROOT 6.40 (PyROOT):
bin-integrated fits (TH1::Fit option "I") of the peak histogram rebinned to
20 bins, and a TEfficiency::Fit (binomial likelihood) of a turn-on.
Run:  python3 root_bins_efficiency.py"""
import json
import math
import os

import ROOT

here = os.path.dirname(os.path.abspath(__file__))
rows = [l.split() for l in open(os.path.join(here, "peak_hist.txt"))]
counts = [float(r[1]) for r in rows]
coarse = [sum(counts[5 * i:5 * i + 5]) for i in range(20)]
h = ROOT.TH1D("h", "", 20, 0.0, 10.0)
for i, c in enumerate(coarse):
    h.SetBinContent(i + 1, c)
    h.SetBinError(i + 1, math.sqrt(c))

out = {"coarse": coarse}
for label, opt in [("chi2", "Q0SNE"), ("chi2_int", "IQ0SNE"), ("lik", "LQ0SN"), ("lik_int", "LIQ0SN")]:
    f = ROOT.TF1("f" + label, "[0]*exp(-0.5*((x-[1])/[2])^2) + exp([3]+[4]*x)", 0, 10)
    f.SetParameters(1500.0, 5.0, 0.8, 8.1, -0.35)
    r = h.Fit(f, opt)
    out[label] = dict(params=[r.Parameter(i) for i in range(5)], errors=[r.ParError(i) for i in range(5)])

# A trigger turn-on: 30 bins of pT, a plateau below 1.
x = [10.0 + 1.0 * i for i in range(30)]
total = [30 + (7 * i) % 23 for i in range(30)]
true = lambda v: 0.95 * 0.5 * (1.0 + math.erf((v - 22.0) / (math.sqrt(2.0) * 2.5)))
# Deterministic "observations" near the truth.
passed = [min(t, max(0, round(t * true(v) + 2.0 * math.sin(3.1 * i)))) for i, (v, t) in enumerate(zip(x, total))]
hp = ROOT.TH1D("hp", "", 30, 9.5, 39.5)
ht = ROOT.TH1D("ht", "", 30, 9.5, 39.5)
for i in range(30):
    hp.SetBinContent(i + 1, passed[i])
    ht.SetBinContent(i + 1, total[i])
eff = ROOT.TEfficiency(hp, ht)
f = ROOT.TF1("turnon", "[2]*0.5*(1+TMath::Erf((x-[0])/(sqrt(2)*[1])))", 9.5, 39.5)
f.SetParameters(20.0, 2.0, 0.9)
f.SetParLimits(2, 0.0, 1.0)
r = eff.Fit(f, "SQ0")
out["efficiency"] = dict(
    x=x, passed=passed, total=total,
    params=[r.Parameter(i) for i in range(3)], errors=[r.ParError(i) for i in range(3)],
)
print(json.dumps(out, indent=1))
