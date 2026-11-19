"""ROOT (through PyROOT; the fits themselves run in C++): TH1::Fit with a TF1
for the binned χ² and Poisson-likelihood ("L") fits, RooFit (an extended
RooAddPdf) for the unbinned ones."""

import sys

import ROOT

from common import BINNED_SEED, HI, LO, UNBINNED, UNBINNED_SEED, emit, read_events, read_hist, timed

ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
d = sys.argv[1]
minimizer = ROOT.Math.MinimizerOptions.DefaultMinimizerType()

x, y = read_hist(d)
h = ROOT.TH1D("peak", "", len(x), LO, HI)
for i, v in enumerate(y, start=1):
    h.SetBinContent(i, v)
    h.SetBinError(i, v**0.5)
f = ROOT.TF1("f", "[0]*exp(-0.5*((x-[1])/[2])^2) + exp([3]+[4]*x)", LO, HI)
seed = list(BINNED_SEED.values())

# ROOT's default χ² fit reports MIGRAD's running covariance estimate (no
# HESSE); option "E" adds HESSE and MINOS — both are shown. Its likelihood fit
# minimizes −ln L; Chi2() is the 2·Σ[f − y + y·ln(y/f)] the others report.
for problem, option, label in [
    ("binned_chi2", "Q0SN", ""),
    ("binned_chi2", "Q0SNE", " (option E)"),
    ("binned_likelihood", "LQ0SN", ""),
]:
    def fit():
        f.SetParameters(*seed)
        return h.Fit(f, option)

    r, ms, _ = timed(200, fit)
    n = f.GetNpar()
    emit(f"ROOT-{minimizer}{label}", problem, [r.Parameter(i) for i in range(n)],
         [r.ParError(i) for i in range(n)], r.Chi2(), r.NCalls(), r.IsValid(), ms, 200)

xv = ROOT.RooRealVar("x", "x", LO, HI)
for name, scale, repeats in UNBINNED:
    events = read_events(d, name)
    data = ROOT.RooDataSet.from_numpy({"x": events}, ROOT.RooArgSet(xv))
    s = dict(UNBINNED_SEED, Ns=UNBINNED_SEED["Ns"] * scale, Nb=UNBINNED_SEED["Nb"] * scale)
    mu = ROOT.RooRealVar("mu", "mu", s["mu"], -1e3, 1e3)
    sigma = ROOT.RooRealVar("sigma", "sigma", s["sigma"], 0.01, 10)
    # RooExponential is exp(c·x): c = −λ.
    c = ROOT.RooRealVar("c", "c", -s["lam"], -10, -0.01)
    ns = ROOT.RooRealVar("Ns", "Ns", s["Ns"], 0, 1e8)
    nb = ROOT.RooRealVar("Nb", "Nb", s["Nb"], 0, 1e8)
    gauss = ROOT.RooGaussian("g", "g", xv, mu, sigma)
    expo = ROOT.RooExponential("e", "e", xv, c)
    model = ROOT.RooAddPdf("m", "m", ROOT.RooArgList(gauss, expo), ROOT.RooArgList(ns, nb))
    params = [ns, mu, sigma, nb, c]
    start = [s["Ns"], s["mu"], s["sigma"], s["Nb"], -s["lam"]]

    def fit():
        for p, v in zip(params, start):
            p.setVal(v)
        return model.fitTo(data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True),
                           ROOT.RooFit.PrintLevel(-1))

    r, ms, first = timed(repeats + 1, fit)
    vals = [p.getVal() for p in params]
    errs = [p.getError() for p in params]
    vals[4] = -vals[4]  # report λ = −c
    # RooFitResult does not report the number of NLL calls: −1 ("—").
    emit(f"RooFit-{minimizer}", f"unbinned_{name}", vals, errs, 2 * r.minNll(),
         -1, r.status() == 0 and r.covQual() == 3, ms, repeats, first_ms=first)
