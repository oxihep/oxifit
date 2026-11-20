# oxifit compared with ROOT, iminuit, scipy and zfit

The same data, models, starting values and costs, fitted with each tool, then
compared on whether the results agree, how fast the fits are, and how they read
as code. Everything here is reproducible with `./run_all.sh`.

Measured on an Apple M4 Pro (12 cores), macOS: oxifit 0.7.0 (Rust 1.100,
release build), ROOT 6.40.04 (PyROOT; Minuit2), iminuit 2.33.0, scipy 1.18.1,
zfit 0.28.0, numpy 2.5 (Python 3.13). [`results/RESULTS.md`](results/RESULTS.md)
has the full tables; the raw outputs are in `results/*.jsonl`.

## The problems

A Gaussian peak on a falling exponential background (`make_data.py`, fixed
seed), every fit starting from the same values:

| Problem | Data | Model | Cost |
|---|---|---|---|
| Binned χ² | 100-bin histogram, 25 000 entries | `A·exp(−½((x−μ)/σ)²) + exp(c₀ + c₁x)` at bin centers | Neyman χ² |
| Binned likelihood | the same | the same | `2Σ[f − n + n·ln(n/f)]` (ROOT's `"L"`) |
| Unbinned, 10⁵ / 10⁶ events | events on [0, 10) | `Nₛ·N(x; μ, σ) + N_b·λe^{−λx}/(1 − e^{−10λ})` | extended −2 ln L |

Every fit includes its error computation (HESSE or equivalent). Times are the
median of 200 (binned), 10 (10⁵) or 3 (10⁶) fits from the seed.

## Do they agree?

Yes. Every valid fit matches iminuit's parameters to within **0.05σ**
(`report.py` fails otherwise). oxifit's binned fits match to `~10⁻⁷σ` and its
unbinned ones to `~10⁻²σ`, MIGRAD's own precision. The binned agreement with
ROOT and iminuit is now an oxifit test
(`crates/oxifit/tests/reference_fitters.rs`), run on every CI build.

Two differences in the **errors** are worth knowing:

- **ROOT's default χ² fit reports MIGRAD's covariance estimate, not HESSE's**:
  its errors are up to 3 % small (`A`: 7.02 instead of 7.24). Option `"E"`
  (HESSE + MINOS) gives the right ones. ROOT's likelihood fit and every other
  tool here run HESSE by default; so does oxifit.
- **On 10⁶ events, iminuit's errors (finite-difference HESSE at MIGRAD's
  default tolerance) are 0.5–0.7 % small.** The tools that differentiate
  exactly — oxifit with an analytic gradient, zfit (autodiff), RooFit — agree
  with each other, and with a profile-likelihood scan: refitting with `Nₛ`
  fixed at ±620 from its best value raises −2 ln L by 0.992 and 0.989, so
  σ(Nₛ) = 623.05; oxifit's gradient path reports 623.05, zfit 623.03, RooFit
  622.84, iminuit 618.83. oxifit without a gradient reports 620.0 at the
  default tolerance and 622.94 with `.tolerance(1e-4)`.

scipy's `curve_fit` errors come from a Gauss–Newton approximation of the
Hessian and differ by up to 3.7 % on this fit.

## How fast?

Median time per fit, including the error computation. "1 core" unless marked.

| Problem | oxifit | oxifit `rayon` | ROOT / RooFit | iminuit | scipy | zfit |
|---|---|---|---|---|---|---|
| Binned χ² | 0.20 ms | 0.19 ms | 0.19 ms (0.88 ms with `"E"`) | 1.00 ms | **0.15 ms** | — |
| Binned likelihood | **0.27 ms** | 0.27 ms | 0.53 ms | 1.25 ms | — | — |
| Unbinned, 10⁵ events | 161 ms | **38 ms** (12 cores) | 102 ms | 575 ms | — | 188 ms (≈4 cores) |
| Unbinned, 10⁶ events | 1116 ms | **174 ms** (12 cores) | 1032 ms | 5778 ms | — | 360 ms (≈4 cores) |

oxifit's numbers are its best variant per problem; for the unbinned fits that
is the model built from its shapes (`Model::gaussian + Model::exponential`),
which carry analytic gradients **and analytic integrals**, so the fit needs no
numerical normalization. `RESULTS.md` has every variant: a closure model takes
2.5 s on 10⁶ events (2.1 s with a hand-written gradient), because it is
normalized by quadrature at every step.

What the numbers say:

- **Small binned fits**: oxifit, ROOT and scipy cost about the same, a few
  hundred microseconds, set by the minimizer; iminuit pays ~5 µs of Python per
  cost call. scipy is fastest on the χ² fit but approximates the errors.
- **Large unbinned fits on one core**: with built-in shapes **oxifit and
  RooFit are level** (1.12 s against 1.03 s on 10⁶ events) — both normalize
  analytically. A closure model is normalized numerically and takes about
  twice as long; give it its integral with `Model::with_integral` (and its
  gradient with `with_gradient`) to close the gap.
- **With threads**: oxifit's `rayon` feature is the fastest here — 5.9× faster
  than RooFit on 10⁶ events, 2× faster than zfit (which uses about 4 cores
  through TensorFlow) — and its results do not depend on the thread count.
- **Start-up**, not in the table: `import zfit` takes 6–7 s (TensorFlow), and
  its first fit adds graph building (0.4–0.6 s); `import ROOT` takes 0.6 s warm,
  1.9 s cold. A Rust binary starts at once.

## How do they read?

The binned likelihood fit, in each tool:

**oxifit**

```rust
let model = (Model::gaussian("sig") + Model::exponential("bkg"))
    .with_params(vec![300.0, 5.0, 0.8, 6.5, -0.35]);
let fit = hist.fit_opts(&model, &FitOptions::new().method(FitMethod::Likelihood));
println!("μ = {:.4} ± {:.4}", fit.params[1], fit.errors[1]);
```

**ROOT**

```python
f = ROOT.TF1("f", "[0]*exp(-0.5*((x-[1])/[2])^2) + exp([3]+[4]*x)", 0, 10)
f.SetParameters(300, 5.0, 0.8, 6.5, -0.35)
r = h.Fit(f, "LQ0SN")          # "L" likelihood, "S" result, "Q" quiet, "0" no draw, "N" don't store
print(r.Parameter(1), r.ParError(1))
```

**iminuit**

```python
def nll(A, mu, sigma, c0, c1):          # bin-center Poisson cost, by hand
    f = np.maximum(peak(x, A, mu, sigma, c0, c1), 1e-300)
    return 2 * (f - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / f), 0)).sum()
nll.errordef = Minuit.LEAST_SQUARES
m = Minuit(nll, A=300, mu=5.0, sigma=0.8, c0=6.5, c1=-0.35)
m.migrad(); m.hesse()
print(m.values["mu"], m.errors["mu"])
```

**zfit** (the extended unbinned fit — its home ground)

```python
obs = zfit.Space("x", limits=(0, 10))
mu, sigma = zfit.Parameter("mu", 5.0), zfit.Parameter("sigma", 0.8, 0.01, 10)
lam = zfit.Parameter("lam", -0.3, -10, -0.01)
ns, nb = zfit.Parameter("ns", 8e3), zfit.Parameter("nb", 8e4)
model = zfit.pdf.SumPDF([zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=ns),
                         zfit.pdf.Exponential(lam=lam, obs=obs, extended=nb)])
nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=zfit.Data.from_numpy(obs=obs, array=x))
result = zfit.minimize.Minuit().minimize(nll); result.hesse()
```

The scripts in this folder are the complete versions.

| | oxifit | ROOT / RooFit | iminuit | scipy | zfit |
|---|---|---|---|---|---|
| Language | Rust | C++ (Python bindings) | Python (C++ Minuit2) | Python | Python (TensorFlow) |
| Install | one `Cargo.toml` line, no system deps | ROOT installation | `pip` | `pip` | `pip` (pulls TensorFlow: 1.1 GB) |
| Errors by default | HESSE | χ²: MIGRAD estimate; likelihood: HESSE | HESSE (on `hesse()`) | Gauss–Newton | HESSE |
| Binned χ² / likelihood | yes / yes (bin centers) | yes / yes (centers, or integrals with `"I"`) | yes / yes (`BinnedNLL` integrates over bins) | χ² only | yes |
| Unbinned, extended | yes (numerical normalization) | RooFit | yes (you normalize) | — | yes |
| Gradients | analytic if you supply them | RooFit: analytic for some pdfs, AD in recent versions | yours, if supplied | — | automatic (TensorFlow) |
| Simultaneous, constraints, toys | yes | RooFit | yes (add costs; `NormalConstraint`); toys by hand | — | yes |
| Profiles, contours, MINOS | yes | yes | yes | — | yes |
| Global minimizers | multistart, DE, CMA-ES | GSL/Genetic plugins | `scan`, `simplex` | `differential_evolution`, … | NLopt global |
| 2-D/3-D | yes | yes | yes (your cost) | via flattening | yes |
| Parallel cost | `rayon` feature | RooFit `NumCPU`, batch mode | numba / your code | — | TensorFlow threads |

Where each one shines: **ROOT/RooFit** for its huge model library, analytic
normalizations and workspace persistence; **iminuit** for Minuit from Python
with very little ceremony; **scipy** for a quick least-squares fit;
**zfit** for automatic differentiation and model building on top of
TensorFlow. **oxifit** is a fast, dependency-free fitter for Rust code —
the same algorithm and the same answers as ROOT and iminuit — with threads
that do not change the result; it does not differentiate models for you.

## Running it

```sh
uv venv -p 3.13 .venv && VIRTUAL_ENV=.venv uv pip install numpy scipy iminuit zfit
PYTHON=.venv/bin/python ROOT_PYTHON=python3 ./run_all.sh
```

`ROOT_PYTHON` must be a Python that can `import ROOT`. The data (`data/`, about
20 MB) is regenerated each run and not committed.
