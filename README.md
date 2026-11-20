# oxifit

[![CI](https://github.com/oxihep/oxifit/actions/workflows/ci.yml/badge.svg)](https://github.com/oxihep/oxifit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.100+](https://img.shields.io/badge/rust-1.100%2B-orange.svg)](https://www.rust-lang.org)
[![API docs](https://img.shields.io/badge/docs-API-blue.svg)](https://oxihep.github.io/oxifit/oxifit/)

Parametric fitting in pure Rust for **data of one to three variables**:
histograms, graphs, raw points or individual events. The minimizer is the pure-Rust
[Minuit2](https://crates.io/crates/minuit2) port, the algorithm ROOT uses.

| Crate | What it is |
| --- | --- |
| [`oxifit`](crates/oxifit) | The fitter: `FitData`/`FitExt`, `Model`, χ² / Poisson-likelihood / unbinned and extended likelihood costs, robust losses, Minuit2 (MIGRAD + HESSE, MINOS), Levenberg–Marquardt, global searches (multistart, differential evolution, CMA-ES) and an optional Nelder–Mead `argmin` backend |
| [`oxifit-formula`](crates/oxifit-formula) | A ROOT `TFormula` expression engine (its only dependency is oxistats, for `erf`/`erfc`/`lgamma`): parse and evaluate formula strings, numerical integral and derivative. `Model::from_formula` and oxiroot's `Func1D`/`Func2D`/`Func3D` are built on it. |

```toml
[dependencies]
oxifit = { git = "https://github.com/oxihep/oxifit", tag = "v0.7.0" }
```

```rust
use oxifit::{FitExt, Model, Points};

let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0], &[0.1; 4]).unwrap();
let line = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
let fit = data.fit(&line); // χ² by default
assert!(fit.valid);
println!("slope = {:.3} ± {:.3}", fit.params[1], fit.errors[1]);
```

Models come from a closure, a built-in shape (Gaussian, exponential,
polynomial, Crystal Ball, Breit–Wigner, Voigt, Novosibirsk, ARGUS, …) or any
formula string (`Model::from_formula("f", "[0]*exp(-0.5*((x-[1])/[2])^2)")`).
Add models to fit a signal on a background (`sig + bkg`, or `Model::sum`;
parameters become `sig.mean`, `bkg.slope`, …), and constrain a parameter
with a Gaussian — an external measurement or a nuisance prior — with
`.constrain("bkg.slope", -0.02, 0.005)`. `Model::convolve("m", &signal,
&resolution, window)` smears a model by a resolution function (a unit-area
`Model::normal`, say): a lifetime by vertex resolution, a resonance by
detector resolution.

## Minimizers

`FitOptions::minimizer` picks the optimizer. Every Minuit-based choice ends with
MIGRAD and HESSE, so values, errors, covariance and `valid` mean the same
thing whichever one found the minimum.

| `Minimizer` | What it does | When |
| --- | --- | --- |
| `Minuit2Combined` (default) | MIGRAD, falling back to SIMPLEX (then MIGRAD again) when it fails — ROOT's `"Minimize"` | Every fit: same cost as MIGRAD when MIGRAD succeeds |
| `Minuit2` | MIGRAD alone, then HESSE (ROOT's `"Migrad"`) | Matching ROOT's default exactly |
| `Simplex` | Minuit2's SIMPLEX, then HESSE | A rough, noisy cost |
| `MultiStart(MultiStart::new(n))` | `n` local fits from points spread over the parameter limits; the lowest wins | A few competing minima |
| `DifferentialEvolution(…)` | A population-based global search over the limits, then MIGRAD | The seed may sit in the wrong basin (a peak of unknown position, a multimodal model) |
| `CmaEs(CmaEs::new())` | CMA-ES: an evolution strategy that learns the valley's shape, then MIGRAD | As above, usually in a third of the evaluations; copes with correlated parameters |
| `LevenbergMarquardt` | Damped Gauss–Newton on the residuals, then MIGRAD | A good seed: fewer evaluations than MIGRAD on least-squares and likelihood fits |
| `LBfgsB` | Limited-memory quasi-Newton within the limits, then MIGRAD | Many parameters, with an analytic gradient |
| `NelderMead` (`argmin` feature) | An independent simplex with a numerical Hessian | Cross-checking Minuit2 |

The global searches look inside each free parameter's limits, so set a
`limit` on the parameters to be searched. They are seeded and reproducible.
`FitOptions::strategy`, `tolerance` and `max_fcn` tune Minuit2, and
`with_hesse(false)` skips HESSE when only the values matter.

## 2-D and 3-D fits

`Model::new2(|x, y, p| …)` and `Model::new3(|x, y, z, p| …)` take two or three
variables; `Model::product("m", &[&fx, &fy])` builds one from 1-D models (its
integral factorizes, so unbinned fits normalize it with 1-D integrals).
`Points2`/`Points3` hold binned or graph data (a 2-D/3-D histogram's bins),
`Events2`/`Events3` unbinned events; `FitOptions::range_y`/`range_z` restrict
and bound the fit. Every fit method, scan and minimizer works on them, and a
`Simultaneous` fit can mix dimensions.

## Analytic gradients

`Model::with_gradient(|x, p, g| …)` gives a model its derivatives `∂f/∂p`
(the Gaussian, exponential and polynomial shapes have theirs built in; a sum
has one when every part does; `check_gradient` compares with finite
differences). Every cost then has an exact gradient: MIGRAD,
Levenberg–Marquardt and L-BFGS-B use it, and the error matrix comes from it in
`2n` gradient calls instead of HESSE's `~2n²` cost calls — a 30-parameter fit
of ten peaks runs in 149 ms instead of 186 ms (126 ms with L-BFGS-B).
`FitOptions::with_gradient(false)` turns it off.

## Unbinned fits

`Events::new(&values)` (or `Events::weighted(&values, &weights)`) holds
individual events. `FitMethod::UnbinnedLikelihood` fits a model's *shape* to
them, normalized numerically over the fit range; `ExtendedUnbinnedLikelihood`
treats the model as the expected event density, so its integral (the yield) is
fitted too. For weighted events, `FitOptions::with_sumw2(true)` corrects the
covariance for the weights (RooFit's `SumW2Error`).

## Simultaneous fits

`Simultaneous::new().channel("A", &data_a, &model_a, &opts_a).channel("B", …)`
fits several datasets together, each with its own model and cost (binned or
unbinned); parameters with the same name are shared — a signal strength across
categories, a mass seen in two decay modes. `.fit(&opts)` returns one
`FitResult` over `param_names()`, and `channel_costs` splits the cost by
channel (RooFit's `RooSimultaneous`).

## Your own cost

Any `−2 ln L` (or χ²) can be minimized with every minimizer above, HESSE,
MINOS, profiles and contours — iminuit's `Minuit(fcn)`. A cost is anything
that implements `Cost` (parameter names and a value, optionally a gradient);
`FnCost` makes one from a closure, `ModelCost` turns a data fit into one, and
`CostSum` adds costs, merging parameters by name. `Fitter` holds the seeds and
the per-parameter settings:

```rust
use oxifit::{CostSum, FitOptions, FnCost, Fitter, ModelCost};

let data_fit = ModelCost::new(&hist, &model, &FitOptions::new()); // as hist.fit_opts
let external = FnCost::new(&["sigma"], |p| ((p[0] - 0.62) / 0.03).powi(2));
let total = CostSum::new().with(data_fit).with(external);
let fit = Fitter::new(&total, seeds)
    .limit("sigma", 0.1, 2.0)
    .fit(&FitOptions::new().with_minos(true));
```

`ModelCost::fitter()` starts from the model's own seeds and settings, so
`ModelCost::new(&d, &m, &o).fitter().fit(&o)` is exactly `d.fit_opts(&m, &o)`.

## Significance and limits

`HypoTest::new(fitter, "mu")` tests a parameter of interest — a signal
strength — on any cost, with the profile likelihood ratio, the pyhf / RooStats
recipe:

- `discovery(1.0)`: the observed `q₀`, p-value and significance, and the
  median significance expected for a signal of strength 1;
- `cls(mu)`: CLs+b, CLb and CLs at `mu` (`q̃_μ` by default, or `q_μ`), with
  the expected −2σ…+2σ band;
- `upper_limit(0.95, (lo, hi))`: the observed 95 % CL upper limit and the
  expected band (the "Brazil band");
- `cls_toys` / `discovery_toys`: the same from pseudo-experiments, for small
  counts, in parallel with the `rayon` feature.

The asymptotic results (Cowan, Cranmer, Gross, Vitells) use the cost's Asimov
dataset and the toys its pseudo-datasets, which `ModelCost` makes for binned and
unbinned data (and `CostSum` for sums of them); the `Fitter`'s Gaussian
constraints are the auxiliary measurements. `upcrossings` and
`global_significance` give the look-elsewhere correction of a mass scan
(Gross–Vitells). CLs, the limits and the discovery p-value match pyhf
(`tests/inference.rs`).

```rust
let cost = ModelCost::new(&hist, &signal_plus_background, &FitOptions::new().method(FitMethod::Likelihood));
let test = HypoTest::new(cost.fitter().limit("mu", 0.0, 10.0), "mu")?;
let limit = test.upper_limit(0.95, (0.0, 5.0))?;
println!("μ < {:.2} (expected {:.2})", limit.observed, limit.expected[2]);
```

## Template fits with systematics

`TemplateModel` is the HistFactory model of ROOT and pyhf: channels of
observed counts, each with samples (simulated templates) scaled and deformed by
modifiers — `norm_factor` (a free normalization), `norm_sys` and `histo_sys`
(normalization and shape uncertainties, with pyhf's interpolations),
`stat_error` (the templates' finite statistics, Barlow–Beeston-lite),
`shape_sys`, `lumi` and `shape_factor`. It is a `Cost`, so `model.fitter()`
fits it and `HypoTest` sets limits on it; `pulls`, `impacts` (the ranking on
the parameter of interest) and `yields` (pre- and post-fit, with errors from
the covariance) summarize the fit. With the `pyhf` feature,
`Workspace::from_json` reads a pyhf JSON workspace. On two workspaces the
best fit, CLs and limits match pyhf and the errors match an exact Hessian of
pyhf's own likelihood (`tests/templates.rs`).

```rust
let model = TemplateModel::new(vec![Channel::new("SR", &observed)
    .sample(Sample::new("signal", &sig).norm_factor("mu"))
    .sample(Sample::new("background", &bkg).norm_sys("xsec", 0.9, 1.1).stat_error("mc", &bkg_err))])?;
let fit = model.fitter().fit(&FitOptions::new());
let limit = HypoTest::new(model.fitter(), "mu")?.upper_limit(0.95, (0.0, 10.0))?;
```

## sWeights and weighted fits

`SWeights::compute(&events, &[&signal, &background], &yields, &opts)` turns an
extended fit in a discriminating variable (a mass) into per-event weights for
each species (sPlot, Pivk and Le Diberder); a weighted unbinned fit of another
variable with the signal weights measures the signal alone. Its covariance
needs care: `FitOptions::with_asymptotic_errors(true)` gives the
asymptotically correct one (Langenbruch), right for weights of any sign and
size, where SumW2 and the naive errors are not — on the `splot_lifetime`
example's toys the pull widths are 1.00 (asymptotic), 1.37 (naive) and 1.46
(SumW2).

## Bin-integrated fits and efficiencies

`FitOptions::with_bin_integration(true)` compares each bin with the model
averaged over the bin rather than its value at the bin center (ROOT's option
`"I"`, iminuit's `BinnedNLL`) — what matters when the model curves within a
bin. The bins come from the data's `bin_edges` (`Points::with_bin_edges` for
variable widths), else halfway between neighbouring centers. On the
comparison's peak rebinned to 0.5-wide bins it moves the fitted width from
0.613 to 0.596 (generated: 0.6), exactly as ROOT does.

`Efficiency::new(&x, &passed, &total)` (or `Efficiency::events(&x, &accepted)`
for single events) fits an efficiency curve with the **binomial likelihood**
(ROOT's `TEfficiency::Fit`, RooFit's `RooEfficiency`), right where a χ² with
binomial errors fails: efficiencies near 0 or 1 and bins with few trials.
`BinomialCost` is the same as a `Cost`, with Asimov data and toys. Both match
ROOT 6.40 (`tests/bins_efficiency.rs`).

## Bayesian sampling

`Sampler::new(fitter).run(burn_in, steps)` draws from a `Fitter`'s posterior
— its cost as `−2 ln L`, limits as flat priors, Gaussian constraints as
Gaussian priors — with the affine-invariant ensemble sampler of emcee
(Goodman–Weare stretch moves: no tuning, robust to correlations; walkers in
parallel with `rayon`, reproducible from a seed). The `Chain` gives means,
quantiles, central credible intervals, the covariance and each parameter's
autocorrelation time, and `r_hat` compares the walkers. Checked against
exact posteriors (a correlated Gaussian, a banana, a Poisson signal) in
`tests/mcmc.rs`.

`Nuts::new(fitter).run(warmup, draws)` samples the same posterior with
Hamiltonian Monte Carlo — the No-U-Turn Sampler of Stan and PyMC, with
Stan's warm-up (step size by dual averaging, a diagonal metric learned from
the warm-up). It follows the posterior's gradient (the cost's analytic one,
or central differences), so its draws are nearly independent: on the
`bayesian_fit` peak an autocorrelation time of about 1.3 steps against the
ensemble's 50. Limits are sampled in log/logit coordinates; chains run in
parallel with `rayon`; `Chain::divergences` flags trajectories that hit
too-curved regions. Checked in `tests/nuts.rs`, up to 20 dimensions.

```rust
let chain = Nuts::new(cost.fitter()).seed(1).run(1000, 2000)?;
let (lo, hi) = chain.interval("mu", 0.6827).unwrap();
assert!(chain.r_hat("mu").unwrap() < 1.01 && chain.divergences() == 0);
```

## Toy studies

`Toys::binned(&truth, &xs)` (Poisson counts) or `Toys::unbinned(&truth, range,
n)` (events; `.extended()` for a Poisson count) generates pseudo-data from a
model at known parameters; `.run(&model, &opts, ntoys)` fits each sample. The
`ToyStudy` gives every fit, the pulls `(fitted − true)/error` and
`pull_summary` — mean 0 and width 1 for an unbiased fit with correct errors
(RooFit's `RooMCStudy`).

## Profiles and contours

Beyond the parabolic errors (and MINOS), `data.profile(&model, &opts, "mean",
(lo, hi), n)` scans the cost in one parameter, refitting the others at each
value; `Profile::interval(1.0)` reads off the 1σ interval. `data.contour(&model,
&opts, "mean", "sigma", 2.30, 40)` traces the 68 % joint region of two
parameters (`Δ = 5.99` for 95 %), again refitting the rest. Both return a
`FitError` for an unknown or fixed parameter or a failed fit.

The [fitting guide](https://oxihep.github.io/oxiroot/guide/fitting/) covers
costs, limits, robust losses and fitting ROOT histograms (it uses the
`oxiroot::fit` re-export, which is this crate).

## Features

| Feature | Default | Adds |
| --- | --- | --- |
| `argmin` | no | A gradient-free Nelder–Mead backend via the `argmin` crate, chosen with `FitOptions::minimizer(Minimizer::NelderMead)` |
| `rayon` | no | Large costs on all cores (an unbinned fit of 10⁶ events: 604 → 107 ms on 12 cores), and toys, profile points and contour rays fitted in parallel. Results do not depend on the number of threads. |

## Examples

Physics fits on toy data with known truth, each printing the fitted values
next to it (`cargo run -p oxifit --example <name>`):

| Example | Shows |
| --- | --- |
| [`resonance_fit`](crates/oxifit/examples/resonance_fit.rs) | A Z → μμ peak on a falling background: a sideband fit seeds ROOT's `gaus(0)+expo(3)`, fitted with χ² and with a Poisson likelihood, with MINOS errors |
| [`trigger_turn_on`](crates/oxifit/examples/trigger_turn_on.rs) | A trigger efficiency turn-on fitted with an error-function model (a custom `Model` closure), and the pT where it is fully efficient |
| [`lifetime_fit`](crates/oxifit/examples/lifetime_fit.rs) | The B⁰ lifetime from its decay times — binned χ², binned likelihood and unbinned likelihood — and why a likelihood beats a χ² on sparse tails |
| [`peak_search`](crates/oxifit/examples/peak_search.rs) | A narrow peak at an unknown mass: MIGRAD from a blind seed misses it, multistart, differential evolution and CMA-ES find it |
| [`template_fit`](crates/oxifit/examples/template_fit.rs) | A HistFactory-style search: signal, top and W+jets templates in a signal and a control region, with normalization, jet-energy-scale, MC-statistics and luminosity uncertainties — pre/post-fit yields, MINOS, pulls, the impact ranking, significance and the CLs limit with its band, checked with toys |
| [`splot_lifetime`](crates/oxifit/examples/splot_lifetime.rs) | A B⁰ lifetime from sWeights: an extended mass fit, the weights, a weighted decay-time fit with naive, SumW2 and asymptotic errors, and 300 toys showing only the asymptotic ones cover |
| [`tag_and_probe`](crates/oxifit/examples/tag_and_probe.rs) | A trigger efficiency by tag and probe: simultaneous pass/fail Z fits per probe-pT bin (ε a shared parameter, MINOS errors), a bin-averaged turn-on fit, and toys of binomial vs χ² efficiency fits |
| [`bayesian_fit`](crates/oxifit/examples/bayesian_fit.rs) | Bayesian and frequentist side by side: a peak's posterior intervals from the ensemble sampler and from NUTS against MINOS, and a small signal's 95 % credible upper limit against CLs (asymptotic and toys) |
| [`resonance_search`](crates/oxifit/examples/resonance_search.rs) | A complete bump hunt: background-function choice (likelihood-ratio test) and its spurious signal, local significance vs mass, the look-elsewhere correction from toy scans, the signal fit, and CLs limits with the expected band, cross-checked with toys (`--release`: a few seconds) |

## Compared with ROOT, iminuit, scipy and zfit

[`comparison/`](comparison) fits the same data with each tool — a binned peak
(χ² and likelihood) and extended unbinned fits of 10⁵ and 10⁶ events — and
checks that they agree (every parameter within 0.05σ of iminuit; the binned
fits within `10⁻⁷σ`, now a test against ROOT and iminuit). On one core,
with a model built from its shapes (which carry analytic gradients and
integrals), oxifit fits binned data 2–7× faster than ROOT and scipy and
14–16× faster than iminuit, and large unbinned data 1.8× faster than RooFit
(0.60 s against 1.06 s on 10⁶ events); with the `rayon` feature it is the
fastest of all (10⁶ events: 0.11 s, against RooFit's 1.06 s and zfit's
0.37 s on four cores). The comparison also covers the code each tool needs
and what each does best.

## The oxihep family

Rust crates for high-energy physics, split the way Python's
[Scikit-HEP](https://scikit-hep.org) is. Each works on its own; together they
cover reading data, histogramming, fitting and plotting.

| Crate | What it does | Python counterpart |
| --- | --- | --- |
| [oxiroot](https://github.com/oxihep/oxiroot) | Read and write ROOT files: `TTree`, RNTuple, ROOT histograms and graphs | uproot |
| [oxihist](https://github.com/oxihep/oxihist) | Histograms: typed axes, weighted and profile storages | boost-histogram, hist |
| [oxiplot](https://github.com/oxihep/oxiplot) | Plotting to SVG/PNG/PDF with an mplhep look | matplotlib, mplhep |
| **oxifit** (this crate) | Curve fitting with Minuit2, and a `TFormula` engine | iminuit |
| [oxistats](https://github.com/oxihep/oxistats) | Statistics, HEP lineshapes and intervals | scipy.stats, hepstats |
| [oxiparticle](https://github.com/oxihep/oxiparticle) | PDG particle data and ID decoding | particle |

The crates are not on crates.io yet: depend on them from GitHub, pinned to a
release tag, as shown above.

## History

These crates were `oxiroot-fit` and `oxiroot-formula` inside the
[oxiroot](https://github.com/oxihep/oxiroot) workspace; their history was
carried over when they moved here.

## AI assistance

This project is developed with the help of AI coding assistants. Under the
author's direction they wrote and reviewed much of the code, the tests and
the documentation. Every change goes through the same checks either way: the
test suite and CI (including cross-checks between its two minimizers, and
fuzzing of the formula parser) and review before it is merged. If something
looks wrong, please open an issue.

## License

MIT — see [LICENSE](LICENSE).
