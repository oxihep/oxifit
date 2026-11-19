# Changelog

Notable changes to oxifit and oxifit-formula, by release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **A public cost API.** The `Cost` trait — parameter names, a `−2 ln L` or
  χ² value, optionally a gradient and a measurement count — and `Fitter`,
  which minimizes any cost with every minimizer, HESSE, MINOS, profiles and
  contours, with seeds, limits, fixed parameters, steps and Gaussian
  constraints set by name (iminuit's `Minuit(fcn)`). `FnCost` makes a cost of
  a closure (with an optional exact gradient), `ModelCost` makes one of a data
  fit (`ModelCost::fitter` is exactly `fit_opts`), and `CostSum` adds costs,
  merging parameters by name — a data fit plus an external measurement, or
  several datasets with shared parameters.

- **A comparison with ROOT, iminuit, scipy and zfit** (`comparison/`): the
  same data, models, seeds and costs fitted with each, with an agreement
  check, timings, and a side-by-side look at the code. Runs with
  `comparison/run_all.sh`; the `fit_comparison` example is oxifit's side.
- **`tests/reference_fitters.rs`**: the binned χ² and likelihood fits of the
  comparison against ROOT 6.40 and iminuit 2.33's parameters, errors and
  minima, for both model forms and three minimizers.

### Changed

- **minuit2 0.5's numerical-gradient panic no longer prints.** oxifit
  already caught it (an invalid fit, or a discarded start in a global
  search), but the default panic hook still printed `min > max, or either
  was NaN` each time — ten times in the `peak_search` example. A panic hook,
  installed once, now drops exactly that message while the thread is inside
  an oxifit minimization; every other panic, a bug in a model closure
  included, still reaches the program's own hook.

## [0.6.0]

### Changed

- **The default minimizer is `Minuit2Combined`**: MIGRAD, falling back to
  SIMPLEX and MIGRAD again when MIGRAD fails (ROOT's `"Minimize"`). Where
  MIGRAD succeeds nothing changes — the same result at the same cost —
  but minuit2 0.5's MIGRAD stalls from some seeds far from the minimum (an
  unbinned fit of 50 000 events seeded off by a width: MIGRAD invalid, the
  default converges). `Minimizer::Minuit2` keeps MIGRAD alone, ROOT's
  default. The combined minimizer is now oxifit's own rather than minuit2's
  `MnMinimize`, so each MIGRAD in it uses the model's analytic gradient.
  Profile and contour refits use it too.

## [0.5.0]

Analytic gradients and L-BFGS-B, 2-D and 3-D fits, convolution, a
`rayon` feature, and stricter models.

### Breaking

- `FitExt` is now implemented for `FitDataNd` (every 1-D `FitData` gets
  it through a blanket implementation, so `data.fit(&model)` is unchanged);
  code that named `FitExt: FitData` as a bound uses `FitDataNd`.
- `FitError` gains `DimensionMismatch`.

### Added

- **2-D and 3-D fits.** `Model::new2(|x, y, p| …)`, `Model::new3(|x, y, z, p|
  …)` and `Model::product("m", &[&fx, &fy(, &fz)])` (RooFit's
  `RooProdPdf`: its gradient by the product rule, its integral factorized);
  `Model::eval_at`, `gradient_at`, `with_gradient_at`, `check_gradient_at`
  and `dim`. Data: `Points2`/`Points3` (binned or graph data, `PointNd`) and
  `Events2`/`Events3` (unbinned, weighted or not), through the new
  `FitDataNd` trait. `FitOptions::range_y`/`range_z` select points and bound
  the unbinned normalization, which uses a model's known integral (products,
  sums of products) or nested adaptive quadrature. Every cost, minimizer,
  gradient, scan and simultaneous fit works in 2-D and 3-D; a model and data
  of different dimensions fit as invalid (`Model::check`,
  `FitError::DimensionMismatch`).

- **Convolution** (RooFit's `RooFFTConvPdf`): `Model::convolve(name,
  &signal, &resolution, window)` is `∫ f(x − t)·g(t) dt` over the window,
  integrated adaptively (a step such as a decay starting at `t = 0` is
  handled). Parameters are named after the two parts; with analytic
  gradients on both it has one too. Checked against the exponentially
  modified Gaussian to 1e-8.
- **`Model::normal`**, a unit-area Gaussian density with its gradient — a
  resolution function, or a shape for the unbinned likelihoods.

- **Analytic gradients.** `Model::with_gradient(|x, p, g| …)` supplies
  `∂f/∂p`; the Gaussian, exponential and polynomial shapes have theirs, a
  sum of models has one when every part does, and `Model::check_gradient`
  compares with finite differences. Every cost (χ², Pearson, both
  likelihoods, unbinned with its normalization, robust losses, Gaussian
  constraints, simultaneous fits) then has an exact gradient, used by:
  MIGRAD (minuit2's analytic path, falling back to finite differences when
  it stalls far from the minimum, as minuit2 0.5's does), the error matrix
  (a central difference of the gradient — `2n` gradient calls instead of
  HESSE's `~2n²` cost calls), and Levenberg–Marquardt's Jacobian.
  `FitOptions::with_gradient(false)` turns it off. A 30-parameter fit of
  ten peaks: 186 → 149 ms.
- **`Minimizer::LBfgsB`**: L-BFGS-B, projected limited-memory quasi-Newton
  within the limits (SciPy's and zfit's gradient minimizer), then MIGRAD +
  HESSE; central differences without an analytic gradient. The ten-peak fit:
  126 ms.

- **The `rayon` feature.** A cost over more than 4096 points (χ², binned and
  unbinned likelihoods, the residuals) is summed in parallel chunks; toys,
  profile points and contour rays are fitted in parallel. The chunk sums
  are added in a fixed order and toy samples are drawn before the fits, so
  a result is bit-identical on any number of threads. An unbinned fit of
  10⁶ events: 920 ms → 151 ms on 12 cores.

### Changed

- **A misspelled parameter name no longer passes silently.** `limit`,
  `lower_limit`, `upper_limit`, `fix`, `step` and `constrain` used to ignore
  a name the model does not have, so a typo gave a different fit than the
  one asked for. The model now records such names: `Model::check()` reports
  the first as `FitError::UnknownParameter`, a fit of the model is invalid,
  and profiles and contours return the error. This carries through sums and
  simultaneous fits.
- **Nelder–Mead holds limits exactly.** It now moves in Minuit's internal
  coordinates (a sine transform for two limits, a square root for one)
  instead of adding a `1e30·d²` penalty outside them. On a fit pinned at a
  limit it lands on the exact constrained minimum.

## [0.4.1]

### Fixed

- **`oxifit_formula::integrate` could run for hours.** It bisected depth
  first, halving its absolute tolerance at every level. Near a narrow peak
  that tolerance soon fell below the 15-point rule's error floor, so every
  interval there recursed 40 levels deep, about 2⁴⁰ rules: a unit Gaussian
  of width 0.01 on [−1, 1] never returned. A `NaN` anywhere in the integrand
  did the same, since a `NaN` error never meets the tolerance. `integrate`
  now uses the globally adaptive scheme of `integrate_adaptive`: it starts
  from 16 intervals, bisects the worst one, and stops at a `1e-11` relative
  (`1e-13` absolute) total error. Its cost is bounded at 2000 intervals,
  about 60 000 evaluations, and a non-finite integrand returns at once. It
  is also more accurate where the old scheme stepped over a feature:
  ∫1/√|x| on [−1, 1] gives 4 instead of ∞. oxiroot's `Func1D` integrals,
  checked against ROOT, are unchanged.

## [0.4.0]

Unbinned and extended likelihood fits with the SumW2 correction, toy
studies, and simultaneous fits.

### Added

- **Simultaneous fits** (RooFit's `RooSimultaneous`).
  `Simultaneous::new().channel(name, &data, &model, &opts)…` fits several
  datasets together, each with its own model and cost settings (any method,
  binned or unbinned, range, robust loss); parameters with the same name are
  shared, their seed and settings taken from the first channel that has
  them. `.fit(&opts)` returns a `FitResult` over `param_names()` — the
  summed cost, `chi2` when every channel has one, `ndf` over all channels —
  and `channel_costs` splits the cost by channel. Every minimizer works
  (Levenberg–Marquardt on the joint residuals when no channel is unbinned).

- **Toy studies** (RooFit's `RooMCStudy`). `Toys::binned(&truth, &xs)`
  draws Poisson counts at fixed points; `Toys::unbinned(&truth, range, n)`
  draws `n` events from the model's shape (`.extended()`: a Poisson number of
  mean `∫f`). `.run(&model, &opts, ntoys)` fits each sample into a
  `ToyStudy`, with `pulls`, `values`, `valid_fraction` and `pull_summary`
  (the pulls' mean and width with their standard errors). Seeded and
  reproducible; Poisson draws are exact at any mean (PTRS above 10).

- **Unbinned likelihood fits.** `Events::new(&values)` /
  `Events::weighted(&values, &weights)` hold individual events.
  `FitMethod::UnbinnedLikelihood` minimizes `−2·Σ w·ln(f/∫f)`, the model
  normalized numerically over the fit range (default: the events' span);
  `FitMethod::ExtendedUnbinnedLikelihood` minimizes `−2·[Σ w·ln f − ∫f]`,
  the model being the expected event density, so the yield is fitted too.
  `chi2` is `NaN` for both (no goodness of fit). Least-squares backends
  (Levenberg–Marquardt) fall back to MIGRAD for them.
- **`FitOptions::with_sumw2`** corrects a weighted unbinned fit's covariance
  for the weights, `V = H⁻¹·W·H⁻¹` (RooFit's `SumW2Error`); without it the
  errors are those of `Σw` unit-weight events.
- **`oxifit_formula::integrate_adaptive`**: globally adaptive Gauss–Kronrod
  with a cap on the number of intervals, so its cost is bounded; it
  normalizes the unbinned likelihoods.
- The `lifetime_fit` example adds an unbinned fit of the decay times.

## [0.3.0]

CMA-ES and Levenberg–Marquardt, sum models with Gaussian constraints, and
profile likelihoods and contours.

### Added

- **Profiles and contours.** `FitExt::profile` scans the cost in one
  parameter, refitting the others at each value; `Profile::interval(level)`
  gives the `Δ = level` interval (1σ at 1, 2σ at 4) from the scan.
  `FitExt::contour` traces the 2-D region where the profiled cost is `Δ`
  above its minimum (2.30 for a 68 % joint region, 5.99 for 95 %), each point
  found along a ray from the minimum with the other parameters refitted.
  (minuit2 0.5's `MnContours` does not re-minimize at its starting points,
  so it misplaces contours of correlated parameters; oxifit does not use it.)
- **`FitError`**, the error type of the scans: an unknown or fixed
  parameter, or a failed fit.

- **Sum models.** `Model::sum("m", &[&sig, &bkg])`, or `sig + bkg`, adds
  models term by term. Parameters are named after their part (`sig.mean`,
  `bkg.slope`; a repeated part name is numbered, `g_2.mean`), and each part
  keeps its seeds, limits, fixed parameters and constraints. `a + b + c`
  appends to the sum rather than nesting it.
- **Gaussian constraints.** `.constrain("name", mean, sigma)` adds
  `((p − mean)/sigma)²` to the cost — an external measurement, or a nuisance
  parameter's prior — and counts as one more measurement in `ndf`. It works
  with every cost and minimizer (Levenberg–Marquardt sees it as a residual).

- **`Minimizer::CmaEs(CmaEs::new())`** — CMA-ES, the covariance matrix
  adaptation evolution strategy, over the parameter limits, then MIGRAD +
  HESSE. It learns the shape of the cost's valley, so it copes with
  correlated or badly scaled parameters. On the 200-toy peak search it finds
  the peak every time, like differential evolution, in a third of the
  evaluations (≈1700 against ≈5600).
- **`Minimizer::LevenbergMarquardt`** — damped Gauss–Newton steps on the
  residuals, then MIGRAD + HESSE. Every cost is written as a sum of squared
  residuals (`(y − f)/σ`, Pearson's `(y − f)/√f`, the likelihood's signed
  deviance residuals, a robust loss as `√ρ`), so it applies to all of them.
  From a good seed it needs 20–30 % fewer evaluations than MIGRAD.

## [0.2.0]

Global minimizers, HESSE by default, and a fit that can no longer panic.

### Breaking

- **`FitResult` is `#[non_exhaustive]`** and gains `fval` (the minimized
  cost), `edm` (Minuit2's estimated distance to the minimum) and `nfcn` (cost
  evaluations, HESSE and MINOS included). Code that built a `FitResult` with a
  struct literal uses `FitResult::new(params, errors, chi2, ndf, valid)` and
  sets the remaining public fields.
- **`chi2` is `NaN` under a robust loss**, whose cost is no χ²; the cost is in
  `fval`, and `p_value()` is `NaN` too. Without a robust loss `chi2 == fval`
  as before.
- `Minimizer` has new variants, two of which carry settings.

### Added

- **Global minimizers** for a seed that may sit in the wrong basin:
  `Minimizer::MultiStart(MultiStart::new(n))` runs `n` local fits from a
  Latin hypercube over the parameter limits and keeps the lowest;
  `Minimizer::DifferentialEvolution(DifferentialEvolution::new())` evolves a
  population over the limits (DE/rand/1/bin) and polishes the best point with
  MIGRAD. Both search inside each free parameter's limits, are reproducible
  from their `seed`, and end with MIGRAD + HESSE like any Minuit fit. On a
  peak fit seeded across the whole range (200 toys), MIGRAD found the peak in
  35, multistart and differential evolution in all 200.
- `Minimizer::Minuit2Combined` (MIGRAD with a SIMPLEX fallback, ROOT's
  `"Minimize"`) and `Minimizer::Simplex`.
- `FitOptions::strategy`, `tolerance` and `max_fcn` for Minuit2 (ROOT's
  `SetStrategy`, `SetTolerance`, `SetMaxFunctionCalls`).
- The `peak_search` example.

- `erf`, `erfc` and `lgamma` (`TMath::Erf`, `TMath::Erfc`, `TMath::LnGamma`)
  in formulas — enough for an error-function turn-on curve.

### Changed

- The engine is split into the Minuit2 backends, the global searches and the
  Nelder–Mead backend, sharing one finishing step (HESSE, MINOS, the result).

- **HESSE runs after MIGRAD by default**, as in ROOT's `Fit`: `errors` and
  `covariance` now come from a finite-difference Hessian at the minimum rather
  than MIGRAD's running estimate, and a minimum whose Hessian is not
  positive-definite is reported invalid. `FitOptions::with_hesse(false)` turns
  it off (e.g. for many toy fits where only the values matter).

### Fixed

- **A fit could panic.** From some poor seeds the numerical gradient in the
  `minuit2` 0.5 dependency calls `f64::clamp` with its bounds reversed, which
  panics; the panic unwound through `fit`/`fit_opts` into the caller. The
  engine now catches a panic from the minimizer and returns an invalid
  `FitResult` (the default panic hook still prints the message).

### Testing

- The costs against their closed forms (a constant fitted with the Poisson
  likelihood, Neyman and Pearson χ²), exact HESSE errors of a linear fit,
  MINOS, limits, and a recorded dataset that made the backend panic.

## [0.1.2]

### Fixed

- **A Nelder–Mead fit stopped by its iteration cap reported `valid`.** With
  the `argmin` feature, `Minimizer::NelderMead` set `valid` from a finite χ²
  alone, so a run that hit the 10 000-iteration limit before the simplex
  converged still looked like a good fit. `valid` now also requires
  convergence, as Minuit2's `is_valid()` does.

### Changed

- **Rust 1.100 and edition 2024.** The minimum supported Rust version
  (`rust-version`) moves from 1.95 to 1.100, the current stable release that
  CI already builds with, and the crates move to the 2024 edition (migrated
  with `cargo fix --edition`; formatting follows the 2024 style).

### Testing

- A cargo-fuzz target for the formula parser (`fuzz/`): whatever parses must
  evaluate without panicking, and its ROOT spelling must parse again. CI
  builds it on every PR and runs it for two minutes weekly.

## [0.1.1]

### Fixed

- **oxifit-formula: formulas from files can no longer crash the process.**
  Parsing is limited to 200 levels of nesting (deep parentheses, unary
  operators or calls used to overflow the stack), parameter indices to 10 000
  (`[1e12]` and huge shortcut offsets used to allocate or overflow), and `polN`
  to degree 100. Each limit is a `ParseError`.

## [0.1.0]

First release under the oxifit names. The code is `oxiroot-fit` and
`oxiroot-formula` 0.3.0 from the [oxiroot](https://github.com/oxihep/oxiroot)
workspace, moved to their own repository with their history.

### Changed

- Renamed `oxiroot-fit` to `oxifit` and `oxiroot-formula` to `oxifit-formula`
  (oxiroot users keep `oxiroot::fit`, which re-exports `oxifit`).
- Depends on `oxistats` (formerly `oxiroot-stat`); `oxifit::StatError` is
  `oxistats::StatError`.
