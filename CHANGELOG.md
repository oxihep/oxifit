# Changelog

Notable changes to oxifit and oxifit-formula, by release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
