# oxifit

[![CI](https://github.com/oxihep/oxifit/actions/workflows/ci.yml/badge.svg)](https://github.com/oxihep/oxifit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.100+](https://img.shields.io/badge/rust-1.100%2B-orange.svg)](https://www.rust-lang.org)
[![API docs](https://img.shields.io/badge/docs-API-blue.svg)](https://oxihep.github.io/oxifit/oxifit/)

Parametric curve fitting in pure Rust for **any 1-D data**: histograms, graphs
or raw `(x, y, σ)` points. The minimizer is the pure-Rust
[Minuit2](https://crates.io/crates/minuit2) port, the algorithm ROOT uses.

| Crate | What it is |
| --- | --- |
| [`oxifit`](crates/oxifit) | The fitter: `FitData`/`FitExt`, `Model`, χ² / Poisson-likelihood costs, robust losses, Minuit2 (MIGRAD + HESSE, MINOS), Levenberg–Marquardt, global searches (multistart, differential evolution, CMA-ES) and an optional Nelder–Mead `argmin` backend |
| [`oxifit-formula`](crates/oxifit-formula) | A ROOT `TFormula` expression engine (its only dependency is oxistats, for `erf`/`erfc`/`lgamma`): parse and evaluate formula strings, numerical integral and derivative. `Model::from_formula` and oxiroot's `Func1D`/`Func2D`/`Func3D` are built on it. |

```toml
[dependencies]
oxifit = { git = "https://github.com/oxihep/oxifit", tag = "v0.2.0" }
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
`.constrain("bkg.slope", -0.02, 0.005)`.

## Minimizers

`FitOptions::minimizer` picks the optimizer. Every Minuit-based choice ends with
MIGRAD and HESSE, so values, errors, covariance and `valid` mean the same
thing whichever one found the minimum.

| `Minimizer` | What it does | When |
| --- | --- | --- |
| `Minuit2` (default) | MIGRAD from the model's parameters, then HESSE | A seed near the answer |
| `Minuit2Combined` | MIGRAD, falling back to SIMPLEX when it fails (ROOT's `"Minimize"`) | MIGRAD sometimes fails to converge |
| `Simplex` | Minuit2's SIMPLEX, then HESSE | A rough, noisy cost |
| `MultiStart(MultiStart::new(n))` | `n` local fits from points spread over the parameter limits; the lowest wins | A few competing minima |
| `DifferentialEvolution(…)` | A population-based global search over the limits, then MIGRAD | The seed may sit in the wrong basin (a peak of unknown position, a multimodal model) |
| `CmaEs(CmaEs::new())` | CMA-ES: an evolution strategy that learns the valley's shape, then MIGRAD | As above, usually in a third of the evaluations; copes with correlated parameters |
| `LevenbergMarquardt` | Damped Gauss–Newton on the residuals, then MIGRAD | A good seed: fewer evaluations than MIGRAD on least-squares and likelihood fits |
| `NelderMead` (`argmin` feature) | An independent simplex with a numerical Hessian | Cross-checking Minuit2 |

The global searches look inside each free parameter's limits, so set a
`limit` on the parameters to be searched. They are seeded and reproducible.
`FitOptions::strategy`, `tolerance` and `max_fcn` tune Minuit2, and
`with_hesse(false)` skips HESSE when only the values matter.

The [fitting guide](https://oxihep.github.io/oxiroot/guide/fitting/) covers
costs, limits, robust losses and fitting ROOT histograms (it uses the
`oxiroot::fit` re-export, which is this crate).

## Features

| Feature | Default | Adds |
| --- | --- | --- |
| `argmin` | no | A gradient-free Nelder–Mead backend via the `argmin` crate, chosen with `FitOptions::minimizer(Minimizer::NelderMead)` |

## Examples

Physics fits on toy data with known truth, each printing the fitted values
next to it (`cargo run -p oxifit --example <name>`):

| Example | Shows |
| --- | --- |
| [`resonance_fit`](crates/oxifit/examples/resonance_fit.rs) | A Z → μμ peak on a falling background: a sideband fit seeds ROOT's `gaus(0)+expo(3)`, fitted with χ² and with a Poisson likelihood, with MINOS errors |
| [`trigger_turn_on`](crates/oxifit/examples/trigger_turn_on.rs) | A trigger efficiency turn-on fitted with an error-function model (a custom `Model` closure), and the pT where it is fully efficient |
| [`lifetime_fit`](crates/oxifit/examples/lifetime_fit.rs) | The B⁰ lifetime from its decay-time distribution, and why a likelihood beats a χ² on sparse tails |
| [`peak_search`](crates/oxifit/examples/peak_search.rs) | A narrow peak at an unknown mass: MIGRAD from a blind seed misses it, multistart, differential evolution and CMA-ES find it |

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
