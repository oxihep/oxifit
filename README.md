# oxifit

[![CI](https://github.com/oxihep/oxifit/actions/workflows/ci.yml/badge.svg)](https://github.com/oxihep/oxifit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.95+](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org)

Parametric curve fitting in pure Rust for **any 1-D data**: histograms, graphs
or raw `(x, y, σ)` points. The minimizer is the pure-Rust
[Minuit2](https://crates.io/crates/minuit2) port, the algorithm ROOT uses.

Part of the [oxihep](https://github.com/oxihep) family of Rust crates for
high-energy physics, alongside [oxiroot](https://github.com/oxihep/oxiroot)
(ROOT file IO, whose histograms and graphs implement `FitData`) and
[oxistats](https://github.com/oxihep/oxistats) (statistics and lineshapes).

| Crate | What it is |
| --- | --- |
| [`oxifit`](crates/oxifit) | The fitter: `FitData`/`FitExt`, `Model`, χ² / Poisson-likelihood costs, robust losses, Minuit2 (and an optional Nelder–Mead `argmin` backend) |
| [`oxifit-formula`](crates/oxifit-formula) | A ROOT `TFormula` expression engine with no dependencies: parse and evaluate formula strings, numerical integral and derivative. `Model::from_formula` and oxiroot's `Func1D`/`Func2D`/`Func3D` are built on it. |

```toml
[dependencies]
oxifit = { git = "https://github.com/oxihep/oxifit" }
```

```rust
use oxifit::{FitExt, Model, Points};

let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0], &[0.1; 4])?;
let line = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
let fit = data.fit(&line); // χ² by default
assert!(fit.valid);
# Ok::<(), oxifit::StatError>(())
```

Models come from a closure, a built-in shape (Gaussian, exponential,
polynomial, Crystal Ball, Breit–Wigner, Voigt, Novosibirsk, ARGUS, …) or any
formula string (`Model::from_formula("f", "[0]*exp(-0.5*((x-[1])/[2])^2)")`).

The [fitting guide](https://oxihep.github.io/oxiroot/guide/fitting/) covers
costs, limits, robust losses and fitting ROOT histograms (it uses the
`oxiroot::fit` re-export, which is this crate).

## Features

| Feature | Default | Adds |
| --- | --- | --- |
| `argmin` | no | A gradient-free Nelder–Mead backend via the `argmin` crate, chosen with `FitOptions::minimizer(Minimizer::NelderMead)` |

## History

These crates were `oxiroot-fit` and `oxiroot-formula` inside the
[oxiroot](https://github.com/oxihep/oxiroot) workspace; their history was
carried over when they moved here.

## License

MIT — see [LICENSE](LICENSE).
