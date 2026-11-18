# Changelog

Notable changes to oxifit and oxifit-formula, by release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
