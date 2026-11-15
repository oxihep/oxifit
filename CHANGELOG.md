# Changelog

Notable changes to oxifit and oxifit-formula, by release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
