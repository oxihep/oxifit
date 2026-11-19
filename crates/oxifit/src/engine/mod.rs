//! The fit engine: turns a set of [`Point`]s, a [`Model`], and [`FitOptions`]
//! into a [`FitResult`]. Data-source agnostic — it only ever sees `(x, y, σ)`
//! triples, so it fits histograms, graphs, and custom data alike.
//!
//! [`run_fit`] builds the cost and a [`Fit`] (everything a backend needs), then
//! hands it to the chosen [`Minimizer`]:
//!
//! - [`minuit`] — MIGRAD, MIGRAD-then-SIMPLEX, or SIMPLEX, and the shared
//!   [`minuit::finish`] step (HESSE, MINOS, the [`FitResult`]) every
//!   Minuit-polished backend ends with;
//! - [`global`] and [`cmaes`] — multistart, differential evolution and
//!   CMA-ES, which search for the right basin and then polish the best point
//!   with MIGRAD;
//! - [`lm`] — Levenberg–Marquardt on the residual vector, then MIGRAD;
//! - `nelder_mead` — the independent argmin simplex (`argmin` feature).

mod cmaes;
mod global;
mod linalg;
mod lm;
pub(crate) mod minuit;
#[cfg(feature = "argmin")]
mod nelder_mead;

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::data::Point;
use crate::model::{Constraint, Model};
use crate::result::{FitMethod, FitOptions, FitResult, Loss, Minimizer};

/// The most intervals the unbinned normalization integral may use.
const NORM_INTERVALS: usize = 200;

/// A cost over the full parameter vector.
pub(crate) type CostFn<'a> = &'a dyn Fn(&[f64]) -> f64;

/// Everything a backend needs to minimize one fit.
pub(crate) struct Fit<'a> {
    /// The model: parameter names, seeds, and constraints.
    pub(crate) model: &'a Model,
    /// The cost over the full parameter vector (fixed parameters included).
    pub(crate) cost: &'a dyn Fn(&[f64]) -> f64,
    /// The residuals over the full parameter vector, written into a slice of
    /// [`n_residuals`](Self::n_residuals): their squares sum to the cost.
    pub(crate) residuals: &'a dyn Fn(&[f64], &mut [f64]),
    /// How many residuals there are (`0`: the cost is no sum of squares — the
    /// unbinned likelihoods — and least-squares backends fall back to MIGRAD).
    pub(crate) n_residuals: usize,
    /// For a SumW2-corrected fit of weighted events: the cost with squared
    /// weights, whose Hessian is the sandwich's middle.
    pub(crate) cost_w2: Option<CostFn<'a>>,
    /// Degrees of freedom: fitted points + constraints − free parameters.
    pub(crate) ndf: usize,
    /// The fit options (strategy, tolerance, HESSE, MINOS, …).
    pub(crate) opts: &'a FitOptions,
    /// Whether the cost at the minimum is a goodness-of-fit statistic (a χ² or
    /// a Poisson likelihood ratio) — not so under a robust loss.
    pub(crate) gof: bool,
}

impl Fit<'_> {
    /// Parameter `i`'s constraints.
    pub(crate) fn constraint(&self, i: usize) -> Constraint {
        self.model.constraints.get(i).copied().unwrap_or_default()
    }

    /// Indices of the free (non-fixed) parameters.
    pub(crate) fn free(&self) -> Vec<usize> {
        (0..self.model.params.len())
            .filter(|&i| !self.constraint(i).fixed)
            .collect()
    }

    /// The initial step for parameter `i` at `value`: the hint, else 10 % of
    /// the value with a floor.
    pub(crate) fn step(&self, i: usize, value: f64) -> f64 {
        self.constraint(i)
            .step
            .unwrap_or_else(|| (value.abs() * 0.1).max(0.01))
    }

    /// The model's seed, moved inside any limits it lies outside.
    pub(crate) fn seed(&self) -> Vec<f64> {
        self.model
            .params
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let c = self.constraint(i);
                let v = c.lower.map_or(v, |lo| v.max(lo));
                c.upper.map_or(v, |hi| v.min(hi))
            })
            .collect()
    }
}

/// Fit `model` to `all` under `opts`. The points entering the fit are those with
/// `x` in `opts.range` (default: all); Neyman chi-square additionally drops
/// points with no usable error (`sigma <= 0`).
pub(crate) fn run_fit(all: &[Point], model: &Model, opts: &FitOptions) -> FitResult {
    let run = |fit: &Fit| {
        // A backend may panic on a pathological point (minuit2 0.5's numerical
        // gradient can, from a poor seed). Report that as a failed fit rather
        // than unwinding into the caller. Nothing is shared across the
        // boundary but the read-only model and data, so a caught panic leaves
        // no broken state behind. The default panic hook still prints the
        // message to stderr.
        catch_unwind(AssertUnwindSafe(|| match opts.minimizer {
            Minimizer::Minuit2 => minuit::run(fit, minuit::Local::Migrad),
            Minimizer::Minuit2Combined => minuit::run(fit, minuit::Local::Combined),
            Minimizer::Simplex => minuit::run(fit, minuit::Local::Simplex),
            Minimizer::MultiStart(cfg) => global::multi_start(fit, cfg),
            Minimizer::DifferentialEvolution(cfg) => global::differential_evolution(fit, cfg),
            Minimizer::CmaEs(cfg) => cmaes::cma_es(fit, cfg),
            Minimizer::LevenbergMarquardt => lm::levenberg_marquardt(fit),
            #[cfg(feature = "argmin")]
            Minimizer::NelderMead => nelder_mead::minimize(fit),
        }))
        .unwrap_or_else(|_| failed(model))
        .with_sumw2(fit)
    };
    match with_fit(all, model, opts, run) {
        Some((mut result, calls)) => {
            result.nfcn = calls;
            result
        }
        None => failed(model),
    }
}

/// Build the [`Fit`] for `model` on `all` under `opts` — the selected points,
/// the cost and residuals (with any Gaussian constraints), `ndf` — and run
/// `body` on it. Returns `body`'s result and the number of cost evaluations it
/// made, or `None` when there are too few measurements to determine the free
/// parameters.
pub(crate) fn with_fit<R>(
    all: &[Point],
    model: &Model,
    opts: &FitOptions,
    body: impl FnOnce(&Fit) -> R,
) -> Option<(R, usize)> {
    let np = model.params.len();
    // Neyman chi-square uses only points with an error; the others use every one.
    let drop_empty = matches!(opts.method, FitMethod::Chi2);
    let points: Vec<(f64, f64, f64)> = all
        .iter()
        .filter_map(|p| {
            if let Some((lo, hi)) = opts.range
                && (p.x < lo || p.x > hi)
            {
                return None;
            }
            (!drop_empty || p.sigma > 0.0).then_some((p.x, p.y, p.sigma))
        })
        .collect();

    // Free (non-fixed) parameters determine the degrees of freedom.
    let n_free = (0..np)
        .filter(|&i| !model.constraints.get(i).map(|c| c.fixed).unwrap_or(false))
        .count();

    // Gaussian constraints on free parameters: `(index, mean, sigma)`. Each is
    // an auxiliary measurement — a term in the cost, a residual, and a point
    // in `ndf`.
    let penalties: Vec<(usize, f64, f64)> = model
        .constraints
        .iter()
        .take(np)
        .enumerate()
        .filter(|(_, c)| !c.fixed)
        .filter_map(|(i, c)| c.gauss.map(|(m, s)| (i, m, s)))
        .collect();
    let n_meas = points.len() + penalties.len();

    // Too few measurements to determine the free parameters: report failure
    // rather than a meaningless minimum of a flat cost.
    if np == 0 || n_meas < n_free.max(1) {
        return None;
    }

    let func = &model.func;
    let method = opts.method;
    let unbinned = matches!(
        method,
        FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood
    );
    // The unbinned likelihoods normalize over the fit range: the given one,
    // else the span of the events.
    let (lo, hi) = opts.range.unwrap_or_else(|| {
        points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p.0), b.max(p.0))
            })
    });
    let (sum_w, sum_w2) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), p| (a + p.1, b + p.1 * p.1));
    // `−2 ln L` with each event weighted by `w^power` (power 2 for the SumW2
    // sandwich, whose extended term scales by Σw²/Σw as in RooFit).
    let unbinned_cost = |p: &[f64], power: i32| -> f64 {
        // Bounded, globally adaptive quadrature: a spike-like trial shape
        // (a width driven toward zero) costs at most NORM_INTERVALS splits.
        let norm =
            oxifit_formula::integrate_adaptive(|x| func(x, p), lo, hi, 1e-10, NORM_INTERVALS).0;
        let w = |y: f64| if power == 1 { y } else { y.powi(power) };
        let logs: f64 = points
            .iter()
            .map(|&(x, y, _)| w(y) * func(x, p).max(1e-300).ln())
            .sum();
        if method == FitMethod::ExtendedUnbinnedLikelihood {
            let scale = if power == 1 || sum_w == 0.0 {
                1.0
            } else {
                sum_w2 / sum_w
            };
            -2.0 * (logs - scale * norm)
        } else {
            let wsum = if power == 1 { sum_w } else { sum_w2 };
            -2.0 * (logs - wsum * norm.max(1e-300).ln())
        }
    };
    let loss = opts.loss;
    // Robust-loss residual scale squared; a non-positive `f_scale` means 1.
    let fs2 = {
        let fs = if opts.f_scale > 0.0 {
            opts.f_scale
        } else {
            1.0
        };
        fs * fs
    };
    let data_cost = |p: &[f64]| -> f64 {
        match method {
            FitMethod::Chi2 => points
                .iter()
                .map(|&(x, y, e)| {
                    let d = (y - func(x, p)) / e;
                    robust(d * d, loss, fs2)
                })
                .sum(),
            FitMethod::PearsonChi2 => points
                .iter()
                .map(|&(x, y, _)| {
                    let f = func(x, p).max(1e-300); // expected (model) variance
                    robust((y - f) * (y - f) / f, loss, fs2)
                })
                .sum(),
            FitMethod::Likelihood => {
                2.0 * points
                    .iter()
                    .map(|&(x, y, _)| {
                        let f = func(x, p).max(1e-300);
                        if y > 0.0 { f - y + y * (y / f).ln() } else { f }
                    })
                    .sum::<f64>()
            }
            FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => {
                unbinned_cost(p, 1)
            }
        }
    };

    let pull = |p: &[f64], &(i, m, sd): &(usize, f64, f64)| (p[i] - m) / sd;
    let cost = |p: &[f64]| -> f64 {
        data_cost(p) + penalties.iter().map(|c| pull(p, c).powi(2)).sum::<f64>()
    };

    // The same cost as residuals whose squares sum to it, for least-squares
    // backends: (y − f)/σ for χ², with a robust loss as √ρ, and the signed
    // deviance residual √(2·[f − y + y·ln(y/f)]) for the likelihood.
    let residuals = |p: &[f64], out: &mut [f64]| {
        for (o, &(x, y, e)) in out.iter_mut().zip(&points) {
            *o = match method {
                FitMethod::Chi2 => {
                    let d = (y - func(x, p)) / e;
                    robust(d * d, loss, fs2).max(0.0).sqrt().copysign(d)
                }
                FitMethod::PearsonChi2 => {
                    let f = func(x, p).max(1e-300);
                    let d = (y - f) / f.sqrt();
                    robust(d * d, loss, fs2).max(0.0).sqrt().copysign(d)
                }
                FitMethod::Likelihood => {
                    let f = func(x, p).max(1e-300);
                    let t = if y > 0.0 { f - y + y * (y / f).ln() } else { f };
                    (2.0 * t).max(0.0).sqrt().copysign(f - y)
                }
                // No residuals: `n_residuals` is 0 for these.
                FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => 0.0,
            };
        }
        for (o, c) in out[points.len()..].iter_mut().zip(&penalties) {
            *o = pull(p, c);
        }
    };

    // Count every cost evaluation, whichever backend makes it (a residual
    // vector counts as one).
    let calls = Cell::new(0_usize);
    let counted = |p: &[f64]| {
        calls.set(calls.get() + 1);
        cost(p)
    };
    let counted_residuals = |p: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        residuals(p, out);
    };
    let weighted = points.iter().any(|p| p.1 != 1.0);
    let cost_w2 = |p: &[f64]| -> f64 {
        unbinned_cost(p, 2) + penalties.iter().map(|c| pull(p, c).powi(2)).sum::<f64>()
    };
    let fit = Fit {
        model,
        cost: &counted,
        residuals: &counted_residuals,
        n_residuals: if unbinned { 0 } else { n_meas },
        cost_w2: (unbinned && weighted && opts.sumw2).then_some(&cost_w2 as CostFn<'_>),
        ndf: n_meas - n_free,
        opts,
        // The unbinned likelihoods have no goodness of fit; a robust loss is
        // ignored by the binned likelihood, so only it changes the meaning of
        // the least-squares costs.
        gof: !unbinned
            && (matches!(opts.method, FitMethod::Likelihood) || opts.loss == Loss::Linear),
    };

    let result = body(&fit);
    Some((result, calls.get()))
}

impl FitResult {
    /// Replace the covariance of a weighted fit by the SumW2 sandwich
    /// `V = H⁻¹·W·H⁻¹` (RooFit's `SumW2Error`), when the fit asks for it. With
    /// the cost `F = −2 ln L`, HESSE's covariance is `C = 2·H_F⁻¹`, so
    /// `V = C·W_F·C / 2`, `W_F` the Hessian of `F` with squared weights.
    fn with_sumw2(mut self, fit: &Fit) -> FitResult {
        let (Some(cost_w2), Some(cov)) = (fit.cost_w2, self.covariance.as_ref()) else {
            return self;
        };
        if !self.valid {
            return self;
        }
        let free = fit.free();
        let sub: Vec<f64> = free.iter().map(|&i| self.params[i]).collect();
        // Steps of a twentieth of each error: well inside the quadratic region.
        let h: Vec<f64> = (0..free.len())
            .map(|k| {
                let e = cov[k][k].sqrt();
                if e.is_finite() && e > 0.0 {
                    0.05 * e
                } else {
                    1e-4 * sub[k].abs().max(1.0)
                }
            })
            .collect();
        let params = self.params.clone();
        let f = |s: &[f64]| cost_w2(&global::scatter(&params, &free, s));
        let w = linalg::hessian(&f, &sub, &h);
        let n = free.len();
        let cw: Vec<Vec<f64>> = (0..n)
            .map(|a| {
                (0..n)
                    .map(|b| (0..n).map(|k| cov[a][k] * w[k][b]).sum())
                    .collect()
            })
            .collect();
        let v: Vec<Vec<f64>> = (0..n)
            .map(|a| {
                (0..n)
                    .map(|b| 0.5 * (0..n).map(|k| cw[a][k] * cov[k][b]).sum::<f64>())
                    .collect()
            })
            .collect();
        for (k, &i) in free.iter().enumerate() {
            self.errors[i] = if v[k][k] > 0.0 {
                v[k][k].sqrt()
            } else {
                f64::NAN
            };
        }
        self.covariance = Some(v);
        self
    }
}

/// An always-invalid result: no minimum was found (too few points, or the
/// optimizer failed to start, converge, or panicked).
pub(crate) fn failed(model: &Model) -> FitResult {
    let np = model.params.len();
    FitResult::new(model.params.clone(), vec![f64::NAN; np], f64::NAN, 0, false)
}

/// Apply a robust loss `ρ` to a squared residual `z` at residual scale `C`
/// (`fs2 = C²`): the point's cost contribution is `C²·ρ(z / C²)`, matching the
/// `loss` of `scipy.optimize.least_squares`. [`Loss::Linear`] returns `z`
/// unchanged (an ordinary least-squares fit), so the default path is untouched.
fn robust(z: f64, loss: Loss, fs2: f64) -> f64 {
    match loss {
        Loss::Linear => z,
        Loss::SoftL1 => 2.0 * fs2 * ((1.0 + z / fs2).sqrt() - 1.0),
        Loss::Huber => {
            let u = z / fs2;
            fs2 * if u <= 1.0 { u } else { 2.0 * u.sqrt() - 1.0 }
        }
        Loss::Cauchy => fs2 * (1.0 + z / fs2).ln(),
        Loss::Arctan => fs2 * (z / fs2).atan(),
    }
}
