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
//! - [`lbfgsb`] — projected L-BFGS within the limits, then MIGRAD;
//! - `nelder_mead` — the independent argmin simplex (`argmin` feature).

mod cmaes;
mod global;
mod lbfgsb;
mod linalg;
mod lm;
pub(crate) mod minuit;
#[cfg(feature = "argmin")]
mod nelder_mead;
#[cfg(feature = "argmin")]
mod transform;

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::data::PointNd;
use crate::model::{Constraint, Model};
use crate::result::{FitMethod, FitOptions, FitResult, Loss, Minimizer};

/// Points per chunk of a cost sum. The sum is taken chunk by chunk and the
/// chunk sums are added in order, whether the chunks run on one thread or
/// many (`rayon` feature), so the result does not depend on the threads.
const CHUNK: usize = 4096;

/// `Σ term(point)` over `points`, chunked (see [`CHUNK`]); in parallel with the
/// `rayon` feature once there is more than one chunk.
fn chunked_sum<T: Sync>(points: &[T], term: impl Fn(&T) -> f64 + Sync) -> f64 {
    let chunk_sum = |c: &[T]| c.iter().map(&term).sum::<f64>();
    #[cfg(feature = "rayon")]
    if points.len() > CHUNK {
        use rayon::prelude::*;
        let sums: Vec<f64> = points.par_chunks(CHUNK).map(chunk_sum).collect();
        return sums.iter().sum();
    }
    points.chunks(CHUNK).map(chunk_sum).sum()
}

/// `Σ term(point, acc)` into a vector of length `n`, chunked like
/// [`chunked_sum`]: each chunk accumulates its own vector and the chunk
/// vectors are added in order.
fn chunked_vec_sum<T: Sync>(
    points: &[T],
    n: usize,
    term: impl Fn(&T, &mut [f64], &mut [f64]) + Sync,
    out: &mut [f64],
) {
    // Each chunk gets its accumulator and a scratch buffer of length `n`
    // (allocated once per chunk, not per point).
    let chunk_vec = |c: &[T]| {
        let mut acc = vec![0.0; n];
        let mut scratch = vec![0.0; n];
        for pt in c {
            term(pt, &mut acc, &mut scratch);
        }
        acc
    };
    #[cfg(feature = "rayon")]
    let parts: Vec<Vec<f64>> = if points.len() > CHUNK {
        use rayon::prelude::*;
        points.par_chunks(CHUNK).map(chunk_vec).collect()
    } else {
        points.chunks(CHUNK).map(chunk_vec).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let parts: Vec<Vec<f64>> = points.chunks(CHUNK).map(chunk_vec).collect();
    for part in parts {
        for (o, v) in out.iter_mut().zip(part) {
            *o += v;
        }
    }
}

/// The derivative `ρ'(z/C²)` of a robust loss's contribution `C²·ρ(z/C²)`
/// with respect to the squared residual `z`.
fn robust_slope(z: f64, loss: Loss, fs2: f64) -> f64 {
    let u = z / fs2;
    match loss {
        Loss::Linear => 1.0,
        Loss::SoftL1 => 1.0 / (1.0 + u).sqrt(),
        Loss::Huber => {
            if u <= 1.0 {
                1.0
            } else {
                1.0 / u.sqrt()
            }
        }
        Loss::Cauchy => 1.0 / (1.0 + u),
        Loss::Arctan => 1.0 / (1.0 + u * u),
    }
}

/// The most intervals the unbinned normalization integral may use (1-D; each
/// level of a nested 2-D or 3-D one uses [`NESTED_INTERVALS`]).
const NORM_INTERVALS: usize = 200;

/// The most intervals each level of a nested 2-D / 3-D integral may use.
const NESTED_INTERVALS: usize = 48;

/// `∫ f` over a region of 1–3 variables (one `(lo, hi)` each): bounded,
/// globally adaptive Gauss–Kronrod, nested for 2-D and 3-D.
fn integrate_region(f: &dyn Fn(&[f64]) -> f64, region: &[(f64, f64)]) -> f64 {
    use oxifit_formula::integrate_adaptive as quad;
    match *region {
        [(a, b)] => quad(|x| f(&[x]), a, b, 1e-10, NORM_INTERVALS).0,
        [(a, b), (c, d)] => {
            quad(
                |x| quad(|y| f(&[x, y]), c, d, 1e-9, NESTED_INTERVALS).0,
                a,
                b,
                1e-9,
                NESTED_INTERVALS,
            )
            .0
        }
        [(a, b), (c, d), (e, g)] => {
            quad(
                |x| {
                    quad(
                        |y| quad(|z| f(&[x, y, z]), e, g, 1e-8, NESTED_INTERVALS).0,
                        c,
                        d,
                        1e-8,
                        NESTED_INTERVALS,
                    )
                    .0
                },
                a,
                b,
                1e-8,
                NESTED_INTERVALS,
            )
            .0
        }
        _ => f64::NAN,
    }
}

/// A cost over the full parameter vector.
pub(crate) type CostFn<'a> = &'a dyn Fn(&[f64]) -> f64;

/// A model's gradient `∂f/∂p` at `x`, written into `out`.
type ModelGrad = dyn Fn(&[f64], &[f64], &mut [f64]) + Send + Sync;

/// A selected point: its variables (the first `dim` used), value and sigma.
type NdPoint = ([f64; 3], f64, f64);

/// A model's integral over a region at given parameters.
type ModelIntegral = dyn Fn(&[f64], &[(f64, f64)]) -> f64 + Send + Sync;

/// A cost's gradient over the full parameter vector, written into `out`.
pub(crate) type GradCostFn<'a> = &'a dyn Fn(&[f64], &mut [f64]);

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
    /// The cost's exact gradient, when the model has an analytic one and the
    /// options allow it ([`FitOptions::gradient`]).
    pub(crate) grad: Option<GradCostFn<'a>>,
    /// The residuals' exact Jacobian, row-major (`n_residuals × n_params`),
    /// when the model has an analytic gradient and the options allow it.
    pub(crate) jacobian: Option<GradCostFn<'a>>,
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
pub(crate) fn run_fit(all: &[PointNd], dim: usize, model: &Model, opts: &FitOptions) -> FitResult {
    // A setter given a name the model does not have, or data of another
    // dimension: not the fit that was asked for (`Model::check` says why).
    if model.check().is_err() || dim != model.dim {
        return failed(model);
    }
    finish_run(
        model,
        with_fit(all, dim, model, opts, |fit| dispatch(fit, opts)),
    )
}

/// Fit `model` to a prepared data term (a simultaneous fit's combined cost).
pub(crate) fn run_terms(model: &Model, opts: &FitOptions, term: &DataTerm) -> FitResult {
    if model.check().is_err() {
        return failed(model);
    }
    finish_run(
        model,
        with_terms(model, opts, term, |fit| dispatch(fit, opts)),
    )
}

/// The result with its evaluation count, or a failure when there were too few
/// measurements to fit.
fn finish_run(model: &Model, run: Option<(FitResult, usize)>) -> FitResult {
    match run {
        Some((mut result, calls)) => {
            result.nfcn = calls;
            result
        }
        None => failed(model),
    }
}

/// Minimize with the chosen backend, then apply any SumW2 correction.
fn dispatch(fit: &Fit, opts: &FitOptions) -> FitResult {
    // A backend may panic on a pathological point (minuit2 0.5's numerical
    // gradient can, from a poor seed). Report that as a failed fit rather than
    // unwinding into the caller. Nothing is shared across the boundary but the
    // read-only model and data, so a caught panic leaves no broken state
    // behind. The default panic hook still prints the message to stderr.
    catch_unwind(AssertUnwindSafe(|| match opts.minimizer {
        Minimizer::Minuit2 => minuit::run(fit, minuit::Local::Migrad),
        Minimizer::Minuit2Combined => minuit::run(fit, minuit::Local::Combined),
        Minimizer::Simplex => minuit::run(fit, minuit::Local::Simplex),
        Minimizer::MultiStart(cfg) => global::multi_start(fit, cfg),
        Minimizer::DifferentialEvolution(cfg) => global::differential_evolution(fit, cfg),
        Minimizer::CmaEs(cfg) => cmaes::cma_es(fit, cfg),
        Minimizer::LevenbergMarquardt => lm::levenberg_marquardt(fit),
        Minimizer::LBfgsB => lbfgsb::lbfgsb(fit),
        #[cfg(feature = "argmin")]
        Minimizer::NelderMead => nelder_mead::minimize(fit),
    }))
    .unwrap_or_else(|_| failed(fit.model))
    .with_sumw2(fit)
}

/// One dataset's cost under its options: the selected points, the method,
/// and (for the unbinned likelihoods) the normalization range and weight sums.
/// Its costs are over the parameters of the model whose `func` it holds.
pub(crate) struct Prepared<'m> {
    func: &'m (dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync),
    grad: Option<&'m ModelGrad>,
    /// The model's integral over a region, when known.
    integral: Option<&'m ModelIntegral>,
    /// The number of variables.
    dim: usize,
    /// `(at, value, sigma)`: the first `dim` entries of `at` are the variables.
    points: Vec<NdPoint>,
    method: FitMethod,
    loss: Loss,
    /// Robust-loss residual scale squared.
    fs2: f64,
    /// The unbinned normalization region, one `(lo, hi)` per variable.
    region: Vec<(f64, f64)>,
    sum_w: f64,
    sum_w2: f64,
}

impl<'m> Prepared<'m> {
    /// Select the points of `all` (of `dim` variables) that enter a fit under
    /// `opts`: those inside `opts.range` (and `range_y`, `range_z`; default:
    /// all), and for Neyman χ² only those with an error.
    pub(crate) fn new(
        all: &[PointNd],
        dim: usize,
        model: &'m Model,
        opts: &FitOptions,
    ) -> Prepared<'m> {
        let func = &*model.func;
        let grad = model.grad.as_deref();
        let dim = dim.clamp(1, 3);
        let ranges = [opts.range, opts.range_y, opts.range_z];
        let drop_empty = matches!(opts.method, FitMethod::Chi2);
        let points: Vec<NdPoint> = all
            .iter()
            .filter_map(|p| {
                for (k, r) in ranges.iter().enumerate().take(dim) {
                    if let Some((lo, hi)) = r
                        && (p.at[k] < *lo || p.at[k] > *hi)
                    {
                        return None;
                    }
                }
                (!drop_empty || p.sigma > 0.0).then_some((p.at, p.value, p.sigma))
            })
            .collect();
        // The unbinned likelihoods normalize over the fit region: the given
        // ranges, else the span of the events in each variable.
        let region: Vec<(f64, f64)> = (0..dim)
            .map(|k| {
                ranges[k].unwrap_or_else(|| {
                    points
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                            (a.min(p.0[k]), b.max(p.0[k]))
                        })
                })
            })
            .collect();
        let (sum_w, sum_w2) = points
            .iter()
            .fold((0.0, 0.0), |(a, b), p| (a + p.1, b + p.1 * p.1));
        let fs = if opts.f_scale > 0.0 {
            opts.f_scale
        } else {
            1.0
        };
        Prepared {
            func,
            grad,
            integral: model.integral.as_deref(),
            dim,
            points,
            method: opts.method,
            loss: opts.loss,
            fs2: fs * fs,
            region,
            sum_w,
            sum_w2,
        }
    }

    /// The number of points (events) entering the fit.
    pub(crate) fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Whether the cost is an unbinned likelihood.
    pub(crate) fn unbinned(&self) -> bool {
        matches!(
            self.method,
            FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood
        )
    }

    /// Whether any event carries a weight other than 1.
    pub(crate) fn weighted(&self) -> bool {
        self.points.iter().any(|p| p.1 != 1.0)
    }

    /// Whether the cost at the minimum is a goodness-of-fit statistic: not for
    /// the unbinned likelihoods, nor a χ² under a robust loss (which the
    /// binned likelihood ignores).
    pub(crate) fn gof(&self) -> bool {
        !self.unbinned() && (self.method == FitMethod::Likelihood || self.loss == Loss::Linear)
    }

    /// How many residuals [`residuals`](Self::residuals) writes: one per
    /// point, or none for the unbinned likelihoods (no sum of squares).
    pub(crate) fn n_residuals(&self) -> usize {
        if self.unbinned() {
            0
        } else {
            self.points.len()
        }
    }

    /// The cost at parameters `p`.
    pub(crate) fn cost(&self, p: &[f64]) -> f64 {
        let func = self.func;
        let dim = self.dim;
        let (loss, fs2) = (self.loss, self.fs2);
        match self.method {
            FitMethod::Chi2 => chunked_sum(&self.points, |&(x, y, e)| {
                let d = (y - func(&x[..dim], p)) / e;
                robust(d * d, loss, fs2)
            }),
            FitMethod::PearsonChi2 => chunked_sum(&self.points, |&(x, y, _)| {
                let f = func(&x[..dim], p).max(1e-300); // expected (model) variance
                robust((y - f) * (y - f) / f, loss, fs2)
            }),
            FitMethod::Likelihood => {
                2.0 * chunked_sum(&self.points, |&(x, y, _)| {
                    let f = func(&x[..dim], p).max(1e-300);
                    if y > 0.0 { f - y + y * (y / f).ln() } else { f }
                })
            }
            FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => {
                self.unbinned_cost(p, 1)
            }
        }
    }

    /// Whether the model has an analytic gradient (and so the cost one).
    pub(crate) fn has_gradient(&self) -> bool {
        self.grad.is_some()
    }

    /// Add the cost's gradient at `p` into `out` (one entry per parameter of
    /// the model), from the model's analytic gradient: per point the cost
    /// term's derivative with respect to `f` times `∂f/∂p`, and for the
    /// unbinned likelihoods the normalization's `∂(∫f)/∂p`. A no-op without
    /// a model gradient.
    pub(crate) fn add_gradient(&self, p: &[f64], out: &mut [f64]) {
        let Some(grad) = self.grad else {
            return;
        };
        let func = self.func;
        let dim = self.dim;
        let np = p.len();
        let (loss, fs2, method) = (self.loss, self.fs2, self.method);
        // d(cost term)/df at a point (0 where the model was clamped).
        let slope = |x: [f64; 3], y: f64, e: f64| -> f64 {
            let raw = func(&x[..dim], p);
            match method {
                FitMethod::Chi2 => {
                    let d = (y - raw) / e;
                    -2.0 * d * robust_slope(d * d, loss, fs2) / e
                }
                FitMethod::PearsonChi2 => {
                    if raw <= 1e-300 {
                        return 0.0;
                    }
                    let h = (y - raw) * (y - raw) / raw;
                    robust_slope(h, loss, fs2) * (raw * raw - y * y) / (raw * raw)
                }
                FitMethod::Likelihood => {
                    if raw <= 1e-300 {
                        0.0
                    } else if y > 0.0 {
                        2.0 * (1.0 - y / raw)
                    } else {
                        2.0
                    }
                }
                FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => {
                    if raw <= 1e-300 { 0.0 } else { -2.0 * y / raw }
                }
            }
        };
        chunked_vec_sum(
            &self.points,
            np,
            |&(x, y, e), acc, g| {
                let c = slope(x, y, e);
                if c != 0.0 {
                    g.iter_mut().for_each(|v| *v = 0.0);
                    grad(&x[..dim], p, g);
                    for (a, gk) in acc.iter_mut().zip(g.iter()) {
                        *a += c * gk;
                    }
                }
            },
            out,
        );
        if self.unbinned() {
            // The normalization N = ∫f and its gradient ∂N/∂pₖ = ∫∂f/∂pₖ.
            let norm = self.norm(p);
            for (k, o) in out.iter_mut().enumerate().take(np) {
                let dn = self.norm_grad(p, k);
                *o += if self.method == FitMethod::ExtendedUnbinnedLikelihood {
                    2.0 * dn
                } else {
                    2.0 * self.sum_w * dn / norm.max(1e-300)
                };
            }
        }
    }

    /// Write the residuals' Jacobian at `p` into `out` (row-major, one row of
    /// `p.len()` per residual): `dr/df · ∂f/∂p`, with each residual's
    /// derivative in `f` taken exactly — including its limit where the
    /// residual vanishes. A no-op without a model gradient or residuals.
    pub(crate) fn residual_jacobian(&self, p: &[f64], out: &mut [f64]) {
        let Some(grad) = self.grad else {
            return;
        };
        if self.n_residuals() == 0 {
            return;
        }
        let func = self.func;
        let dim = self.dim;
        let np = p.len();
        let (loss, fs2, method) = (self.loss, self.fs2, self.method);
        // dr/dd for r = sign(d)·√ρ(d²): ρ'(d²)·|d|/√ρ(d²), → 1 as d → 0.
        let robust_dr = |d: f64| -> f64 {
            let rho = robust(d * d, loss, fs2);
            if d.abs() < 1e-12 || rho <= 0.0 {
                1.0
            } else {
                robust_slope(d * d, loss, fs2) * d.abs() / rho.sqrt()
            }
        };
        let dr_df = |x: [f64; 3], y: f64, e: f64| -> f64 {
            let raw = func(&x[..dim], p);
            match method {
                FitMethod::Chi2 => -robust_dr((y - raw) / e) / e,
                FitMethod::PearsonChi2 => {
                    if raw <= 1e-300 {
                        return 0.0;
                    }
                    let d = (y - raw) / raw.sqrt();
                    -robust_dr(d) * (y + raw) / (2.0 * raw * raw.sqrt())
                }
                FitMethod::Likelihood => {
                    if raw <= 1e-300 {
                        return 0.0;
                    }
                    let t = if y > 0.0 {
                        raw - y + y * (y / raw).ln()
                    } else {
                        raw
                    };
                    // r = sign(f − y)·√(2t), so r² = 2t and dr/df = (dt/df)/r.
                    let r = (2.0 * t).max(0.0).sqrt().copysign(raw - y);
                    let dt = if y > 0.0 { 1.0 - y / raw } else { 1.0 };
                    if r.abs() > 1e-8 {
                        dt / r
                    } else {
                        // t ≈ (f − y)²/(2y), so r ≈ (f − y)/√y.
                        1.0 / y.max(1e-300).sqrt()
                    }
                }
                FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => 0.0,
            }
        };
        let mut g = vec![0.0; np];
        for (row, &(x, y, e)) in out.chunks_mut(np).zip(&self.points) {
            let c = dr_df(x, y, e);
            g.iter_mut().for_each(|v| *v = 0.0);
            grad(&x[..dim], p, &mut g);
            for (o, gk) in row.iter_mut().zip(&g) {
                *o = c * gk;
            }
        }
    }

    /// The model's integral over the fit region at `p`: its known integral
    /// when it has one (products, sums of them), else numerical — bounded,
    /// globally adaptive quadrature, nested for 2-D and 3-D, so a spike-like
    /// trial shape (a width driven toward zero) costs a bounded amount.
    fn norm(&self, p: &[f64]) -> f64 {
        if let Some(i) = self.integral {
            return i(p, &self.region);
        }
        let func = self.func;
        integrate_region(&|c: &[f64]| func(c, p), &self.region)
    }

    /// `∂N/∂pₖ` of the normalization: a central difference of a known
    /// integral, else the integral of the model gradient's `k`-th component.
    fn norm_grad(&self, p: &[f64], k: usize) -> f64 {
        if self.integral.is_some() {
            let h = 1e-6 * p[k].abs().max(1.0);
            let (mut up, mut down) = (p.to_vec(), p.to_vec());
            up[k] += h;
            down[k] -= h;
            return (self.norm(&up) - self.norm(&down)) / (2.0 * h);
        }
        let Some(grad) = self.grad else {
            return 0.0;
        };
        let np = p.len();
        integrate_region(
            &|c: &[f64]| {
                let mut g = vec![0.0; np];
                grad(c, p, &mut g);
                g[k]
            },
            &self.region,
        )
    }

    /// For the SumW2 sandwich: the unbinned cost with squared weights (whose
    /// extended term scales by Σw²/Σw, as in RooFit); otherwise the cost.
    pub(crate) fn cost_w2(&self, p: &[f64]) -> f64 {
        if self.unbinned() {
            self.unbinned_cost(p, 2)
        } else {
            self.cost(p)
        }
    }

    /// `−2 ln L` with each event weighted by `w^power`.
    fn unbinned_cost(&self, p: &[f64], power: i32) -> f64 {
        let func = self.func;
        let dim = self.dim;
        let norm = self.norm(p);
        let w = |y: f64| if power == 1 { y } else { y.powi(power) };
        let logs = chunked_sum(&self.points, |&(x, y, _)| {
            w(y) * func(&x[..dim], p).max(1e-300).ln()
        });
        if self.method == FitMethod::ExtendedUnbinnedLikelihood {
            let scale = if power == 1 || self.sum_w == 0.0 {
                1.0
            } else {
                self.sum_w2 / self.sum_w
            };
            -2.0 * (logs - scale * norm)
        } else {
            let wsum = if power == 1 { self.sum_w } else { self.sum_w2 };
            -2.0 * (logs - wsum * norm.max(1e-300).ln())
        }
    }

    /// The residuals at `p`, whose squares sum to the cost: (y − f)/σ for χ²
    /// (a robust loss as √ρ), (y − f)/√f for Pearson, and the signed deviance
    /// residual √(2·[f − y + y·ln(y/f)]) for the binned likelihood. `out`
    /// holds [`n_residuals`](Self::n_residuals) values.
    pub(crate) fn residuals(&self, p: &[f64], out: &mut [f64]) {
        let func = self.func;
        let dim = self.dim;
        let (loss, fs2, method) = (self.loss, self.fs2, self.method);
        let residual = |&(x, y, e): &NdPoint| match method {
            FitMethod::Chi2 => {
                let d = (y - func(&x[..dim], p)) / e;
                robust(d * d, loss, fs2).max(0.0).sqrt().copysign(d)
            }
            FitMethod::PearsonChi2 => {
                let f = func(&x[..dim], p).max(1e-300);
                let d = (y - f) / f.sqrt();
                robust(d * d, loss, fs2).max(0.0).sqrt().copysign(d)
            }
            FitMethod::Likelihood => {
                let f = func(&x[..dim], p).max(1e-300);
                let t = if y > 0.0 { f - y + y * (y / f).ln() } else { f };
                (2.0 * t).max(0.0).sqrt().copysign(f - y)
            }
            // No residuals: `n_residuals` is 0 for these.
            FitMethod::UnbinnedLikelihood | FitMethod::ExtendedUnbinnedLikelihood => 0.0,
        };
        let n = self.points.len().min(out.len());
        let fill = |(o, pts): (&mut [f64], &[NdPoint])| {
            for (o, pt) in o.iter_mut().zip(pts) {
                *o = residual(pt);
            }
        };
        #[cfg(feature = "rayon")]
        if n > CHUNK {
            use rayon::prelude::*;
            out[..n]
                .par_chunks_mut(CHUNK)
                .zip(self.points[..n].par_chunks(CHUNK))
                .for_each(fill);
            return;
        }
        out[..n]
            .chunks_mut(CHUNK)
            .zip(self.points[..n].chunks(CHUNK))
            .for_each(fill);
    }
}

/// The data part of a fit's cost, over the full parameter vector of the
/// fit's model: one [`Prepared`] dataset, or a simultaneous fit's sum of them.
pub(crate) struct DataTerm<'a> {
    /// The data cost.
    pub(crate) cost: CostFn<'a>,
    /// The data residuals (`n_residuals` of them).
    pub(crate) residuals: &'a dyn Fn(&[f64], &mut [f64]),
    /// How many residuals (0: not a sum of squares).
    pub(crate) n_residuals: usize,
    /// Data points entering `ndf`.
    pub(crate) n_points: usize,
    /// Whether the cost is a goodness-of-fit statistic.
    pub(crate) gof: bool,
    /// The cost with squared weights, when a SumW2 correction applies.
    pub(crate) cost_w2: Option<CostFn<'a>>,
    /// The data cost's gradient (added into `out`), when every model has an
    /// analytic one.
    pub(crate) grad: Option<GradCostFn<'a>>,
    /// The data residuals' Jacobian (row-major, `n_residuals × n_params`),
    /// when every model has an analytic gradient.
    pub(crate) jacobian: Option<GradCostFn<'a>>,
}

/// Build the [`Fit`] for `model` on `all` under `opts` — the selected points,
/// the cost and residuals (with any Gaussian constraints), `ndf` — and run
/// `body` on it. Returns `body`'s result and the number of cost evaluations it
/// made, or `None` when there are too few measurements to determine the free
/// parameters.
pub(crate) fn with_fit<R>(
    all: &[PointNd],
    dim: usize,
    model: &Model,
    opts: &FitOptions,
    body: impl FnOnce(&Fit) -> R,
) -> Option<(R, usize)> {
    let prep = Prepared::new(all, dim, model, opts);
    let cost = |p: &[f64]| prep.cost(p);
    let residuals = |p: &[f64], out: &mut [f64]| prep.residuals(p, out);
    let cost_w2 = |p: &[f64]| prep.cost_w2(p);
    let grad = |p: &[f64], out: &mut [f64]| prep.add_gradient(p, out);
    let jac = |p: &[f64], out: &mut [f64]| prep.residual_jacobian(p, out);
    let term = DataTerm {
        cost: &cost,
        residuals: &residuals,
        n_residuals: prep.n_residuals(),
        n_points: prep.n_points(),
        gof: prep.gof(),
        cost_w2: (prep.unbinned() && prep.weighted() && opts.sumw2)
            .then_some(&cost_w2 as CostFn<'_>),
        grad: prep.has_gradient().then_some(&grad as GradCostFn<'_>),
        jacobian: prep.has_gradient().then_some(&jac as GradCostFn<'_>),
    };
    with_terms(model, opts, &term, body)
}

/// Build the [`Fit`] for `model` from a data term — adding the model's
/// Gaussian constraints and counting cost evaluations — and run `body` on it.
/// `None` when there are too few measurements for the free parameters.
pub(crate) fn with_terms<R>(
    model: &Model,
    opts: &FitOptions,
    term: &DataTerm,
    body: impl FnOnce(&Fit) -> R,
) -> Option<(R, usize)> {
    let np = model.params.len();
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
    let n_meas = term.n_points + penalties.len();

    // Too few measurements to determine the free parameters: report failure
    // rather than a meaningless minimum of a flat cost.
    if np == 0 || n_meas < n_free.max(1) {
        return None;
    }

    let pull = |p: &[f64], &(i, m, sd): &(usize, f64, f64)| (p[i] - m) / sd;
    let penalty = |p: &[f64]| penalties.iter().map(|c| pull(p, c).powi(2)).sum::<f64>();
    let n_data_res = term.n_residuals;

    // Count every cost evaluation, whichever backend makes it (a residual
    // vector counts as one).
    let calls = Cell::new(0_usize);
    let cost = |p: &[f64]| {
        calls.set(calls.get() + 1);
        (term.cost)(p) + penalty(p)
    };
    let residuals = |p: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        (term.residuals)(p, &mut out[..n_data_res]);
        for (o, c) in out[n_data_res..].iter_mut().zip(&penalties) {
            *o = pull(p, c);
        }
    };
    let cost_w2_fn = |p: &[f64]| term.cost_w2.map_or(f64::NAN, |c| c(p)) + penalty(p);
    let grad_fn = |p: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out.iter_mut().for_each(|o| *o = 0.0);
        if let Some(g) = term.grad {
            g(p, out);
        }
        for &(i, m, sd) in &penalties {
            out[i] += 2.0 * (p[i] - m) / (sd * sd);
        }
    };
    // The full Jacobian: the data rows, then one row per constraint
    // (∂/∂pᵢ of (pᵢ − m)/σ is 1/σ).
    let jac_fn = |p: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|o| *o = 0.0);
        if let Some(j) = term.jacobian {
            j(p, &mut out[..n_data_res * np]);
        }
        for (r, &(i, _, sd)) in penalties.iter().enumerate() {
            out[(n_data_res + r) * np + i] = 1.0 / sd;
        }
    };
    let fit = Fit {
        model,
        cost: &cost,
        residuals: &residuals,
        // Least-squares backends need every term as residuals.
        n_residuals: if n_data_res == 0 && term.n_points > 0 {
            0
        } else {
            n_data_res + penalties.len()
        },
        cost_w2: term.cost_w2.is_some().then_some(&cost_w2_fn as CostFn<'_>),
        grad: (term.grad.is_some() && opts.gradient).then_some(&grad_fn as GradCostFn<'_>),
        jacobian: (term.jacobian.is_some() && opts.gradient && n_data_res > 0)
            .then_some(&jac_fn as GradCostFn<'_>),
        ndf: n_meas - n_free,
        opts,
        gof: term.gof,
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

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use crate::data::PointNd;

    /// Every cost's analytic gradient matches a central difference of the cost.
    #[test]
    fn cost_gradients_match_finite_differences() {
        let pts: Vec<PointNd> = (0..60)
            .map(|i| {
                let x = -3.0 + 0.1 * f64::from(i);
                let y = (40.0 * (-0.5 * ((x - 0.3) / 0.8f64).powi(2)).exp() + 5.0).round();
                PointNd::new([x, 0.0, 0.0], y, y.sqrt().max(1.0))
            })
            .collect();
        let model = (Model::gaussian("g") + Model::polynomial("c", 0))
            .with_params(vec![30.0, 0.1, 1.1, 4.0])
            .constrain("c.p0", 5.0, 2.0);
        for method in [
            FitMethod::Chi2,
            FitMethod::PearsonChi2,
            FitMethod::Likelihood,
            FitMethod::UnbinnedLikelihood,
            FitMethod::ExtendedUnbinnedLikelihood,
        ] {
            for loss in [Loss::Linear, Loss::Cauchy] {
                let opts = FitOptions::new().method(method).loss(loss).range(-3.0, 3.0);
                with_fit(&pts, 1, &model, &opts, |fit| {
                    let grad = fit.grad.expect("the model has a gradient");
                    let p = model.params.clone();
                    let mut g = vec![0.0; p.len()];
                    grad(&p, &mut g);
                    for k in 0..p.len() {
                        let h = 1e-5 * p[k].abs().max(1.0);
                        let (mut up, mut down) = (p.clone(), p.clone());
                        up[k] += h;
                        down[k] -= h;
                        let num = ((fit.cost)(&up) - (fit.cost)(&down)) / (2.0 * h);
                        let rel = (g[k] - num).abs() / num.abs().max(1.0);
                        assert!(rel < 1e-5, "{method:?}/{loss:?} p{k}: {} vs {num}", g[k]);
                    }
                })
                .unwrap();
            }
        }
    }
}

#[cfg(test)]
mod jacobian_tests {
    use super::*;
    use crate::data::PointNd;

    /// The residuals' exact Jacobian matches a central difference.
    #[test]
    fn residual_jacobians_match_finite_differences() {
        let pts: Vec<PointNd> = (0..40)
            .map(|i| {
                let x = -2.0 + 0.1 * f64::from(i);
                let y = (30.0 * (-0.5 * ((x - 0.2) / 0.7f64).powi(2)).exp() + 4.0).round();
                PointNd::new([x, 0.0, 0.0], y, y.sqrt().max(1.0))
            })
            .collect();
        let model = (Model::gaussian("g") + Model::polynomial("c", 0))
            .with_params(vec![25.0, 0.1, 0.9, 3.0])
            .constrain("g.sigma", 0.7, 0.1);
        for method in [
            FitMethod::Chi2,
            FitMethod::PearsonChi2,
            FitMethod::Likelihood,
        ] {
            for loss in [Loss::Linear, Loss::SoftL1] {
                let opts = FitOptions::new().method(method).loss(loss);
                with_fit(&pts, 1, &model, &opts, |fit| {
                    let jac = fit.jacobian.expect("an analytic Jacobian");
                    let (n, p) = (fit.n_residuals, model.params.clone());
                    let np = p.len();
                    let mut j = vec![0.0; n * np];
                    jac(&p, &mut j);
                    for k in 0..np {
                        let h = 1e-6 * p[k].abs().max(1.0);
                        let (mut up, mut down) = (p.clone(), p.clone());
                        up[k] += h;
                        down[k] -= h;
                        let (mut ru, mut rd) = (vec![0.0; n], vec![0.0; n]);
                        (fit.residuals)(&up, &mut ru);
                        (fit.residuals)(&down, &mut rd);
                        for r in 0..n {
                            let num = (ru[r] - rd[r]) / (2.0 * h);
                            let got = j[r * np + k];
                            assert!(
                                (got - num).abs() < 1e-5 * num.abs().max(1.0),
                                "{method:?}/{loss:?} row {r} p{k}: {got} vs {num}"
                            );
                        }
                    }
                })
                .unwrap();
            }
        }
    }
}
