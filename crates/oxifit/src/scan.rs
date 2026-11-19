//! Likelihood scans beyond the parabolic errors: a [`Profile`] of the cost in
//! one parameter and a 2-D [`Contour`] of two, both refitting the other
//! parameters at each point, with [`FitError`] for what can go wrong.

use crate::data::PointNd;
use crate::engine;
use crate::model::Model;
use crate::result::{FitOptions, FitResult, Minimizer};

/// What a scan refits: a dataset, or a user's cost. `run` fits under a model's
/// parameter settings; `value` is the cost at the model's seeds (constraints
/// included), for a point with nothing left to fit.
pub(crate) trait Refit: Sync {
    fn run(&self, params: &Model, opts: &FitOptions) -> FitResult;
    fn value(&self, params: &Model, opts: &FitOptions) -> f64;
}

/// A dataset of `dim` variables.
pub(crate) struct DataRefit<'a> {
    pub(crate) data: &'a [PointNd],
    pub(crate) dim: usize,
}

impl Refit for DataRefit<'_> {
    fn run(&self, params: &Model, opts: &FitOptions) -> FitResult {
        engine::run_fit(self.data, self.dim, params, opts)
    }
    fn value(&self, params: &Model, opts: &FitOptions) -> f64 {
        engine::with_fit(self.data, self.dim, params, opts, |fit| {
            (fit.cost)(&fit.seed())
        })
        .map_or(f64::NAN, |(f, _)| f)
    }
}

/// Why a [`profile`](crate::FitExt::profile) or
/// [`contour`](crate::FitExt::contour) could not be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitError {
    /// No parameter of the model has this name.
    UnknownParameter(String),
    /// The parameter is fixed, so there is nothing to scan.
    FixedParameter(String),
    /// The fit at the minimum was not valid, or the scan itself failed (e.g.
    /// the cost never rose `Δ` above its minimum along some direction).
    FitFailed,
    /// The data and the model (or a model's parts) have different numbers of
    /// variables.
    DimensionMismatch {
        /// The dimension required.
        expected: usize,
        /// The dimension found.
        found: usize,
    },
    /// The cost cannot make the Asimov dataset or pseudo-datasets a
    /// hypothesis test needs (see [`Cost::asimov`](crate::Cost::asimov)) — a
    /// closure cost, or a sum with one.
    NotGenerative,
    /// An upper limit was not bracketed: the CLs value does not cross the
    /// requested level inside the scanned range — widen it.
    NoCrossing,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::UnknownParameter(p) => write!(f, "no parameter named {p:?}"),
            FitError::FixedParameter(p) => write!(f, "parameter {p:?} is fixed"),
            FitError::FitFailed => write!(f, "the fit did not converge to a valid minimum"),
            FitError::DimensionMismatch { expected, found } => {
                write!(f, "expected {expected} variable(s), found {found}")
            }
            FitError::NotGenerative => {
                write!(f, "the cost cannot make Asimov or pseudo-datasets")
            }
            FitError::NoCrossing => write!(f, "CLs does not cross the level in the range"),
        }
    }
}

impl std::error::Error for FitError {}

/// The cost minimized over the other parameters, as one parameter is stepped
/// across a range — a profile likelihood (or profile χ²). Its width at
/// `Δ = 1` is the 1σ interval, valid even where the cost is far from
/// parabolic (compare MINOS); `Δ = 4` gives 2σ.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Profile {
    /// The scanned parameter.
    pub param: String,
    /// Its best-fit value.
    pub best: f64,
    /// The cost at the global minimum.
    pub fval_min: f64,
    /// The scanned values.
    pub values: Vec<f64>,
    /// `Δcost` at each value: the refitted cost minus
    /// [`fval_min`](Self::fval_min) (`NaN` where the refit failed).
    pub delta: Vec<f64>,
}

impl Profile {
    /// The interval where `Δcost ≤ level` around the minimum, from linear
    /// interpolation between the scanned points: `level = 1` gives the 1σ
    /// (68.3 %) interval of one parameter, `4` the 2σ one. `None` when a side
    /// never crosses `level` inside the scanned range — widen the range.
    #[must_use]
    pub fn interval(&self, level: f64) -> Option<(f64, f64)> {
        let ok: Vec<(f64, f64)> = self
            .values
            .iter()
            .zip(&self.delta)
            .filter(|(_, d)| d.is_finite())
            .map(|(&v, &d)| (v, d))
            .collect();
        let imin = (0..ok.len()).min_by(|&a, &b| ok[a].1.total_cmp(&ok[b].1))?;
        if ok[imin].1 > level {
            return None;
        }
        let cross = |a: (f64, f64), b: (f64, f64)| a.0 + (level - a.1) * (b.0 - a.0) / (b.1 - a.1);
        let lo = (0..imin)
            .rev()
            .find(|&k| ok[k].1 > level)
            .map(|k| cross(ok[k + 1], ok[k]))?;
        let hi = (imin + 1..ok.len())
            .find(|&k| ok[k].1 > level)
            .map(|k| cross(ok[k - 1], ok[k]))?;
        Some((lo, hi))
    }
}

/// A 2-D confidence contour: the points where the cost, minimized over the
/// other parameters, is `Δ` above its minimum. Each point is found along a ray
/// from the minimum (rays spread evenly around the covariance ellipse), so
/// the region is assumed star-shaped around the minimum — true of any
/// contour close to an ellipse.
///
/// For a joint region of the two parameters choose `Δ` by confidence level —
/// `2.30` (68.3 %), `4.61` (90 %), `5.99` (95 %), `9.21` (99 %); `Δ = 1`
/// gives the contour whose projections are the 1σ errors of each parameter
/// alone (39.3 % joint).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Contour {
    /// The parameter on the x axis.
    pub x_param: String,
    /// The parameter on the y axis.
    pub y_param: String,
    /// The best-fit `(x, y)`.
    pub best: (f64, f64),
    /// The `Δcost` the contour is drawn at.
    pub delta: f64,
    /// The contour, as `(x, y)` points going round it (not repeating the
    /// first point at the end).
    pub points: Vec<(f64, f64)>,
}

/// Data of `dim` variables fitted with a model of another dimension.
pub(crate) fn check_dim(model: &Model, dim: usize) -> Result<(), FitError> {
    if model.dim == dim {
        Ok(())
    } else {
        Err(FitError::DimensionMismatch {
            expected: model.dim,
            found: dim,
        })
    }
}

/// The index of free parameter `name`, or the reason it cannot be scanned
/// (including a setter's unknown name: see [`Model::check`]).
pub(crate) fn free_index(model: &Model, name: &str) -> Result<usize, FitError> {
    model.check()?;
    let i = model
        .param_names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| FitError::UnknownParameter(name.to_string()))?;
    if model.constraints.get(i).is_some_and(|c| c.fixed) {
        return Err(FitError::FixedParameter(name.to_string()));
    }
    Ok(i)
}

/// Local refits for scan points: from the best fit, the cost alone (no HESSE
/// or MINOS), with the combined minimizer — points far out in a profile are
/// where MIGRAD alone fails.
pub(crate) fn local_opts(opts: &FitOptions) -> FitOptions {
    FitOptions {
        minimizer: Minimizer::Minuit2Combined,
        hesse: false,
        minos: false,
        ..*opts
    }
}

/// `Δcost` with the parameters in `at` fixed at their values and the other
/// free parameters refitted from `best`; `NaN` if the refit fails.
pub(crate) fn delta_at(
    refit: &dyn Refit,
    model: &Model,
    local: &FitOptions,
    best: &FitResult,
    at: &[(usize, f64)],
) -> f64 {
    let mut params = best.params.clone();
    for &(i, v) in at {
        params[i] = v;
    }
    let mut fixed = model.clone().with_params(params);
    for &(i, _) in at {
        fixed.constraints[i].fixed = true;
    }
    let others_free = fixed.constraints.iter().any(|c| !c.fixed);
    let fval = if others_free {
        let r = refit.run(&fixed, local);
        if r.valid { r.fval } else { f64::NAN }
    } else {
        // Nothing left to minimize: the cost at this point.
        refit.value(&fixed, local)
    };
    fval - best.fval
}

/// See [`FitExt::profile`](crate::FitExt::profile).
pub(crate) fn profile(
    refit: &dyn Refit,
    model: &Model,
    opts: &FitOptions,
    param: &str,
    (lo, hi): (f64, f64),
    n: usize,
) -> Result<Profile, FitError> {
    let i = free_index(model, param)?;
    let best = refit.run(model, opts);
    if !best.valid {
        return Err(FitError::FitFailed);
    }
    let local = local_opts(opts);
    let n = n.max(2);
    let values: Vec<f64> = (0..n)
        .map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64)
        .collect();
    let at = |&v: &f64| delta_at(refit, model, &local, &best, &[(i, v)]);
    #[cfg(feature = "rayon")]
    let delta = {
        use rayon::prelude::*;
        values.par_iter().map(at).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let delta = values.iter().map(at).collect();
    Ok(Profile {
        param: param.to_string(),
        best: best.params[i],
        fval_min: best.fval,
        values,
        delta,
    })
}

/// See [`FitExt::contour`](crate::FitExt::contour).
pub(crate) fn contour(
    refit: &dyn Refit,
    model: &Model,
    opts: &FitOptions,
    (x, y): (&str, &str),
    delta: f64,
    npoints: usize,
) -> Result<Contour, FitError> {
    let ix = free_index(model, x)?;
    let iy = free_index(model, y)?;
    let best = refit.run(model, opts);
    if !best.valid || delta.is_nan() || delta <= 0.0 {
        return Err(FitError::FitFailed);
    }
    let (bx, by) = (best.params[ix], best.params[iy]);

    // The 2×2 covariance of (x, y) and its Cholesky factor L: the ray in
    // direction θ is (bx, by) + r·L·(cos θ, sin θ), on which a quadratic cost
    // rises as r², so r = √Δ is the first guess. Without a usable covariance,
    // the axes scaled by the errors.
    let free: Vec<usize> = (0..model.params.len())
        .filter(|&i| !model.constraints.get(i).is_some_and(|c| c.fixed))
        .collect();
    let pos = |i: usize| free.iter().position(|&f| f == i);
    let (sx, sy) = (best.errors[ix], best.errors[iy]);
    let cxy = match (&best.covariance, pos(ix), pos(iy)) {
        (Some(c), Some(a), Some(b)) => c[a][b],
        _ => 0.0,
    };
    let l11 = sx;
    let l21 = if sx > 0.0 { cxy / sx } else { 0.0 };
    let l22 = (sy * sy - l21 * l21).max(0.0).sqrt();
    if !(l11 > 0.0 && l22 > 0.0) {
        return Err(FitError::FitFailed);
    }

    let local = local_opts(opts);
    let n = npoints.max(4);
    // One root-find per ray; the rays are independent.
    let ray = |k: usize| -> Option<(f64, f64)> {
        let theta = std::f64::consts::TAU * k as f64 / n as f64;
        let (c, s) = (theta.cos(), theta.sin());
        let dir = (l11 * c, l21 * c + l22 * s);
        let at = |r: f64| (bx + r * dir.0, by + r * dir.1);
        let g = |r: f64| {
            let (px, py) = at(r);
            delta_at(refit, model, &local, &best, &[(ix, px), (iy, py)]) - delta
        };
        ray_root(g, delta.sqrt()).map(at)
    };
    #[cfg(feature = "rayon")]
    let points: Option<Vec<(f64, f64)>> = {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(ray).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let points: Option<Vec<(f64, f64)>> = (0..n).map(ray).collect();
    let points = points.ok_or(FitError::FitFailed)?;
    Ok(Contour {
        x_param: x.to_string(),
        y_param: y.to_string(),
        best: (bx, by),
        delta,
        points,
    })
}

/// The radius `r > 0` where `g` (negative at 0, rising) crosses zero: bracket
/// from the guess outwards, then Illinois regula falsi. `None` if `g` turns
/// `NaN` (a failed refit) or never reaches zero.
fn ray_root(g: impl Fn(f64) -> f64, guess: f64) -> Option<f64> {
    let (mut a, mut ga) = (0.0, g(0.0));
    if ga.is_nan() || ga >= 0.0 {
        return None;
    }
    let (mut b, mut gb) = (guess, g(guess));
    let mut grown = 0;
    while gb < 0.0 {
        (a, ga) = (b, gb);
        b *= 1.6;
        gb = g(b);
        grown += 1;
        if grown > 40 {
            return None;
        }
    }
    if gb.is_nan() {
        return None;
    }
    let tol = 1e-5 * guess * guess + 1e-12;
    let mut last = 0; // which end moved last: −1 = a, +1 = b
    for _ in 0..60 {
        let r = (a * gb - b * ga) / (gb - ga);
        let gr = g(r);
        if gr.is_nan() {
            return None;
        }
        if gr.abs() <= tol || (b - a).abs() <= 1e-10 * b {
            return Some(r);
        }
        if gr < 0.0 {
            (a, ga) = (r, gr);
            if last == -1 {
                gb /= 2.0; // Illinois: halve the stale end's value
            }
            last = -1;
        } else {
            (b, gb) = (r, gr);
            if last == 1 {
                ga /= 2.0;
            }
            last = 1;
        }
    }
    Some((a * gb - b * ga) / (gb - ga))
}
