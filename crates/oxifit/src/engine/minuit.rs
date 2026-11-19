//! The Minuit2 backends (MIGRAD, MIGRAD-then-SIMPLEX, SIMPLEX) and
//! [`finish`] — HESSE, MINOS, and the [`FitResult`] — which every
//! Minuit-polished backend, the global searches included, ends with.

use minuit2::{
    FCN, FCNGradient, FunctionMinimum, MnHesse, MnMigrad, MnMinimize, MnMinos, MnSimplex,
};

use super::Fit;
use crate::result::FitResult;

/// A local Minuit2 minimizer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Local {
    /// MIGRAD, the variable-metric minimizer.
    Migrad,
    /// MIGRAD, falling back to SIMPLEX (then MIGRAD again) when it fails —
    /// ROOT's `"Minimize"`.
    Combined,
    /// The SIMPLEX method alone.
    Simplex,
}

/// The parameter-setup methods the Minuit2 front ends share; implemented for
/// each by delegating to its inherent method of the same name.
trait Builder: Sized {
    fn add(self, name: &str, value: f64, step: f64) -> Self;
    fn add_limited(self, name: &str, value: f64, step: f64, lo: f64, hi: f64) -> Self;
    fn add_lower_limited(self, name: &str, value: f64, step: f64, lo: f64) -> Self;
    fn add_upper_limited(self, name: &str, value: f64, step: f64, hi: f64) -> Self;
    fn add_const(self, name: &str, value: f64) -> Self;
    fn with_strategy(self, level: u32) -> Self;
    fn tolerance(self, tol: f64) -> Self;
    fn max_fcn(self, max: usize) -> Self;
}

macro_rules! builder {
    ($($t:ty),*) => {$(
        impl Builder for $t {
            fn add(self, name: &str, value: f64, step: f64) -> Self {
                <$t>::add(self, name, value, step)
            }
            fn add_limited(self, name: &str, value: f64, step: f64, lo: f64, hi: f64) -> Self {
                <$t>::add_limited(self, name, value, step, lo, hi)
            }
            fn add_lower_limited(self, name: &str, value: f64, step: f64, lo: f64) -> Self {
                <$t>::add_lower_limited(self, name, value, step, lo)
            }
            fn add_upper_limited(self, name: &str, value: f64, step: f64, hi: f64) -> Self {
                <$t>::add_upper_limited(self, name, value, step, hi)
            }
            fn add_const(self, name: &str, value: f64) -> Self {
                <$t>::add_const(self, name, value)
            }
            fn with_strategy(self, level: u32) -> Self {
                <$t>::with_strategy(self, level)
            }
            fn tolerance(self, tol: f64) -> Self {
                <$t>::tolerance(self, tol)
            }
            fn max_fcn(self, max: usize) -> Self {
                <$t>::max_fcn(self, max)
            }
        }
    )*};
}
builder!(MnMigrad, MnMinimize, MnSimplex);

/// Declare the model's parameters to a Minuit2 front end, starting from
/// `start` (a full parameter vector), with the fit's strategy, tolerance and
/// call limit.
fn configure<B: Builder>(b: B, fit: &Fit, start: &[f64]) -> B {
    let opts = fit.opts;
    let mut b = b.with_strategy(opts.strategy).tolerance(opts.tolerance);
    if let Some(max) = opts.max_fcn {
        b = b.max_fcn(max);
    }
    for (i, name) in fit.model.param_names.iter().enumerate() {
        let c = fit.constraint(i);
        let v = start[i];
        let step = fit.step(i, v);
        b = if c.fixed {
            b.add_const(name, v)
        } else {
            match (c.lower, c.upper) {
                (Some(lo), Some(hi)) => b.add_limited(name, v, step, lo, hi),
                (Some(lo), None) => b.add_lower_limited(name, v, step, lo),
                (None, Some(hi)) => b.add_upper_limited(name, v, step, hi),
                (None, None) => b.add(name, v, step),
            }
        };
    }
    b
}

/// The cost with its analytic gradient, for MIGRAD's `minimize_grad`.
struct WithGradient<'f, 'a> {
    fit: &'f Fit<'a>,
    grad: super::GradCostFn<'a>,
}

impl FCN for WithGradient<'_, '_> {
    fn value(&self, par: &[f64]) -> f64 {
        (self.fit.cost)(par)
    }
}

impl FCNGradient for WithGradient<'_, '_> {
    fn gradient(&self, par: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; par.len()];
        (self.grad)(par, &mut g);
        g
    }
}

/// Run a local Minuit2 minimizer from `start` (no HESSE, no MINOS). MIGRAD
/// uses the cost's analytic gradient when the fit has one.
pub(crate) fn local(fit: &Fit, start: &[f64], how: Local) -> FunctionMinimum {
    let cost = &fit.cost;
    match how {
        Local::Migrad => {
            let migrad = configure(MnMigrad::new(), fit, start);
            match fit.grad {
                // minuit2 0.5's analytic-gradient MIGRAD seeds its metric from
                // the step sizes alone and can stall far from the minimum,
                // where the finite-difference MIGRAD (which measures the
                // curvature first) converges: fall back to it rather than
                // let a gradient make a fit worse.
                Some(grad) => {
                    let min = migrad.minimize_grad(&WithGradient { fit, grad });
                    if min.is_valid() {
                        min
                    } else {
                        migrad.minimize(cost)
                    }
                }
                None => migrad.minimize(cost),
            }
        }
        Local::Combined => configure(MnMinimize::new(), fit, start).minimize(cost),
        Local::Simplex => configure(MnSimplex::new(), fit, start).minimize(cost),
    }
}

/// Minimize from the model's seed with a local minimizer, then [`finish`].
pub(crate) fn run(fit: &Fit, how: Local) -> FitResult {
    finish(fit, local(fit, &fit.seed(), how))
}

/// Turn a minimum into a [`FitResult`]: run HESSE (unless turned off) for the
/// full second-derivative matrix at the minimum — replacing MIGRAD's running
/// estimate, or giving SIMPLEX a covariance at all — then MINOS on request.
pub(crate) fn finish(fit: &Fit, mut min: FunctionMinimum) -> FitResult {
    let cost = &fit.cost;
    let opts = fit.opts;
    // With an analytic gradient the Hessian is a central difference of the
    // gradient (2n gradient calls instead of HESSE's ~2n² cost calls);
    // otherwise HESSE. A Hessian that is not positive-definite marks the
    // minimum invalid either way.
    let mut from_gradient = None;
    if opts.hesse && min.is_valid() {
        match fit.grad {
            Some(grad) => from_gradient = Some(gradient_hessian(fit, grad, &min)),
            None => {
                min = MnHesse::new()
                    .with_strategy(opts.strategy)
                    .calculate(cost, &min);
            }
        }
    }

    // Read results by index (external/user space) — robust against duplicate
    // parameter names.
    let params = min.params();
    let mut errors = min.user_state().errors();

    // Covariance of the free parameters (row-major), when Minuit2 has one.
    let mut covariance: Option<Vec<Vec<f64>>> = min.user_state().covariance().map(|cov| {
        let m = cov.nrow();
        (0..m)
            .map(|i| (0..m).map(|j| cov.get(i, j)).collect())
            .collect()
    });
    let mut valid = min.is_valid();
    if let Some(hessian) = from_gradient {
        match hessian {
            Some(cov) => {
                for (k, i) in fit.free().into_iter().enumerate() {
                    errors[i] = cov[k][k].sqrt();
                }
                covariance = Some(cov);
            }
            None => valid = false,
        }
    }

    // Asymmetric MINOS errors per parameter, on request and only at a valid
    // minimum. A fixed parameter is pinned, so its error is (0, 0); a free one
    // gets a likelihood scan (`lower_error` is ≤ 0, `upper_error` is ≥ 0).
    let minos = (opts.minos && min.is_valid()).then(|| {
        let scan = MnMinos::new(cost, &min).with_strategy(opts.strategy);
        (0..params.len())
            .map(|par| {
                if fit.constraint(par).fixed {
                    (0.0, 0.0)
                } else {
                    let e = scan.minos(par);
                    (e.lower_error(), e.upper_error())
                }
            })
            .collect()
    });

    let fval = min.fval();
    let mut r = FitResult::new(
        params,
        errors,
        if fit.gof { fval } else { f64::NAN },
        fit.ndf,
        valid,
    );
    r.minos = minos;
    r.covariance = covariance;
    r.fval = fval;
    r.edm = min.edm(); // `nfcn` is counted by the caller
    r
}

/// The free-parameter covariance `2·H⁻¹` from a central difference of the
/// analytic gradient at the minimum (steps of a thousandth of MIGRAD's error
/// estimate), symmetrized; `None` when `H` is not positive-definite.
fn gradient_hessian(
    fit: &Fit,
    grad: super::GradCostFn,
    min: &FunctionMinimum,
) -> Option<Vec<Vec<f64>>> {
    let free = fit.free();
    let n = free.len();
    let p = min.params();
    let errs = min.user_state().errors();
    let mut h = vec![vec![0.0; n]; n];
    let mut g_up = vec![0.0; p.len()];
    let mut g_down = vec![0.0; p.len()];
    for (a, &i) in free.iter().enumerate() {
        let e = errs[i];
        let step = if e.is_finite() && e > 0.0 {
            1e-3 * e
        } else {
            1e-6 * p[i].abs().max(1.0)
        };
        let (mut up, mut down) = (p.clone(), p.clone());
        up[i] += step;
        down[i] -= step;
        grad(&up, &mut g_up);
        grad(&down, &mut g_down);
        for (b, &j) in free.iter().enumerate() {
            h[a][b] = (g_up[j] - g_down[j]) / (2.0 * step);
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let v = 0.5 * (h[a][b] + h[b][a]);
            h[a][b] = v;
            h[b][a] = v;
        }
    }
    // Positive-definite ⇔ a Cholesky factorization exists.
    let mut l = vec![vec![0.0; n]; n];
    for a in 0..n {
        for b in 0..=a {
            let s: f64 = (0..b).map(|k| l[a][k] * l[b][k]).sum();
            if a == b {
                let d = h[a][a] - s;
                if d.is_nan() || d <= 0.0 {
                    return None;
                }
                l[a][a] = d.sqrt();
            } else {
                l[a][b] = (h[a][b] - s) / l[b][b];
            }
        }
    }
    // cov = 2·H⁻¹, column by column from L·Lᵀ·x = eₖ.
    let columns: Vec<Vec<f64>> = (0..n)
        .map(|c| {
            let mut y = vec![0.0; n];
            for a in 0..n {
                let rhs = if a == c { 1.0 } else { 0.0 };
                y[a] = (rhs - (0..a).map(|k| l[a][k] * y[k]).sum::<f64>()) / l[a][a];
            }
            let mut x = vec![0.0; n];
            for a in (0..n).rev() {
                x[a] = (y[a] - (a + 1..n).map(|k| l[k][a] * x[k]).sum::<f64>()) / l[a][a];
            }
            x
        })
        .collect();
    // H is symmetric, so is its inverse: column c is row c.
    Some(
        columns
            .into_iter()
            .map(|col| col.into_iter().map(|v| 2.0 * v).collect())
            .collect(),
    )
}
