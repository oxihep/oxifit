//! The Minuit2 backends (MIGRAD, MIGRAD-then-SIMPLEX, SIMPLEX) and
//! [`finish`] — HESSE, MINOS, and the [`FitResult`] — which every
//! Minuit-polished backend, the global searches included, ends with.

use minuit2::{FunctionMinimum, MnHesse, MnMigrad, MnMinimize, MnMinos, MnSimplex};

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

/// Run a local Minuit2 minimizer from `start` (no HESSE, no MINOS).
pub(crate) fn local(fit: &Fit, start: &[f64], how: Local) -> FunctionMinimum {
    let cost = &fit.cost;
    match how {
        Local::Migrad => configure(MnMigrad::new(), fit, start).minimize(cost),
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
    // A Hessian that is not positive-definite marks the minimum invalid.
    if opts.hesse && min.is_valid() {
        min = MnHesse::new()
            .with_strategy(opts.strategy)
            .calculate(cost, &min);
    }

    // Read results by index (external/user space) — robust against duplicate
    // parameter names.
    let params = min.params();
    let errors = min.user_state().errors();

    // Covariance of the free parameters (row-major), when Minuit2 has one.
    let covariance = min.user_state().covariance().map(|cov| {
        let m = cov.nrow();
        (0..m)
            .map(|i| (0..m).map(|j| cov.get(i, j)).collect())
            .collect()
    });

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
        min.is_valid(),
    );
    r.minos = minos;
    r.covariance = covariance;
    r.fval = fval;
    r.edm = min.edm(); // `nfcn` is counted by the caller
    r
}
