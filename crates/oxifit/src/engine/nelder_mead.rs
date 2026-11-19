//! The gradient-free Nelder–Mead simplex from the `argmin` crate — an
//! independent cross-check of Minuit2 (`argmin` feature).

use super::Fit;
use super::linalg::{hessian, invert};
use super::transform::Transform;
use crate::result::FitResult;

/// Scatter a free-parameter sub-vector back into the full parameter vector
/// (fixed parameters keep their initial value).
fn scatter(sub: &[f64], free: &[usize], init_full: &[f64]) -> Vec<f64> {
    let mut full = init_full.to_vec();
    for (k, &i) in free.iter().enumerate() {
        full[i] = sub[k];
    }
    full
}

/// The most simplex iterations a Nelder–Mead fit takes before giving up.
const NELDER_MEAD_MAX_ITERS: u64 = 10_000;

/// Minimize with the gradient-free Nelder–Mead simplex from the `argmin` crate.
/// Optimizes the free parameters only (fixed ones are pinned) in Minuit's
/// internal coordinates, so per-parameter limits hold exactly. Parameter
/// errors and the covariance come from a numerical Hessian of the cost in the
/// model's own coordinates at the minimum (`cov = 2·H⁻¹`); MINOS is not
/// provided.
pub(crate) fn minimize(fit: &Fit) -> FitResult {
    minimize_capped(fit, NELDER_MEAD_MAX_ITERS)
}

/// Nelder–Mead with at most `max_iters` iterations. The result is valid only
/// when the simplex converged (its spread fell below the tolerance) to a
/// finite χ²: a run stopped by the cap is reported, but not as valid, as
/// Minuit's `is_valid()` would not be.
fn minimize_capped(fit: &Fit, max_iters: u64) -> FitResult {
    let model = fit.model;
    let cost = fit.cost;
    use argmin::core::{CostFunction, Error, Executor, State, TerminationReason};
    use argmin::solver::neldermead::NelderMead;

    let free = fit.free();
    let init_full = fit.seed();

    // The simplex moves in Minuit's internal coordinates: a limited parameter
    // is a sine (two limits) or square-root (one) of an unbounded internal
    // value, so every vertex maps inside the limits.
    let transforms: Vec<Transform> = free
        .iter()
        .map(|&i| Transform::of(&fit.constraint(i)))
        .collect();
    let to_ext = |int: &[f64]| -> Vec<f64> {
        int.iter()
            .zip(&transforms)
            .map(|(&v, t)| t.to_ext(v))
            .collect()
    };
    let internal = |int: &[f64]| -> f64 { cost(&scatter(&to_ext(int), &free, &init_full)) };

    /// Adapter wrapping our `Fn(&[f64]) -> f64` as an argmin problem.
    struct Prob<F>(F);
    impl<F: Fn(&[f64]) -> f64> CostFunction for Prob<F> {
        type Param = Vec<f64>;
        type Output = f64;
        fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
            Ok((self.0)(p))
        }
    }

    // Initial simplex: the seed point plus one vertex per free parameter,
    // perturbed by its step (the hint, else 10 % of the value with a floor),
    // converted to an internal step.
    let seed: Vec<f64> = free
        .iter()
        .zip(&transforms)
        .map(|(&i, t)| t.to_int(init_full[i]))
        .collect();
    let mut simplex = vec![seed.clone()];
    for (k, &i) in free.iter().enumerate() {
        let step = transforms[k].int_step(seed[k], fit.step(i, init_full[i]));
        let mut v = seed.clone();
        v[k] += step;
        simplex.push(v);
    }

    let Ok(solver) = NelderMead::new(simplex).with_sd_tolerance(1e-10) else {
        return super::failed(model);
    };
    let Ok(res) = Executor::new(Prob(&internal), solver)
        .configure(|s| s.max_iters(max_iters))
        .run()
    else {
        return super::failed(model);
    };
    let Some(best) = res.state().best_param.as_ref().cloned() else {
        return super::failed(model);
    };
    let converged = matches!(
        res.state().get_termination_reason(),
        Some(TerminationReason::SolverConverged)
    );

    // Errors from the Hessian in the external (model) coordinates.
    let best = to_ext(&best);
    let params = scatter(&best, &free, &init_full);
    let chi2 = cost(&params);
    let (errors, covariance) = hessian_errors(&best, &free, &init_full, &cost);

    let mut r = FitResult::new(
        params,
        errors,
        if fit.gof { chi2 } else { f64::NAN },
        fit.ndf,
        converged && chi2.is_finite(),
    );
    r.covariance = covariance;
    r.fval = chi2;
    r
}

/// Parabolic errors + free-parameter covariance from a central-difference
/// Hessian of the cost at the minimum: `cov = 2·H⁻¹` (the `Δcost = 1` contour is
/// 1σ for a χ² / −2 ln L cost). Errors are full-length (fixed parameters → 0);
/// the covariance is the `nf×nf` free block (matching the Minuit2 backend).
fn hessian_errors(
    sub: &[f64],
    free: &[usize],
    init_full: &[f64],
    cost: &impl Fn(&[f64]) -> f64,
) -> (Vec<f64>, Option<Vec<Vec<f64>>>) {
    let np = init_full.len();
    let f = |s: &[f64]| cost(&scatter(s, free, init_full));
    let h: Vec<f64> = sub.iter().map(|&v| (v.abs() * 1e-4).max(1e-6)).collect();
    let hess = hessian(&f, sub, &h);
    let covariance = invert(&hess).map(|inv| {
        inv.iter()
            .map(|row| row.iter().map(|&x| 2.0 * x).collect())
            .collect::<Vec<Vec<f64>>>()
    });

    let mut errors = vec![0.0; np]; // fixed parameters report 0, as Minuit2 does
    match &covariance {
        Some(cov) => {
            for (k, &i) in free.iter().enumerate() {
                errors[i] = if cov[k][k] > 0.0 {
                    cov[k][k].sqrt()
                } else {
                    f64::NAN
                };
            }
        }
        None => {
            for &i in free {
                errors[i] = f64::NAN;
            }
        }
    }
    (errors, covariance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::result::FitOptions;

    /// χ² of a line through (x, 3 + 2x), x = 0..5, unit errors.
    fn line_chi2(p: &[f64]) -> f64 {
        (0..6)
            .map(|i| {
                let x = f64::from(i);
                (p[0] + p[1] * x - (3.0 + 2.0 * x)).powi(2)
            })
            .sum()
    }

    #[test]
    fn valid_only_when_the_simplex_converged() {
        let model = Model::polynomial("l", 1).with_params(vec![0.0, 0.0]);
        let opts = FitOptions::new();
        let fit = Fit {
            model: &model,
            cost: &line_chi2,
            residuals: &|_, _| {},
            n_residuals: 0,
            cost_w2: None,
            ndf: 4,
            opts: &opts,
            gof: true,
        };
        let done = minimize_capped(&fit, NELDER_MEAD_MAX_ITERS);
        assert!(done.valid && done.chi2 < 1e-6, "{done:?}");
        // Stopped after a few steps: a finite χ², but not a minimum.
        let cut = minimize_capped(&fit, 3);
        assert!(cut.chi2.is_finite());
        assert!(!cut.valid, "a capped run must not be valid: {cut:?}");
    }
}
