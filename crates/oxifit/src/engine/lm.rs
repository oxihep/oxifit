//! Levenberg–Marquardt: Gauss–Newton steps on the residual vector, damped
//! toward gradient descent when a step fails (Marquardt's diagonal scaling).
//! Every cost here is a sum of squared residuals (the likelihood as signed
//! deviance residuals), so it applies to all of them. The Jacobian is a
//! forward difference; limits are enforced by projecting each step back into
//! them. It converges in few iterations on least-squares problems; MIGRAD
//! then polishes the point and HESSE supplies the errors, as for every
//! Minuit-finished backend.

use super::Fit;
use super::global::scatter;
use super::linalg::solve;
use super::minuit::{Local, finish, local};
use crate::result::FitResult;

/// The most Gauss–Newton iterations.
const MAX_ITERS: usize = 500;
/// Stop once an accepted step lowers the cost by less than this (relative).
const COST_TOL: f64 = 1e-12;

/// Minimize by Levenberg–Marquardt from the model's seed, then MIGRAD.
pub(crate) fn levenberg_marquardt(fit: &Fit) -> FitResult {
    let seed = fit.seed();
    let free = fit.free();
    let d = free.len();
    let n = fit.n_residuals;
    if d == 0 || n == 0 {
        return super::minuit::run(fit, Local::Migrad);
    }
    let bounds: Vec<(f64, f64)> = free
        .iter()
        .map(|&i| {
            let c = fit.constraint(i);
            (
                c.lower.unwrap_or(f64::NEG_INFINITY),
                c.upper.unwrap_or(f64::INFINITY),
            )
        })
        .collect();
    let residuals = |sub: &[f64], out: &mut [f64]| -> f64 {
        (fit.residuals)(&scatter(&seed, &free, sub), out);
        let c: f64 = out.iter().map(|r| r * r).sum();
        if c.is_finite() { c } else { f64::INFINITY }
    };

    let mut x: Vec<f64> = free.iter().map(|&i| seed[i]).collect();
    let mut r = vec![0.0; n];
    let mut cost = residuals(&x, &mut r);
    let mut lambda = 1e-3;
    let mut jac = vec![vec![0.0; d]; n];
    let mut r_step = vec![0.0; n];

    if cost.is_finite() {
        'outer: for _ in 0..MAX_ITERS {
            // Forward-difference Jacobian, stepping inward at an upper limit.
            for k in 0..d {
                let mut h = 1e-7 * x[k].abs().max(fit.step(free[k], x[k]));
                if x[k] + h > bounds[k].1 {
                    h = -h;
                }
                let mut xh = x.clone();
                xh[k] += h;
                residuals(&xh, &mut r_step);
                for (row, (&rs, &r0)) in jac.iter_mut().zip(r_step.iter().zip(&r)) {
                    row[k] = (rs - r0) / h;
                }
            }
            let jtj: Vec<Vec<f64>> = (0..d)
                .map(|a| {
                    (0..d)
                        .map(|b| jac.iter().map(|row| row[a] * row[b]).sum())
                        .collect()
                })
                .collect();
            let jtr: Vec<f64> = (0..d)
                .map(|a| jac.iter().zip(&r).map(|(row, ri)| row[a] * ri).sum())
                .collect();

            // Damped steps until one lowers the cost.
            loop {
                let mut a = jtj.clone();
                for (k, row) in a.iter_mut().enumerate() {
                    row[k] += lambda * jtj[k][k].max(1e-12);
                }
                let neg: Vec<f64> = jtr.iter().map(|g| -g).collect();
                if let Some(delta) = solve(&a, &neg) {
                    let trial: Vec<f64> = x
                        .iter()
                        .zip(&delta)
                        .zip(&bounds)
                        .map(|((xi, di), &(lo, hi))| (xi + di).max(lo).min(hi))
                        .collect();
                    let c = residuals(&trial, &mut r_step);
                    if c < cost {
                        let gain = cost - c;
                        x = trial;
                        r.copy_from_slice(&r_step);
                        cost = c;
                        lambda = (lambda / 10.0).max(1e-12);
                        if gain <= COST_TOL * (1.0 + cost) {
                            break 'outer;
                        }
                        break;
                    }
                }
                lambda *= 10.0;
                if lambda > 1e12 {
                    break 'outer; // no step lowers the cost: at a minimum
                }
            }
        }
    }

    let start = scatter(&seed, &free, &x);
    finish(fit, local(fit, &start, Local::Migrad))
}
