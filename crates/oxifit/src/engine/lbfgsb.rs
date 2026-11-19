//! L-BFGS-B: a limited-memory quasi-Newton minimizer with bounds — the
//! gradient method of SciPy and zfit. This is the projected form: each step
//! is the L-BFGS direction (two-loop recursion over the last
//! [`MEMORY`] steps) on the parameters not held at a limit, followed along the
//! box-projected path with a backtracking (Armijo) line search. Gradients come
//! from the model when it has an analytic one, else from central differences.
//! MIGRAD + HESSE then polish the point, as for every Minuit-finished backend.

use super::Fit;
use super::global::scatter;
use super::minuit::{Local, finish, local};
use crate::result::FitResult;

/// Steps of curvature history kept.
const MEMORY: usize = 10;
/// The most iterations.
const MAX_ITERS: usize = 2000;
/// Stop once an iteration lowers the cost by less than this (relative)…
const COST_TOL: f64 = 1e-12;
/// …or the projected gradient is this small (relative to the cost scale).
const GRAD_TOL: f64 = 1e-9;

/// Minimize with projected L-BFGS from the model's seed, then MIGRAD.
pub(crate) fn lbfgsb(fit: &Fit) -> FitResult {
    let seed = fit.seed();
    let free = fit.free();
    let d = free.len();
    if d == 0 {
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
    let project = |x: &mut [f64]| {
        for (v, &(lo, hi)) in x.iter_mut().zip(&bounds) {
            *v = v.max(lo).min(hi);
        }
    };
    let cost = |x: &[f64]| {
        let c = (fit.cost)(&scatter(&seed, &free, x));
        if c.is_finite() { c } else { f64::INFINITY }
    };
    // The gradient over the free parameters: analytic, or central differences
    // (stepping inside the limits).
    let gradient = |x: &[f64], g: &mut [f64]| {
        if let Some(grad) = fit.grad {
            let full = scatter(&seed, &free, x);
            let mut all = vec![0.0; full.len()];
            grad(&full, &mut all);
            for (gk, &i) in g.iter_mut().zip(&free) {
                *gk = all[i];
            }
        } else {
            let mut xh = x.to_vec();
            for k in 0..d {
                let h = 1e-6 * x[k].abs().max(fit.step(free[k], x[k]));
                let (lo, hi) = bounds[k];
                let (a, b) = ((x[k] - h).max(lo), (x[k] + h).min(hi));
                xh[k] = b;
                let fb = cost(&xh);
                xh[k] = a;
                let fa = cost(&xh);
                xh[k] = x[k];
                g[k] = if b > a { (fb - fa) / (b - a) } else { 0.0 };
            }
        }
    };

    let mut x: Vec<f64> = free.iter().map(|&i| seed[i]).collect();
    project(&mut x);
    let mut f = cost(&x);
    let mut g = vec![0.0; d];
    gradient(&x, &mut g);
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new(); // (s, y, 1/y·s)

    for iter in 0..MAX_ITERS {
        if !f.is_finite() {
            break;
        }
        // Free set: not held at a limit by a gradient pushing outward.
        let held: Vec<bool> = (0..d)
            .map(|k| {
                let (lo, hi) = bounds[k];
                (x[k] <= lo && g[k] > 0.0) || (x[k] >= hi && g[k] < 0.0)
            })
            .collect();
        let pg_norm = (0..d)
            .filter(|&k| !held[k])
            .map(|k| g[k].abs())
            .fold(0.0, f64::max);
        if pg_norm <= GRAD_TOL * (1.0 + f.abs()) {
            break;
        }

        // Two-loop recursion on the free components.
        let mut q: Vec<f64> = (0..d).map(|k| if held[k] { 0.0 } else { g[k] }).collect();
        let dot = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .zip(&held)
                .filter(|(_, h)| !**h)
                .map(|((u, v), _)| u * v)
                .sum()
        };
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let a = rho * dot(s, &q);
            for k in 0..d {
                q[k] -= a * y[k];
            }
            alphas.push(a);
        }
        if let Some((s, y, _)) = history.last() {
            let gamma = dot(s, y) / dot(y, y).max(1e-300);
            if gamma.is_finite() && gamma > 0.0 {
                q.iter_mut().for_each(|v| *v *= gamma);
            }
        }
        for ((s, y, rho), a) in history.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            for k in 0..d {
                q[k] += s[k] * (a - b);
            }
        }
        let mut dir: Vec<f64> = (0..d).map(|k| if held[k] { 0.0 } else { -q[k] }).collect();
        let mut slope: f64 = dir.iter().zip(&g).map(|(a, b)| a * b).sum();
        if slope >= 0.0 || !slope.is_finite() {
            // Not a descent direction: projected steepest descent, and a
            // fresh curvature history.
            dir = (0..d).map(|k| if held[k] { 0.0 } else { -g[k] }).collect();
            slope = -dir.iter().map(|v| v * v).sum::<f64>();
            history.clear();
        }

        // Backtracking line search along the projected path.
        let mut step = if iter == 0 && history.is_empty() {
            (1.0 / dir.iter().map(|v| v.abs()).fold(0.0, f64::max)).min(1.0)
        } else {
            1.0
        };
        let mut accepted = None;
        for _ in 0..40 {
            let mut trial: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a + step * b).collect();
            project(&mut trial);
            let ft = cost(&trial);
            let decrease: f64 = trial
                .iter()
                .zip(&x)
                .zip(&g)
                .map(|((t, a), gk)| (t - a) * gk)
                .sum();
            if ft <= f + 1e-4 * decrease.min(step * slope).min(0.0) && ft < f {
                accepted = Some((trial, ft));
                break;
            }
            step *= 0.5;
        }
        let Some((x_new, f_new)) = accepted else {
            break; // no descent along the direction: at a (bounded) minimum
        };
        let mut g_new = vec![0.0; d];
        gradient(&x_new, &mut g_new);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let ss: f64 = s.iter().map(|v| v * v).sum();
        let yy: f64 = y.iter().map(|v| v * v).sum();
        if sy > 1e-10 * (ss * yy).sqrt() {
            history.push((s, y, 1.0 / sy));
            if history.len() > MEMORY {
                history.remove(0);
            }
        }
        let gain = f - f_new;
        x = x_new;
        f = f_new;
        g = g_new;
        if gain <= COST_TOL * (1.0 + f.abs()) {
            break;
        }
    }

    let start = scatter(&seed, &free, &x);
    finish(fit, local(fit, &start, Local::Migrad))
}
