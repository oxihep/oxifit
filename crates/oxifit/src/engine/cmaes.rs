//! CMA-ES — the covariance matrix adaptation evolution strategy, `(μ/μ_w, λ)`
//! with rank-one and rank-μ updates and cumulative step-size adaptation
//! (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016). It learns the
//! shape of the cost's valley (correlations and scales) as it goes, so it
//! handles badly scaled, correlated parameters better than differential
//! evolution. Like the other global searches it then polishes the best point
//! with MIGRAD and finishes like any Minuit fit.

use super::Fit;
use super::global::{COST_SPREAD, Rng, scatter, search_box};
use super::linalg::sym_eigen;
use super::minuit::{Local, finish, local};
use crate::result::{CmaEs, FitResult};

/// Initial step size, as a fraction of each parameter's search box.
const SIGMA0: f64 = 0.3;
/// Stop once the search distribution is this small (in box units).
const SIGMA_MIN: f64 = 1e-6;

/// A box coordinate mirrored back into `[0, 1]` (the box is periodic in
/// mirror image), so every sample is evaluated inside the search box.
fn reflect(u: f64) -> f64 {
    let m = u.rem_euclid(2.0);
    if m > 1.0 { 2.0 - m } else { m }
}

/// Run CMA-ES over the free parameters, in coordinates scaled to the search
/// box (`0` and `1` are its edges), started at the model's seed; then MIGRAD
/// from the best point seen.
pub(crate) fn cma_es(fit: &Fit, cfg: CmaEs) -> FitResult {
    let seed = fit.seed();
    let free = fit.free();
    let d = free.len();
    if d == 0 {
        return super::minuit::run(fit, Local::Migrad);
    }
    let bx = search_box(fit, &free, &seed);
    let to_param = |u: &[f64]| -> Vec<f64> {
        let sub: Vec<f64> = u
            .iter()
            .zip(&bx)
            .map(|(&ui, &(lo, hi))| lo + (hi - lo) * reflect(ui))
            .collect();
        scatter(&seed, &free, &sub)
    };
    // A non-finite cost (e.g. a model that overflows) ranks last.
    let eval = |u: &[f64]| {
        let v = (fit.cost)(&to_param(u));
        if v.is_finite() { v } else { f64::INFINITY }
    };

    // Strategy parameters (the tutorial's defaults).
    let df = d as f64;
    // The tutorial's λ = 4 + ⌊3·ln n⌋ is tuned for local refinement; this is a
    // global search, and a larger population finds the right basin far more
    // often (a 4-parameter peak search: 180/200 at λ = 8, 200/200 at λ = 32).
    let lambda = cfg.population.unwrap_or((8 * d).clamp(16, 200)).max(4);
    let mu = lambda / 2;
    let raw: Vec<f64> = (0..mu)
        .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
        .collect();
    let wsum: f64 = raw.iter().sum();
    let w: Vec<f64> = raw.iter().map(|x| x / wsum).collect();
    let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
    let cc = (4.0 + mueff / df) / (df + 4.0 + 2.0 * mueff / df);
    let cs = (mueff + 2.0) / (df + mueff + 5.0);
    let c1 = 2.0 / ((df + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((df + 2.0).powi(2) + mueff));
    let damps = 1.0 + 2.0 * (((mueff - 1.0) / (df + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    let chi_n = df.sqrt() * (1.0 - 1.0 / (4.0 * df) + 1.0 / (21.0 * df * df));

    let mut rng = Rng(cfg.seed);
    let mut mean: Vec<f64> = free
        .iter()
        .zip(&bx)
        .map(|(&i, &(lo, hi))| {
            if hi > lo {
                ((seed[i] - lo) / (hi - lo)).clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect();
    let mut sigma = SIGMA0;
    let mut cov: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut pc = vec![0.0; d];
    let mut ps = vec![0.0; d];
    let mut best = (eval(&mean), mean.clone());

    for generation in 0..cfg.generations {
        // C = B·diag(D²)·Bᵀ; sample x = m + σ·B·D·z.
        let (eig, b) = sym_eigen(&cov);
        let dvec: Vec<f64> = eig.iter().map(|&e| e.max(1e-300).sqrt()).collect();
        let mut pop: Vec<(f64, Vec<f64>, Vec<f64>)> = (0..lambda)
            .map(|_| {
                let z: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                let y: Vec<f64> = (0..d)
                    .map(|r| (0..d).map(|k| b[r][k] * dvec[k] * z[k]).sum())
                    .collect();
                let x: Vec<f64> = mean.iter().zip(&y).map(|(m, yi)| m + sigma * yi).collect();
                (eval(&x), x, y)
            })
            .collect();
        pop.sort_by(|a, b| a.0.total_cmp(&b.0));
        if pop[0].0 < best.0 {
            best = (pop[0].0, pop[0].1.clone());
        }

        // Recombination: the new mean and the weighted step y_w.
        let yw: Vec<f64> = (0..d)
            .map(|k| (0..mu).map(|i| w[i] * pop[i].2[k]).sum())
            .collect();
        for (m, y) in mean.iter_mut().zip(&yw) {
            *m += sigma * y;
        }

        // Step-size path, with C^{-1/2}·y_w = B·D⁻¹·Bᵀ·y_w.
        let bty: Vec<f64> = (0..d)
            .map(|k| (0..d).map(|r| b[r][k] * yw[r]).sum::<f64>() / dvec[k])
            .collect();
        let inv_sqrt_y: Vec<f64> = (0..d)
            .map(|r| (0..d).map(|k| b[r][k] * bty[k]).sum())
            .collect();
        let ks = (cs * (2.0 - cs) * mueff).sqrt();
        for (p, v) in ps.iter_mut().zip(&inv_sqrt_y) {
            *p = (1.0 - cs) * *p + ks * v;
        }
        let ps_norm = ps.iter().map(|x| x * x).sum::<f64>().sqrt();
        let hsig =
            ps_norm / (1.0 - (1.0 - cs).powf(2.0 * (generation as f64 + 1.0))).sqrt() / chi_n
                < 1.4 + 2.0 / (df + 1.0);
        let kc = (cc * (2.0 - cc) * mueff).sqrt();
        for (p, y) in pc.iter_mut().zip(&yw) {
            *p = (1.0 - cc) * *p + if hsig { kc * y } else { 0.0 };
        }

        // Covariance: rank-one (pc) and rank-μ (the selected steps) updates.
        let delta = if hsig { 0.0 } else { cc * (2.0 - cc) };
        for r in 0..d {
            for c in 0..=r {
                let rank_mu: f64 = (0..mu).map(|i| w[i] * pop[i].2[r] * pop[i].2[c]).sum();
                let v = (1.0 - c1 - cmu) * cov[r][c]
                    + c1 * (pc[r] * pc[c] + delta * cov[r][c])
                    + cmu * rank_mu;
                cov[r][c] = v;
                cov[c][r] = v;
            }
        }
        sigma *= ((cs / damps) * (ps_norm / chi_n - 1.0)).exp();

        // Converged: the distribution has shrunk, or the population's costs
        // agree to a small fraction of a 1σ contour.
        let spread = pop[lambda - 1].0 - pop[0].0;
        let width = sigma * eig.iter().copied().fold(0.0, f64::max).sqrt();
        if width < SIGMA_MIN || (spread.is_finite() && spread <= COST_SPREAD) {
            break;
        }
    }

    let start = to_param(&best.1);
    finish(fit, local(fit, &start, Local::Migrad))
}

#[cfg(test)]
mod tests {
    use super::reflect;

    #[test]
    fn reflect_folds_into_the_unit_interval() {
        for (u, want) in [
            (0.3, 0.3),
            (1.2, 0.8),
            (-0.25, 0.25),
            (2.5, 0.5),
            (-1.5, 0.5),
        ] {
            assert!((reflect(u) - want).abs() < 1e-12, "{u} -> {}", reflect(u));
        }
    }
}
