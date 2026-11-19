//! Global searches: [`multi_start`] and [`differential_evolution`]. Both look
//! for the right basin inside a search box, then polish the best point with a
//! Minuit2 minimization and [`finish`] it like any Minuit fit, so the values,
//! errors, covariance and validity always come from Minuit2 + HESSE.

use std::panic::{AssertUnwindSafe, catch_unwind};

use minuit2::FunctionMinimum;

use super::Fit;
use super::minuit::{Local, finish, local};
use crate::result::{DifferentialEvolution, FitResult, MultiStart};

/// A small deterministic generator (SplitMix64): global searches must be
/// reproducible from their seed, and need nothing stronger.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        // n is a population or dimension count: the modulo bias is negligible.
        (self.next_u64() % n as u64) as usize
    }
}

/// The search box `[lo, hi]` of each free parameter: its limits when it has
/// both; otherwise the open side reaches 10 initial steps beyond the seed.
fn search_box(fit: &Fit, free: &[usize], seed: &[f64]) -> Vec<(f64, f64)> {
    free.iter()
        .map(|&i| {
            let c = fit.constraint(i);
            let x0 = seed[i];
            let w = 10.0 * fit.step(i, x0);
            match (c.lower, c.upper) {
                (Some(lo), Some(hi)) => (lo, hi),
                (Some(lo), None) => (lo, x0.max(lo) + w),
                (None, Some(hi)) => (x0.min(hi) - w, hi),
                (None, None) => (x0 - w, x0 + w),
            }
        })
        .collect()
}

/// `n` Latin-hypercube points in the box: each coordinate's range is cut into
/// `n` strata and every stratum is used exactly once, so even a few points
/// spread over every parameter's whole range.
fn latin_hypercube(rng: &mut Rng, bx: &[(f64, f64)], n: usize) -> Vec<Vec<f64>> {
    let mut points = vec![vec![0.0; bx.len()]; n];
    for (d, &(lo, hi)) in bx.iter().enumerate() {
        let mut strata: Vec<usize> = (0..n).collect();
        for k in (1..n).rev() {
            strata.swap(k, rng.below(k + 1)); // Fisher–Yates
        }
        for (p, &s) in points.iter_mut().zip(&strata) {
            p[d] = lo + (hi - lo) * (s as f64 + rng.uniform()) / n as f64;
        }
    }
    points
}

/// A full parameter vector: `seed` with the free parameters set from `sub`.
fn scatter(seed: &[f64], free: &[usize], sub: &[f64]) -> Vec<f64> {
    let mut full = seed.to_vec();
    for (&i, &v) in free.iter().zip(sub) {
        full[i] = v;
    }
    full
}

/// The cost of a minimum for ranking: a non-finite cost ranks last.
fn rank(m: &FunctionMinimum) -> f64 {
    let f = m.fval();
    if f.is_finite() { f } else { f64::INFINITY }
}

/// Minimize (MIGRAD, falling back to SIMPLEX) from the model's seed and from
/// `starts − 1` Latin-hypercube points in the search box, and keep the
/// lowest-cost minimum. A start whose minimization panics is skipped. The
/// winner is ranked by cost alone, not validity: a run that reached the right
/// basin but stopped short (invalid) beats a valid minimum in the wrong one,
/// and is minimized again from where it stopped. (MIGRAD alone can stall far
/// from a start in a bad region; the SIMPLEX fallback is what makes every start
/// count.)
pub(crate) fn multi_start(fit: &Fit, cfg: MultiStart) -> FitResult {
    let seed = fit.seed();
    let free = fit.free();
    let n = cfg.starts.max(1);
    let mut rng = Rng(cfg.seed);
    let bx = search_box(fit, &free, &seed);

    let mut starts = vec![seed.clone()];
    starts.extend(
        latin_hypercube(&mut rng, &bx, n - 1)
            .iter()
            .map(|sub| scatter(&seed, &free, sub)),
    );

    let mut best: Option<FunctionMinimum> = None;
    for start in &starts {
        let Ok(min) = catch_unwind(AssertUnwindSafe(|| local(fit, start, Local::Combined))) else {
            continue;
        };
        if best.as_ref().is_none_or(|b| rank(&min) < rank(b)) {
            best = Some(min);
        }
    }
    let Some(mut min) = best else {
        return super::failed(fit.model);
    };
    if !min.is_valid()
        && let Ok(again) = catch_unwind(AssertUnwindSafe(|| {
            local(fit, &min.params(), Local::Combined)
        }))
        && (again.is_valid() || rank(&again) <= rank(&min))
    {
        min = again;
    }
    finish(fit, min)
}

/// Crossover rate: the chance each coordinate comes from the mutant. A high
/// value suits the correlated parameters of a typical fit.
const CROSSOVER: f64 = 0.9;
/// Stop once every member's cost is within this of the best: the population
/// then sits in one basin, a small fraction of a 1σ (`Δcost = 1`) contour.
const COST_SPREAD: f64 = 1e-2;

/// Differential evolution (`DE/rand/1/bin`, dithered `F ∈ [0.5, 1)`) over the
/// free parameters inside the search box, the model's seed among the initial
/// population; then MIGRAD from the best member. A trial outside the box is
/// pulled back between its parent and the crossed bound.
pub(crate) fn differential_evolution(fit: &Fit, cfg: DifferentialEvolution) -> FitResult {
    let seed = fit.seed();
    let free = fit.free();
    let d = free.len();
    if d == 0 {
        return super::minuit::run(fit, Local::Migrad);
    }
    let np = cfg.population.unwrap_or((15 * d).clamp(20, 300)).max(4);
    let mut rng = Rng(cfg.seed);
    let bx = search_box(fit, &free, &seed);

    // A non-finite cost (e.g. a model that overflows) loses every comparison.
    let eval = |sub: &[f64]| {
        let v = (fit.cost)(&scatter(&seed, &free, sub));
        if v.is_finite() { v } else { f64::INFINITY }
    };

    let mut pop = latin_hypercube(&mut rng, &bx, np);
    pop[0] = free.iter().map(|&i| seed[i]).collect();
    let mut vals: Vec<f64> = pop.iter().map(|x| eval(x)).collect();

    let mut trial = vec![0.0; d];
    for _ in 0..cfg.generations {
        let f = 0.5 + 0.5 * rng.uniform();
        for i in 0..np {
            let (a, b, c) = distinct3(&mut rng, np, i);
            let forced = rng.below(d);
            for j in 0..d {
                let x = pop[i][j];
                let v = if j == forced || rng.uniform() < CROSSOVER {
                    pop[a][j] + f * (pop[b][j] - pop[c][j])
                } else {
                    x
                };
                let (lo, hi) = bx[j];
                trial[j] = if v < lo {
                    lo + rng.uniform() * (x - lo)
                } else if v > hi {
                    hi - rng.uniform() * (hi - x)
                } else {
                    v
                };
            }
            let val = eval(&trial);
            if val <= vals[i] {
                pop[i].copy_from_slice(&trial);
                vals[i] = val;
            }
        }
        let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if hi - lo <= COST_SPREAD {
            break;
        }
    }

    let best = (0..np)
        .min_by(|&a, &b| vals[a].total_cmp(&vals[b]))
        .expect("population is non-empty");
    let start = scatter(&seed, &free, &pop[best]);
    finish(fit, local(fit, &start, Local::Migrad))
}

/// Three distinct population indices, all different from `i` (`n ≥ 4`).
fn distinct3(rng: &mut Rng, n: usize, i: usize) -> (usize, usize, usize) {
    let mut pick = |taken: &[usize]| loop {
        let k = rng.below(n);
        if !taken.contains(&k) {
            break k;
        }
    };
    let a = pick(&[i]);
    let b = pick(&[i, a]);
    let c = pick(&[i, a, b]);
    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_hypercube_uses_every_stratum_once() {
        let mut rng = Rng(7);
        let bx = [(0.0, 10.0), (-1.0, 1.0)];
        let pts = latin_hypercube(&mut rng, &bx, 10);
        for (d, &(lo, hi)) in bx.iter().enumerate() {
            let mut strata: Vec<usize> = pts
                .iter()
                .map(|p| ((p[d] - lo) / (hi - lo) * 10.0) as usize)
                .collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn distinct3_never_repeats() {
        let mut rng = Rng(1);
        for i in 0..4 {
            for _ in 0..100 {
                let (a, b, c) = distinct3(&mut rng, 4, i);
                let mut all = [i, a, b, c];
                all.sort_unstable();
                assert_eq!(all, [0, 1, 2, 3]);
            }
        }
    }
}
