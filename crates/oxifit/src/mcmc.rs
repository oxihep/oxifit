//! Bayesian sampling of a likelihood: the affine-invariant ensemble sampler of
//! Goodman and Weare (Comm. App. Math. Comp. Sci. 5 (2010) 65), the algorithm
//! of emcee, on any [`Fitter`]'s cost.

use crate::fitter::Fitter;
use crate::result::FitOptions;
use crate::rng::Rng;
use crate::scan::FitError;

/// Draws from the posterior `p(θ | data) ∝ exp(−cost(θ)/2)·π(θ)` of a
/// [`Fitter`]: its cost is `−2 ln L` (with its Gaussian constraints, which act
/// as Gaussian priors), its limits are flat priors (zero density outside), its
/// fixed parameters stay fixed, and every other parameter has a flat prior.
///
/// An ensemble of walkers moves by stretch moves: each walker proposes a
/// point on the line through itself and another walker, so the sampler adapts
/// to correlated and differently scaled parameters with no tuning (emcee's
/// algorithm). The walkers start in a small ball around the best fit. With
/// the `rayon` feature, half of the walkers move in parallel at each step; the
/// chain depends only on the seed.
///
/// ```
/// use oxifit::{FitOptions, FnCost, Fitter, Sampler};
///
/// // A Gaussian likelihood for a mean: x̄ = 1.2 ± 0.3.
/// let cost = FnCost::new(&["mu"], |p| ((p[0] - 1.2) / 0.3).powi(2));
/// let chain = Sampler::new(Fitter::new(&cost, vec![1.0])).seed(3).run(200, 1000)?;
/// assert!((chain.mean("mu").unwrap() - 1.2).abs() < 0.03);
/// assert!((chain.std_dev("mu").unwrap() - 0.3).abs() < 0.03);
/// # Ok::<(), oxifit::FitError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Sampler<'c> {
    fitter: Fitter<'c>,
    walkers: Option<usize>,
    seed: u64,
    stretch: f64,
}

/// The samples of a [`Sampler`] or [`Nuts`](crate::Nuts) run: per step, one
/// draw per walker (or per chain).
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    names: Vec<String>,
    /// Indices of the sampled (free) parameters.
    free: Vec<usize>,
    /// `[step][walker]` → full parameter vector, after the burn-in.
    samples: Vec<Vec<Vec<f64>>>,
    acceptance: f64,
    divergences: usize,
}

impl<'c> Sampler<'c> {
    /// A sampler of `fitter`'s posterior: by default `max(32, 4·d)` walkers
    /// for `d` free parameters, seed 1, stretch scale 2.
    #[must_use]
    pub fn new(fitter: Fitter<'c>) -> Sampler<'c> {
        Sampler {
            fitter,
            walkers: None,
            seed: 1,
            stretch: 2.0,
        }
    }

    /// The number of walkers (an even number, at least twice the free
    /// parameters).
    #[must_use]
    pub fn walkers(mut self, n: usize) -> Sampler<'c> {
        self.walkers = Some(n);
        self
    }

    /// The generator seed: the same seed gives the same chain.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Sampler<'c> {
        self.seed = seed;
        self
    }

    /// The stretch move's scale `a > 1` (emcee's default 2).
    #[must_use]
    pub fn stretch(mut self, a: f64) -> Sampler<'c> {
        self.stretch = a;
        self
    }

    /// `ln p(θ)` up to a constant: `−cost/2` inside the limits, `−∞` outside.
    fn log_post(&self, p: &[f64]) -> f64 {
        for (i, c) in self.fitter.params.constraints.iter().enumerate() {
            if c.lower.is_some_and(|lo| p[i] < lo) || c.upper.is_some_and(|hi| p[i] > hi) {
                return f64::NEG_INFINITY;
            }
        }
        let v = self.fitter.value(p);
        if v.is_finite() {
            -0.5 * v
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Run `burn_in` steps (discarded), then `steps` steps kept, every
    /// walker's position at each.
    ///
    /// # Errors
    /// [`FitError::FitFailed`] if no walker can start at a finite posterior
    /// density; as [`Model::check`](crate::Model::check) for a parameter
    /// setting of an unknown name.
    pub fn run(&self, burn_in: usize, steps: usize) -> Result<Chain, FitError> {
        self.fitter.params.check()?;
        let spec = &self.fitter.params;
        let names = spec.param_names.clone();
        let free: Vec<usize> = (0..spec.params.len())
            .filter(|&i| !spec.constraints.get(i).is_some_and(|c| c.fixed))
            .collect();
        let d = free.len();
        if d == 0 || spec.params.len() != self.fitter.cost.param_names().len() {
            return Err(FitError::FitFailed);
        }
        let n = self.walkers.unwrap_or((4 * d).max(32)).max(2 * d + 2);
        let n = n + n % 2;
        let a = self.stretch.max(1.0 + 1e-6);

        // Start: a ball around the best fit, a tenth of each error wide.
        let best = self.fitter.fit(&FitOptions::new().hesse(true));
        let (center, scale): (Vec<f64>, Vec<f64>) = if best.valid {
            (
                best.params.clone(),
                best.errors
                    .iter()
                    .map(|e| {
                        if e.is_finite() && *e > 0.0 {
                            0.1 * e
                        } else {
                            1e-3
                        }
                    })
                    .collect(),
            )
        } else {
            (
                spec.params.clone(),
                spec.params
                    .iter()
                    .map(|v| 1e-3 * v.abs().max(1.0))
                    .collect(),
            )
        };
        let mut start_rng = Rng::stream(self.seed, u64::MAX);
        let mut pos: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut lp: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut found = None;
            for _ in 0..1000 {
                let mut p = center.clone();
                for &i in &free {
                    p[i] += scale[i] * start_rng.gauss();
                }
                let l = self.log_post(&p);
                if l.is_finite() {
                    found = Some((p, l));
                    break;
                }
            }
            let (p, l) = found.ok_or(FitError::FitFailed)?;
            pos.push(p);
            lp.push(l);
        }

        let half = n / 2;
        let mut samples = Vec::with_capacity(steps);
        let mut accepted = 0usize;
        for step in 0..burn_in + steps {
            for k in 0..2 {
                let (moving, others) = if k == 0 {
                    (0..half, half..n)
                } else {
                    (half..n, 0..half)
                };
                let others: Vec<Vec<f64>> = pos[others].to_vec();
                let propose = |j: usize, x: &[f64], lx: f64| -> (Vec<f64>, f64, bool) {
                    let mut rng = Rng::stream(self.seed, (step * n + j) as u64);
                    let partner = &others[rng.below(others.len())];
                    let z = ((a - 1.0) * rng.uniform() + 1.0).powi(2) / a;
                    let mut y = x.to_vec();
                    for &i in &free {
                        y[i] = partner[i] + z * (x[i] - partner[i]);
                    }
                    let ly = self.log_post(&y);
                    let log_ratio = (d as f64 - 1.0) * z.ln() + ly - lx;
                    if ly.is_finite() && rng.uniform().ln() < log_ratio {
                        (y, ly, true)
                    } else {
                        (x.to_vec(), lx, false)
                    }
                };
                let range: Vec<usize> = moving.collect();
                #[cfg(feature = "rayon")]
                let moved: Vec<(Vec<f64>, f64, bool)> = {
                    use rayon::prelude::*;
                    range
                        .par_iter()
                        .map(|&j| propose(j, &pos[j], lp[j]))
                        .collect()
                };
                #[cfg(not(feature = "rayon"))]
                let moved: Vec<(Vec<f64>, f64, bool)> =
                    range.iter().map(|&j| propose(j, &pos[j], lp[j])).collect();
                for (&j, (p, l, ok)) in range.iter().zip(moved) {
                    pos[j] = p;
                    lp[j] = l;
                    if ok && step >= burn_in {
                        accepted += 1;
                    }
                }
            }
            if step >= burn_in {
                samples.push(pos.clone());
            }
        }
        Ok(Chain {
            names,
            free,
            acceptance: accepted as f64 / (steps * n).max(1) as f64,
            samples,
            divergences: 0,
        })
    }
}

impl Chain {
    pub(crate) fn from_parts(
        names: Vec<String>,
        free: Vec<usize>,
        samples: Vec<Vec<Vec<f64>>>,
        acceptance: f64,
        divergences: usize,
    ) -> Chain {
        Chain {
            names,
            free,
            samples,
            acceptance,
            divergences,
        }
    }

    /// The parameter names (every parameter, fixed ones included).
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Every sample, as full parameter vectors (steps × walkers of them).
    #[must_use]
    pub fn samples(&self) -> Vec<Vec<f64>> {
        self.samples.iter().flatten().cloned().collect()
    }

    /// The fraction of proposals accepted (0.2–0.5 is healthy for the
    /// ensemble sampler); for [`Nuts`](crate::Nuts), the mean acceptance
    /// statistic (near its target, 0.8 by default).
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        self.acceptance
    }

    /// The number of [`Nuts`](crate::Nuts) draws whose trajectory diverged:
    /// the posterior has a region too curved for the step size, which biases
    /// the draws. Should be 0; if not, raise
    /// [`target_accept`](crate::Nuts::target_accept) or reparametrize. Always
    /// 0 for the ensemble sampler.
    #[must_use]
    pub fn divergences(&self) -> usize {
        self.divergences
    }

    /// The split-R̂ of `name` (Gelman et al., *Bayesian Data Analysis*, 3rd
    /// ed.): each walker's or chain's draws are cut in two halves, and the
    /// spread between the halves' means is compared with the spread within
    /// them. Near 1 (below 1.01–1.05) when the chains agree; well above 1
    /// when they have not mixed. `None` for an unknown name or fewer than 4
    /// steps.
    #[must_use]
    pub fn r_hat(&self, name: &str) -> Option<f64> {
        let i = self.index(name)?;
        let steps = self.samples.len();
        let walkers = self.samples.first()?.len();
        if steps < 4 {
            return None;
        }
        let half = steps / 2;
        let mut means = Vec::with_capacity(2 * walkers);
        let mut vars = Vec::with_capacity(2 * walkers);
        for w in 0..walkers {
            for part in [0..half, steps - half..steps] {
                let x: Vec<f64> = self.samples[part].iter().map(|s| s[w][i]).collect();
                let m = x.iter().sum::<f64>() / half as f64;
                vars.push(x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (half - 1) as f64);
                means.push(m);
            }
        }
        let k = means.len() as f64;
        let n = half as f64;
        let grand = means.iter().sum::<f64>() / k;
        let b = n / (k - 1.0) * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>();
        let w = vars.iter().sum::<f64>() / k;
        if w <= 0.0 {
            return Some(if b > 0.0 { f64::INFINITY } else { 1.0 });
        }
        Some((((n - 1.0) / n * w + b / n) / w).sqrt())
    }

    fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Every sampled value of `name` (`None` for an unknown name).
    #[must_use]
    pub fn values(&self, name: &str) -> Option<Vec<f64>> {
        let i = self.index(name)?;
        Some(self.samples.iter().flatten().map(|p| p[i]).collect())
    }

    /// The posterior mean of `name`.
    #[must_use]
    pub fn mean(&self, name: &str) -> Option<f64> {
        let v = self.values(name)?;
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }

    /// The posterior standard deviation of `name`.
    #[must_use]
    pub fn std_dev(&self, name: &str) -> Option<f64> {
        let v = self.values(name)?;
        let m = v.iter().sum::<f64>() / v.len() as f64;
        Some((v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1).max(1) as f64).sqrt())
    }

    /// The posterior quantile `q` (0–1) of `name`.
    #[must_use]
    pub fn quantile(&self, name: &str, q: f64) -> Option<f64> {
        let mut v = self.values(name)?;
        v.sort_by(f64::total_cmp);
        let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos in [0, len − 1]: an index"
        )]
        let (lo, frac) = (pos.floor() as usize, pos.fract());
        Some(v[lo] + frac * (v.get(lo + 1).copied().unwrap_or(v[lo]) - v[lo]))
    }

    /// The central credible interval of probability `level` (e.g. 0.683).
    #[must_use]
    pub fn interval(&self, name: &str, level: f64) -> Option<(f64, f64)> {
        let tail = 0.5 * (1.0 - level);
        Some((self.quantile(name, tail)?, self.quantile(name, 1.0 - tail)?))
    }

    /// The posterior covariance of the sampled parameters, in their order
    /// (the free ones).
    #[must_use]
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let all = self.samples();
        let n = all.len() as f64;
        let means: Vec<f64> = self
            .free
            .iter()
            .map(|&i| all.iter().map(|p| p[i]).sum::<f64>() / n)
            .collect();
        self.free
            .iter()
            .enumerate()
            .map(|(a, &i)| {
                self.free
                    .iter()
                    .enumerate()
                    .map(|(b, &j)| {
                        all.iter()
                            .map(|p| (p[i] - means[a]) * (p[j] - means[b]))
                            .sum::<f64>()
                            / (n - 1.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// The integrated autocorrelation time of `name`, in steps: the walkers'
    /// mean autocorrelation function summed up to the first window `M` with
    /// `M ≥ 5τ` (Sokal's rule, as emcee). The chain should be many times
    /// longer; independent samples ≈ steps × walkers / τ.
    #[must_use]
    pub fn autocorrelation_time(&self, name: &str) -> Option<f64> {
        let i = self.index(name)?;
        let steps = self.samples.len();
        let walkers = self.samples.first()?.len();
        if steps < 4 {
            return None;
        }
        let mut rho = vec![0.0; steps];
        for w in 0..walkers {
            let x: Vec<f64> = self.samples.iter().map(|s| s[w][i]).collect();
            let m = x.iter().sum::<f64>() / steps as f64;
            let c0: f64 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / steps as f64;
            if c0 <= 0.0 {
                continue;
            }
            for (t, r) in rho.iter_mut().enumerate().take(steps / 2) {
                let c: f64 = (0..steps - t)
                    .map(|k| (x[k] - m) * (x[k + t] - m))
                    .sum::<f64>()
                    / steps as f64;
                *r += c / c0 / walkers as f64;
            }
        }
        let mut tau = 1.0;
        for (m, r) in rho.iter().enumerate().take(steps / 2).skip(1) {
            tau += 2.0 * r;
            if (m as f64) >= 5.0 * tau {
                break;
            }
        }
        Some(tau.max(1.0))
    }
}
