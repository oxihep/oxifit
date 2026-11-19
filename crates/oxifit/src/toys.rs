//! Toy (pseudo-experiment) studies — RooFit's `RooMCStudy`: generate data from
//! a model at known parameters, fit each sample, and check the fits through
//! their pulls `(fitted − true)/error`, which are standard normal for an
//! unbiased fit with correct errors.

use crate::data::{Events, FitExt, Point, Points};
use crate::model::Model;
use crate::result::{FitOptions, FitResult};
use crate::rng::Rng;

/// Cells of the grid the unbinned generator draws from: the density is
/// sampled at their edges and taken linear within each.
const GRID: usize = 4096;

/// How a [`Toys`] study generates each sample.
#[derive(Debug, Clone)]
enum Kind {
    /// Poisson counts at fixed `x` (the expected count is the model there).
    Binned(Vec<f64>),
    /// Events on `[lo, hi]`, `n` of them, or a Poisson number of mean `∫f`.
    Unbinned {
        lo: f64,
        hi: f64,
        n: usize,
        extended: bool,
    },
}

/// A toy study: generate pseudo-data from `truth` (a model at its true
/// parameters) and fit each sample. Build it with [`binned`](Self::binned) or
/// [`unbinned`](Self::unbinned), then [`run`](Self::run) it.
///
/// ```
/// # use oxifit::{FitMethod, FitOptions, Model, Toys};
/// let truth = Model::polynomial("flat", 0).with_params(vec![50.0]);
/// let xs: Vec<f64> = (0..20).map(f64::from).collect();
/// let study = Toys::binned(&truth, &xs)
///     .seed(1)
///     .run(&truth, &FitOptions::new().method(FitMethod::Likelihood), 200);
/// let pull = study.pull_summary("p0").unwrap();
/// assert!(pull.mean.abs() < 4.0 * pull.mean_err && (pull.width - 1.0).abs() < 4.0 * pull.width_err);
/// ```
#[derive(Debug, Clone)]
pub struct Toys {
    truth: Model,
    kind: Kind,
    seed: u64,
}

impl Toys {
    /// Binned toys: at each `x`, a Poisson count of mean `truth(x)` (with
    /// error `√count`) — pseudo-histograms for the binned fits. A negative
    /// model value counts as zero.
    #[must_use]
    pub fn binned(truth: &Model, xs: &[f64]) -> Toys {
        Toys {
            truth: truth.clone(),
            kind: Kind::Binned(xs.to_vec()),
            seed: 1,
        }
    }

    /// Unbinned toys: `n` events on `range` drawn from the shape of `truth`
    /// (for [`FitMethod::UnbinnedLikelihood`](crate::FitMethod::UnbinnedLikelihood)).
    /// Use [`extended`](Self::extended) to draw a Poisson number of events of
    /// mean `∫truth` instead.
    #[must_use]
    pub fn unbinned(truth: &Model, range: (f64, f64), n: usize) -> Toys {
        let (lo, hi) = if range.0 <= range.1 {
            range
        } else {
            (range.1, range.0)
        };
        Toys {
            truth: truth.clone(),
            kind: Kind::Unbinned {
                lo,
                hi,
                n,
                extended: false,
            },
            seed: 1,
        }
    }

    /// Draw a Poisson number of events of mean `∫truth` over the range — the
    /// truth being an event density, for
    /// [`FitMethod::ExtendedUnbinnedLikelihood`](crate::FitMethod::ExtendedUnbinnedLikelihood).
    /// No effect on binned toys.
    #[must_use]
    pub fn extended(mut self) -> Toys {
        if let Kind::Unbinned { extended, .. } = &mut self.kind {
            *extended = true;
        }
        self
    }

    /// Set the generator seed (default 1): the same seed gives the same study.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Toys {
        self.seed = seed;
        self
    }

    /// Generate `ntoys` samples and fit `model` (seeded from its own
    /// parameters, usually the truth) to each with `opts`. The truth is
    /// `truth`'s parameters; `model` must have the same parameters, in the
    /// same order, for the pulls to compare them.
    #[must_use]
    pub fn run(&self, model: &Model, opts: &FitOptions, ntoys: usize) -> ToyStudy {
        let mut rng = Rng(self.seed);
        let f = |x: f64| self.truth.eval(x);
        let fits = match &self.kind {
            Kind::Binned(xs) => {
                let mu: Vec<f64> = xs.iter().map(|&x| f(x).max(0.0)).collect();
                (0..ntoys)
                    .map(|_| {
                        let pts = xs
                            .iter()
                            .zip(&mu)
                            .map(|(&x, &m)| {
                                let n = rng.poisson(m) as f64;
                                Point::new(x, n, n.sqrt())
                            })
                            .collect();
                        Points::from_points(pts).fit_opts(model, opts)
                    })
                    .collect()
            }
            &Kind::Unbinned {
                lo,
                hi,
                n,
                extended,
            } => {
                let sampler = Sampler::new(f, lo, hi);
                (0..ntoys)
                    .map(|_| {
                        let count = if extended {
                            rng.poisson(sampler.total) as usize
                        } else {
                            n
                        };
                        let xs: Vec<f64> = (0..count).map(|_| sampler.draw(&mut rng)).collect();
                        Events::new(&xs).fit_opts(model, opts)
                    })
                    .collect()
            }
        };
        ToyStudy {
            param_names: model.param_names.clone(),
            truth: self.truth.params.clone(),
            fits,
        }
    }
}

/// Draws from a density on `[lo, hi]`: the density tabulated at the edges of
/// [`GRID`] equal cells and taken linear within each, so a cell is chosen by
/// its trapezoid area and a point within it by inverting the linear CDF.
struct Sampler {
    lo: f64,
    width: f64,
    /// The density at the cell edges (clamped at zero).
    edge: Vec<f64>,
    /// Cumulative cell areas, normalized to end at 1.
    cdf: Vec<f64>,
    /// The density's integral over the range (the trapezoid sum).
    total: f64,
}

impl Sampler {
    fn new(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> Sampler {
        let width = (hi - lo) / GRID as f64;
        let edge: Vec<f64> = (0..=GRID)
            .map(|k| {
                let v = f(lo + width * k as f64);
                if v.is_finite() { v.max(0.0) } else { 0.0 }
            })
            .collect();
        let mut cdf = Vec::with_capacity(GRID);
        let mut acc = 0.0;
        for k in 0..GRID {
            acc += 0.5 * (edge[k] + edge[k + 1]) * width;
            cdf.push(acc);
        }
        let total = acc;
        if total > 0.0 {
            for c in &mut cdf {
                *c /= total;
            }
        }
        Sampler {
            lo,
            width,
            edge,
            cdf,
            total,
        }
    }

    fn draw(&self, rng: &mut Rng) -> f64 {
        let u = rng.uniform();
        let k = self.cdf.partition_point(|&c| c < u).min(GRID - 1);
        let (a, b) = (self.edge[k], self.edge[k + 1]);
        // Within the cell the density is a + (b − a)·t, t ∈ [0, 1]: solve
        // a·t + (b − a)·t²/2 = v·(a + b)/2 for a uniform v.
        let v = rng.uniform();
        let t = if (b - a).abs() <= 1e-12 * (a + b) {
            v
        } else {
            let area = v * 0.5 * (a + b);
            (-a + (a * a + 2.0 * (b - a) * area).max(0.0).sqrt()) / (b - a)
        };
        self.lo + self.width * (k as f64 + t.clamp(0.0, 1.0))
    }
}

/// The outcome of a [`Toys`] study: every toy's fit and the truth they are
/// compared with.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToyStudy {
    /// The fitted model's parameter names.
    pub param_names: Vec<String>,
    /// The true parameter values the toys were generated with.
    pub truth: Vec<f64>,
    /// Each toy's fit (invalid ones included).
    pub fits: Vec<FitResult>,
}

/// The pull distribution of one parameter over a [`ToyStudy`]'s valid fits:
/// for an unbiased fit with correct errors, mean 0 and width 1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PullSummary {
    /// Valid fits with a usable error, the pulls' count.
    pub n: usize,
    /// The pulls' mean (the bias, in units of the error).
    pub mean: f64,
    /// The mean's standard error, `width/√n`.
    pub mean_err: f64,
    /// The pulls' standard deviation (the error's scale: above 1, the errors
    /// are too small).
    pub width: f64,
    /// The width's standard error, `width/√(2(n − 1))`.
    pub width_err: f64,
}

impl ToyStudy {
    /// The fraction of toys whose fit was valid.
    #[must_use]
    pub fn valid_fraction(&self) -> f64 {
        if self.fits.is_empty() {
            return f64::NAN;
        }
        self.fits.iter().filter(|f| f.valid).count() as f64 / self.fits.len() as f64
    }

    /// The pulls `(fitted − true)/error` of parameter `param` over the valid
    /// fits with a positive, finite error; `None` for an unknown parameter.
    #[must_use]
    pub fn pulls(&self, param: &str) -> Option<Vec<f64>> {
        let i = self.param_names.iter().position(|n| n == param)?;
        let truth = *self.truth.get(i)?;
        Some(
            self.fits
                .iter()
                .filter(|f| f.valid)
                .filter_map(|f| {
                    let e = f.errors[i];
                    (e.is_finite() && e > 0.0).then(|| (f.params[i] - truth) / e)
                })
                .collect(),
        )
    }

    /// The fitted values of `param` over the valid fits; `None` for an
    /// unknown parameter.
    #[must_use]
    pub fn values(&self, param: &str) -> Option<Vec<f64>> {
        let i = self.param_names.iter().position(|n| n == param)?;
        Some(
            self.fits
                .iter()
                .filter(|f| f.valid)
                .map(|f| f.params[i])
                .collect(),
        )
    }

    /// The mean and width of `param`'s pulls, with their standard errors;
    /// `None` for an unknown parameter or fewer than two pulls.
    #[must_use]
    pub fn pull_summary(&self, param: &str) -> Option<PullSummary> {
        let p = self.pulls(param)?;
        let n = p.len();
        if n < 2 {
            return None;
        }
        let mean = p.iter().sum::<f64>() / n as f64;
        let width = (p.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        Some(PullSummary {
            n,
            mean,
            mean_err: width / (n as f64).sqrt(),
            width,
            width_err: width / (2.0 * (n - 1) as f64).sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sampler_follows_its_density() {
        // exp(−x/2) on [0, 5]: mean 2 − 5·e^{−2.5}/(1 − e^{−2.5}).
        let s = Sampler::new(|x| (-x / 2.0).exp(), 0.0, 5.0);
        let want_total = 2.0 * (1.0 - (-2.5f64).exp());
        assert!((s.total / want_total - 1.0).abs() < 1e-6);
        let mut rng = Rng(3);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| s.draw(&mut rng)).collect();
        assert!(xs.iter().all(|&x| (0.0..=5.0).contains(&x)));
        let mean = xs.iter().sum::<f64>() / n as f64;
        let e = (-2.5f64).exp();
        let want = 2.0 - 5.0 * e / (1.0 - e);
        assert!((mean - want).abs() < 0.01, "{mean} vs {want}");
    }
}
