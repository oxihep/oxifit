//! Hypothesis tests on a likelihood: discovery significance, CLs and upper
//! limits on a parameter of interest — asymptotic (Cowan, Cranmer, Gross,
//! Vitells, EPJC 71 (2011) 1554) or from toys — and the look-elsewhere effect
//! of a scan (Gross, Vitells, EPJC 70 (2010) 525).
//!
//! Every test works on a [`Fitter`]: a [`Cost`] (`−2 ln L`) with its seeds and
//! settings. The asymptotic results need the cost's Asimov dataset and the
//! toy results its pseudo-datasets ([`Cost::asimov`], [`Cost::toy`]), which
//! a data fit ([`ModelCost`](crate::ModelCost)) and sums of them provide. The
//! Fitter's Gaussian constraints are the auxiliary measurements: for an
//! Asimov dataset their means are set to the generating values, for a toy
//! drawn around them.

use std::sync::OnceLock;

use crate::cost::Cost;
use crate::fitter::{CostRefit, Fitter};
use crate::result::{FitOptions, FitResult};
use crate::rng::Rng;
use crate::scan::{FitError, Refit};

/// `1 − Φ(x)`, the standard normal upper tail, accurate far out.
fn tail(x: f64) -> f64 {
    0.5 * oxistats::erfc(x / std::f64::consts::SQRT_2)
}

/// The significance `Z` of a one-sided p-value: `p = 1 − Φ(Z)`.
fn z_of(p: f64) -> f64 {
    oxistats::significance_from_pvalue(p)
}

/// The test statistic a hypothesis test uses for upper limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TestStatistic {
    /// `q̃_μ`: `q_μ` with the best-fit `μ̂` held at zero or above — for a
    /// signal strength that cannot be negative (the LHC default, pyhf's
    /// `qtilde`).
    #[default]
    QMuTilde,
    /// `q_μ`: `−2 ln λ(μ)` for `μ̂ ≤ μ`, zero above (pyhf's `q`).
    QMu,
}

/// A discovery test: is there a signal (`μ = 0` rejected)?
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Discovery {
    /// The best-fit parameter of interest.
    pub mu_hat: f64,
    /// The observed `q₀ = −2 ln λ(0)` (zero when `μ̂ < 0`).
    pub q0: f64,
    /// The observed p-value of the background-only hypothesis.
    pub p0: f64,
    /// The observed significance, `Z = Φ⁻¹(1 − p₀)` (`√q₀` asymptotically).
    pub z: f64,
    /// The signal strength the expected significance assumes.
    pub mu_expected: f64,
    /// The median expected significance for a signal of `mu_expected`:
    /// `√q₀` on its Asimov dataset.
    pub z_expected: f64,
}

/// CLs for one value of the parameter of interest.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Cls {
    /// The tested value of the parameter of interest.
    pub mu: f64,
    /// The observed test statistic.
    pub q: f64,
    /// The test statistic on the background-only Asimov dataset (asymptotic
    /// tests; `NaN` for toys).
    pub q_asimov: f64,
    /// `CLs+b`: the p-value of the signal-plus-background hypothesis.
    pub clsb: f64,
    /// `CLb`: one minus the p-value of the background-only hypothesis, i.e.
    /// the probability under background only of a test statistic at least as
    /// large as observed.
    pub clb: f64,
    /// `CLs = CLs+b / CLb`; the value is excluded at confidence level `1 − α`
    /// when `CLs < α`.
    pub cls: f64,
    /// The expected CLs under background only, at the −2σ, −1σ, median, +1σ
    /// and +2σ quantiles of the background-only distribution — ordered so
    /// that the limits they give grow (the "Brazil band").
    pub expected: [f64; 5],
}

/// An upper limit on the parameter of interest.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UpperLimit {
    /// The confidence level, e.g. `0.95`.
    pub cl: f64,
    /// The observed limit: where CLs crosses `1 − cl`.
    pub observed: f64,
    /// The expected limits: −2σ, −1σ, median, +1σ, +2σ.
    pub expected: [f64; 5],
}

/// A toy-based CLs result, with the toys' test-statistic distributions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToyCls {
    /// The CLs values (`q_asimov` is `NaN`).
    pub result: Cls,
    /// The statistical (binomial) uncertainty on `cls` from the toy count.
    pub cls_error: f64,
    /// The test statistic on each valid signal-plus-background toy.
    pub q_sb: Vec<f64>,
    /// The test statistic on each valid background-only toy.
    pub q_b: Vec<f64>,
}

/// A toy-based discovery test.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToyDiscovery {
    /// The best-fit parameter of interest.
    pub mu_hat: f64,
    /// The observed `q₀`.
    pub q0: f64,
    /// The fraction of background-only toys with `q₀` at least as large.
    pub p0: f64,
    /// Its binomial uncertainty.
    pub p0_error: f64,
    /// The significance of `p0` (infinite when no toy reached `q₀`: then
    /// `p₀ < 1/ntoys`).
    pub z: f64,
    /// `q₀` on each valid background-only toy.
    pub q0_toys: Vec<f64>,
}

/// An Asimov dataset's cost, its generating parameters and its free fit.
type AsimovFit<'c> = (Box<dyn Cost + 'c>, Vec<f64>, FitResult);

/// The CLs bands are at these background-only quantiles, in σ of `μ̂`.
const BANDS: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];

/// Hypothesis tests of a parameter of interest (POI) — a signal strength,
/// usually — on a [`Fitter`]'s cost: the [`discovery`](Self::discovery)
/// significance, [`cls`](Self::cls) at a value, an
/// [`upper_limit`](Self::upper_limit) with its expected band, and their
/// toy-based versions ([`cls_toys`](Self::cls_toys),
/// [`discovery_toys`](Self::discovery_toys)). The pyhf / RooStats recipe,
/// with the profile likelihood ratio as test statistic.
///
/// ```
/// use oxifit::{FitMethod, FitOptions, HypoTest, Model, ModelCost, Points};
///
/// // A three-bin counting experiment: expected signal s, background b.
/// let (s, b) = ([5.0, 10.0, 3.0], [50.0, 52.0, 40.0]);
/// let model = Model::new("sb", &["mu"], vec![1.0], move |x, p| {
///     let i = x as usize;
///     p[0] * s[i] + b[i]
/// })
/// .limit("mu", 0.0, 10.0);
/// let data = Points::unweighted(&[0.0, 1.0, 2.0], &[55.0, 62.0, 44.0])?;
/// let cost = ModelCost::new(&data, &model, &FitOptions::new().method(FitMethod::Likelihood));
/// let test = HypoTest::new(cost.fitter(), "mu")?;
/// let limit = test.upper_limit(0.95, (0.0, 5.0))?;
/// assert!(limit.observed > limit.expected[0] && limit.expected[0] < limit.expected[4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct HypoTest<'c> {
    fitter: Fitter<'c>,
    poi: usize,
    opts: FitOptions,
    stat: TestStatistic,
    /// The fit to the data with everything free.
    free: OnceLock<FitResult>,
    /// The fit to the data with the POI fixed at 0.
    background: OnceLock<FitResult>,
    /// The background-only Asimov dataset and its free fit.
    asimov_b: OnceLock<Option<AsimovFit<'c>>>,
}

impl std::fmt::Debug for HypoTest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HypoTest")
            .field("poi", &self.fitter.param_names()[self.poi])
            .field("stat", &self.stat)
            .finish_non_exhaustive()
    }
}

impl<'c> HypoTest<'c> {
    /// Tests of parameter `poi` of `fitter`'s cost, with the default fit
    /// options and the `q̃_μ` statistic for limits.
    ///
    /// # Errors
    /// [`FitError::UnknownParameter`] or [`FitError::FixedParameter`] for a
    /// POI that is not a free parameter.
    pub fn new(fitter: Fitter<'c>, poi: &str) -> Result<HypoTest<'c>, FitError> {
        let poi = crate::scan::free_index(&fitter.params, poi)?;
        Ok(HypoTest {
            fitter,
            poi,
            opts: FitOptions::new(),
            stat: TestStatistic::default(),
            free: OnceLock::new(),
            background: OnceLock::new(),
            asimov_b: OnceLock::new(),
        })
    }

    /// The fit options (minimizer, strategy, tolerance) for every fit the
    /// tests make; HESSE and MINOS are not run for the profiled fits.
    #[must_use]
    pub fn options(mut self, opts: &FitOptions) -> HypoTest<'c> {
        self.opts = *opts;
        self
    }

    /// The test statistic for [`cls`](Self::cls) and the limits.
    #[must_use]
    pub fn statistic(mut self, stat: TestStatistic) -> HypoTest<'c> {
        self.stat = stat;
        self
    }

    fn quiet(&self) -> FitOptions {
        FitOptions {
            hesse: false,
            minos: false,
            ..self.opts
        }
    }

    /// The fit to the data with every parameter free (HESSE as the options
    /// say).
    pub fn fit_free(&self) -> &FitResult {
        self.free.get_or_init(|| self.fitter.fit(&self.opts))
    }

    /// The fit to the data with the POI fixed at `mu` (the conditional fit,
    /// `θ̂̂(μ)`).
    #[must_use]
    pub fn fit_at(&self, mu: f64) -> FitResult {
        self.conditional(&self.fitter, self.fit_free(), mu)
    }

    fn background(&self) -> &FitResult {
        self.background.get_or_init(|| self.fit_at(0.0))
    }

    /// Refit `fitter` with the POI fixed at `mu`, seeded from `from`.
    fn conditional(&self, fitter: &Fitter, from: &FitResult, mu: f64) -> FitResult {
        let mut spec = fitter.params.clone();
        if from.params.len() == spec.params.len() && from.params.iter().all(|v| v.is_finite()) {
            spec.params.clone_from(&from.params);
        }
        spec.params[self.poi] = mu;
        spec.constraints[self.poi].fixed = true;
        CostRefit(fitter.cost).run(&spec, &self.quiet())
    }

    /// A fitter on `cost` with this test's settings, the Gaussian constraints'
    /// means at `at` (an Asimov dataset) or drawn around it (a toy, `rng`).
    fn derived<'x>(&self, cost: &'x dyn Cost, at: &[f64], rng: Option<&mut Rng>) -> Fitter<'x> {
        let mut spec = self.fitter.params.clone();
        spec.params = at.to_vec();
        let mut rng = rng;
        for (i, c) in spec.constraints.iter_mut().enumerate() {
            if let Some((_, sd)) = c.gauss {
                let shift = rng.as_mut().map_or(0.0, |r| sd * r.gauss());
                c.gauss = Some((at[i] + shift, sd));
            }
        }
        Fitter { cost, params: spec }
    }

    /// The upper-limit statistic for `mu` on `fitter`, given its free fit.
    fn q_mu(&self, fitter: &Fitter, free: &FitResult, mu: f64) -> f64 {
        if !free.valid {
            return f64::NAN;
        }
        let mu_hat = free.params[self.poi];
        if mu_hat > mu {
            return 0.0;
        }
        let cond = self.conditional(fitter, free, mu);
        if !cond.valid {
            return f64::NAN;
        }
        let reference = if self.stat == TestStatistic::QMuTilde && mu_hat < 0.0 {
            let zero = self.conditional(fitter, free, 0.0);
            if !zero.valid {
                return f64::NAN;
            }
            zero.fval
        } else {
            free.fval
        };
        (cond.fval - reference).max(0.0)
    }

    /// The discovery statistic `q₀` on `fitter`, given its free fit.
    fn q_zero(&self, fitter: &Fitter, free: &FitResult) -> f64 {
        if !free.valid {
            return f64::NAN;
        }
        if free.params[self.poi] < 0.0 {
            return 0.0;
        }
        let cond = self.conditional(fitter, free, 0.0);
        if cond.valid {
            (cond.fval - free.fval).max(0.0)
        } else {
            f64::NAN
        }
    }

    /// The observed upper-limit test statistic at `mu`.
    #[must_use]
    pub fn q(&self, mu: f64) -> f64 {
        self.q_mu(&self.fitter, self.fit_free(), mu)
    }

    /// The background-only Asimov dataset (the POI at 0, the other
    /// parameters at their conditional fit to the data), its generating
    /// parameters and its free fit.
    fn asimov_b(&self) -> Result<(&dyn Cost, &[f64], &FitResult), FitError> {
        let entry = self.asimov_b.get_or_init(|| {
            let bg = self.background();
            if !bg.valid {
                return None;
            }
            let at = bg.params.clone();
            let cost = self.fitter.cost.asimov(&at)?;
            let free = {
                let f = self.derived(&*cost, &at, None);
                f.fit(&self.quiet())
            };
            Some((cost, at, free))
        });
        match entry {
            Some((cost, at, free)) => Ok((&**cost, at, free)),
            None if !self.background().valid => Err(FitError::FitFailed),
            None => Err(FitError::NotGenerative),
        }
    }

    /// The test statistic at `mu` on the background-only Asimov dataset.
    fn q_asimov(&self, mu: f64) -> Result<f64, FitError> {
        let (cost, at, free) = self.asimov_b()?;
        let f = self.derived(cost, at, None);
        let q = self.q_mu(&f, free, mu);
        if q.is_nan() {
            Err(FitError::FitFailed)
        } else {
            Ok(q)
        }
    }

    /// `(CLs+b, CLb)` of an observed statistic `q` given its Asimov value.
    fn pvalues(&self, q: f64, qa: f64) -> (f64, f64) {
        let t = self.teststat(q, qa);
        (tail(t + qa.sqrt()), tail(t))
    }

    /// The statistic on pyhf's asymptotic scale: `√q − √q_A`, or for `q̃`
    /// beyond `q_A`, `(q − q_A)/(2√q_A)`.
    fn teststat(&self, q: f64, qa: f64) -> f64 {
        let (sq, sqa) = (q.max(0.0).sqrt(), qa.max(0.0).sqrt());
        if self.stat == TestStatistic::QMuTilde && sq >= sqa && sqa > 0.0 {
            (q - qa) / (2.0 * sqa)
        } else {
            sq - sqa
        }
    }

    /// The expected CLs bands given the Asimov statistic.
    fn expected(&self, qa: f64) -> [f64; 5] {
        let sqa = qa.max(0.0).sqrt();
        BANDS.map(|n| {
            // Under background only the statistic sits at −n (in σ); q_μ
            // cannot go below zero, i.e. −√q_A on this scale.
            let mut t = -n;
            if self.stat == TestStatistic::QMu {
                t = t.max(-sqa);
            }
            let (clsb, clb) = (tail(t + sqa), tail(t));
            clsb / clb
        })
    }

    /// Asymptotic CLs at `mu`: observed, with the expected band.
    ///
    /// # Errors
    /// [`FitError::NotGenerative`] if the cost cannot make an Asimov dataset;
    /// [`FitError::FitFailed`] if a fit fails.
    pub fn cls(&self, mu: f64) -> Result<Cls, FitError> {
        let q = self.q(mu);
        if q.is_nan() {
            return Err(FitError::FitFailed);
        }
        let qa = self.q_asimov(mu)?;
        let (clsb, clb) = self.pvalues(q, qa);
        Ok(Cls {
            mu,
            q,
            q_asimov: qa,
            clsb,
            clb,
            cls: clsb / clb,
            expected: self.expected(qa),
        })
    }

    /// The asymptotic upper limit at confidence level `cl` (e.g. `0.95`):
    /// where CLs — observed, and each expected band — crosses `1 − cl`,
    /// searched for inside `range`.
    ///
    /// # Errors
    /// [`FitError::NoCrossing`] if a CLs curve does not cross `1 − cl` inside
    /// `range`; as [`cls`](Self::cls) otherwise.
    pub fn upper_limit(&self, cl: f64, (lo, hi): (f64, f64)) -> Result<UpperLimit, FitError> {
        let alpha = 1.0 - cl;
        let observed = root(|mu| self.cls(mu).map(|c| c.cls - alpha), lo, hi)?;
        let mut expected = [f64::NAN; 5];
        for (k, e) in expected.iter_mut().enumerate() {
            *e = root(
                |mu| self.q_asimov(mu).map(|qa| self.expected(qa)[k] - alpha),
                lo,
                hi,
            )?;
        }
        Ok(UpperLimit {
            cl,
            observed,
            expected,
        })
    }

    /// The asymptotic discovery test, with the median significance expected
    /// for a signal of strength `mu_expected` (usually 1).
    ///
    /// # Errors
    /// [`FitError::NotGenerative`] if the cost cannot make an Asimov dataset;
    /// [`FitError::FitFailed`] if a fit fails.
    pub fn discovery(&self, mu_expected: f64) -> Result<Discovery, FitError> {
        let free = self.fit_free();
        let q0 = self.q_zero(&self.fitter, free);
        if q0.is_nan() {
            return Err(FitError::FitFailed);
        }
        // The Asimov dataset of the signal: the other parameters at their
        // conditional fit to the data at that strength.
        let at = self.fit_at(mu_expected);
        if !at.valid {
            return Err(FitError::FitFailed);
        }
        let cost = self
            .fitter
            .cost
            .asimov(&at.params)
            .ok_or(FitError::NotGenerative)?;
        let f = self.derived(&*cost, &at.params, None);
        let free_a = f.fit(&self.quiet());
        let q0a = self.q_zero(&f, &free_a);
        if q0a.is_nan() {
            return Err(FitError::FitFailed);
        }
        let p0 = tail(q0.sqrt());
        Ok(Discovery {
            mu_hat: free.params[self.poi],
            q0,
            p0,
            z: q0.sqrt(),
            mu_expected,
            z_expected: q0a.sqrt(),
        })
    }

    /// Fit each toy of `gen` (drawn at `at`, streams `first..first + n`) and
    /// return the statistic `stat` on the valid ones, in toy order.
    fn toy_stats(
        &self,
        at: &[f64],
        seed: u64,
        first: u64,
        n: usize,
        stat: &(dyn Fn(&Fitter, &FitResult) -> f64 + Sync),
    ) -> Result<Vec<f64>, FitError> {
        if self.fitter.cost.toy(at, 0).is_none() {
            return Err(FitError::NotGenerative);
        }
        let one = |i: usize| -> f64 {
            let mut rng = Rng::stream(seed, first + i as u64);
            let toy_seed = rng.next_u64();
            let Some(cost) = self.fitter.cost.toy(at, toy_seed) else {
                return f64::NAN;
            };
            let f = self.derived(&*cost, at, Some(&mut rng));
            let free = f.fit(&self.quiet());
            stat(&f, &free)
        };
        #[cfg(feature = "rayon")]
        let all: Vec<f64> = {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(one).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let all: Vec<f64> = (0..n).map(one).collect();
        Ok(all.into_iter().filter(|q| q.is_finite()).collect())
    }

    /// CLs at `mu` from `ntoys` signal-plus-background and `ntoys`
    /// background-only pseudo-experiments, generated at the conditional fits
    /// to the data at `mu` and at 0 (the generator seeded with `seed`; the
    /// same seed gives the same result, on any number of threads with the
    /// `rayon` feature). The expected band comes from the quantiles of the
    /// background-only toys.
    ///
    /// # Errors
    /// [`FitError::NotGenerative`] if the cost cannot make pseudo-datasets;
    /// [`FitError::FitFailed`] if a data fit fails or no toy fit succeeds.
    pub fn cls_toys(&self, mu: f64, ntoys: usize, seed: u64) -> Result<ToyCls, FitError> {
        let q = self.q(mu);
        let sb = self.fit_at(mu);
        let bg = self.background();
        if q.is_nan() || !sb.valid || !bg.valid {
            return Err(FitError::FitFailed);
        }
        let stat = |f: &Fitter, free: &FitResult| self.q_mu(f, free, mu);
        let q_sb = self.toy_stats(&sb.params, seed, 0, ntoys, &stat)?;
        let q_b = self.toy_stats(&bg.params, seed, ntoys as u64, ntoys, &stat)?;
        if q_sb.is_empty() || q_b.is_empty() {
            return Err(FitError::FitFailed);
        }
        let frac =
            |qs: &[f64], at: f64| qs.iter().filter(|&&v| v >= at).count() as f64 / qs.len() as f64;
        let (clsb, clb) = (frac(&q_sb, q), frac(&q_b, q));
        let cls = if clb > 0.0 { clsb / clb } else { f64::NAN };
        let mut sorted = q_b.clone();
        sorted.sort_by(f64::total_cmp);
        let expected = BANDS.map(|n| {
            // Band n: the background-only toys' statistic at the (1 − Φ(n))
            // quantile — a larger μ̂ gives a smaller statistic.
            let p = tail(n);
            let k = ((p * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
            let qn = sorted[k];
            let b = frac(&q_b, qn);
            if b > 0.0 {
                frac(&q_sb, qn) / b
            } else {
                f64::NAN
            }
        });
        let err_sb = (clsb * (1.0 - clsb) / q_sb.len() as f64).sqrt();
        let err_b = (clb * (1.0 - clb) / q_b.len() as f64).sqrt();
        let cls_error = if clb > 0.0 {
            cls * ((err_sb / clsb.max(1e-300)).powi(2) + (err_b / clb).powi(2)).sqrt()
        } else {
            f64::NAN
        };
        Ok(ToyCls {
            result: Cls {
                mu,
                q,
                q_asimov: f64::NAN,
                clsb,
                clb,
                cls,
                expected,
            },
            cls_error,
            q_sb,
            q_b,
        })
    }

    /// The discovery p-value from `ntoys` background-only pseudo-experiments
    /// (generated at the conditional fit to the data at 0).
    ///
    /// # Errors
    /// As [`cls_toys`](Self::cls_toys).
    pub fn discovery_toys(&self, ntoys: usize, seed: u64) -> Result<ToyDiscovery, FitError> {
        let free = self.fit_free();
        let q0 = self.q_zero(&self.fitter, free);
        let bg = self.background();
        if q0.is_nan() || !bg.valid {
            return Err(FitError::FitFailed);
        }
        let stat = |f: &Fitter, free: &FitResult| self.q_zero(f, free);
        let q0_toys = self.toy_stats(&bg.params, seed, 0, ntoys, &stat)?;
        if q0_toys.is_empty() {
            return Err(FitError::FitFailed);
        }
        let n = q0_toys.len() as f64;
        let p0 = q0_toys.iter().filter(|&&v| v >= q0).count() as f64 / n;
        Ok(ToyDiscovery {
            mu_hat: free.params[self.poi],
            q0,
            p0,
            p0_error: (p0 * (1.0 - p0) / n).sqrt(),
            z: z_of(p0),
            q0_toys,
        })
    }
}

/// The crossing of a decreasing `g` through zero inside `[lo, hi]`: Illinois
/// regula falsi.
fn root(g: impl Fn(f64) -> Result<f64, FitError>, lo: f64, hi: f64) -> Result<f64, FitError> {
    let (mut a, mut ga) = (lo, g(lo)?);
    let (mut b, mut gb) = (hi, g(hi)?);
    if ga.is_nan() || gb.is_nan() || ga.signum() == gb.signum() {
        return Err(FitError::NoCrossing);
    }
    let tol = 1e-6 * (hi - lo).abs();
    let mut last = 0;
    for _ in 0..100 {
        let m = (a * gb - b * ga) / (gb - ga);
        let gm = g(m)?;
        if gm == 0.0 || (b - a).abs() <= tol {
            return Ok(m);
        }
        if gm.signum() == ga.signum() {
            (a, ga) = (m, gm);
            if last == -1 {
                gb /= 2.0;
            }
            last = -1;
        } else {
            (b, gb) = (m, gm);
            if last == 1 {
                ga /= 2.0;
            }
            last = 1;
        }
        if gm.abs() < 1e-9 {
            return Ok(m);
        }
    }
    Ok((a * gb - b * ga) / (gb - ga))
}

/// The number of upcrossings of `level` in a scan `q` (each step from below
/// `level` to at or above it) — Gross and Vitells' `N(c₀)`.
#[must_use]
pub fn upcrossings(q: &[f64], level: f64) -> usize {
    q.windows(2)
        .filter(|w| w[0] < level && w[1] >= level)
        .count()
}

/// The global p-value of the largest local `q₀ = q_max` of a scan over a
/// parameter the background does not depend on (a signal mass), by Gross and
/// Vitells: `p_global ≈ p_local + ⟨N(c₀)⟩·e^{−(q_max − c₀)/2}`, with
/// `⟨N(c₀)⟩` the mean number of upcrossings of a low reference level `c₀`
/// (from the data scan, or better from a few background-only toy scans; see
/// [`upcrossings`]). Returns `(p_global, Z_global)`.
#[must_use]
pub fn global_significance(q_max: f64, level: f64, mean_upcrossings: f64) -> (f64, f64) {
    let p_local = tail(q_max.max(0.0).sqrt());
    let p = (p_local + mean_upcrossings * (-(q_max - level) / 2.0).exp()).min(1.0);
    (p, z_of(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upcrossings_and_global_significance() {
        let q = [0.0, 0.6, 0.2, 1.5, 0.1, 0.4, 0.9, 0.3];
        assert_eq!(upcrossings(&q, 0.5), 3);
        // No upcrossings: the global p-value is the local one.
        let (p, z) = global_significance(9.0, 0.5, 0.0);
        assert!((z - 3.0).abs() < 1e-9 && (p - tail(3.0)).abs() < 1e-15);
        let (p, _) = global_significance(9.0, 0.5, 4.0);
        assert!((p - (tail(3.0) + 4.0 * (-4.25f64).exp())).abs() < 1e-15);
    }

    #[test]
    fn tail_is_the_normal_upper_tail() {
        assert!((tail(0.0) - 0.5).abs() < 1e-15);
        assert!((tail(1.959_963_984_540_054) - 0.025).abs() < 1e-12);
        assert!((tail(5.0) - 2.866_515_718_791_939e-7).abs() < 1e-18);
    }
}
