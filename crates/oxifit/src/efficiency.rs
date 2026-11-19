//! Efficiency fits with the binomial likelihood: a model `ε(x)` in [0, 1]
//! fitted to pass/total counts per bin, or to individual pass/fail events —
//! ROOT's `TEfficiency::Fit`, RooFit's `RooEfficiency`.

use oxistats::StatError;

use crate::cost::Cost;
use crate::fitter::Fitter;
use crate::model::Model;
use crate::result::{FitOptions, FitResult};
use crate::rng::Rng;

/// Efficiency data: at each `x`, `passed` of `total` trials — the bins of a
/// pass/total histogram pair, or single events (`total` 1, `passed` 0 or 1).
///
/// A χ² fit to `passed/total` with binomial errors fails where the
/// efficiency is near 0 or 1 or a bin has few trials (the error vanishes or
/// is undefined); the binomial likelihood is right everywhere.
///
/// ```
/// use oxifit::{Efficiency, FitOptions, Model};
///
/// // A turn-on: ε = ½·(1 + erf((x − 20)/(√2·3))), seen in 30 bins.
/// let x: Vec<f64> = (0..30).map(|i| 5.0 + i as f64).collect();
/// let total = vec![40.0; 30];
/// let passed: Vec<f64> = x
///     .iter()
///     .map(|&x| (40.0 * 0.5 * (1.0 + oxistats::erf((x - 20.0) / (2f64.sqrt() * 3.0)))).round())
///     .collect();
/// let data = Efficiency::new(&x, &passed, &total)?;
/// let turn_on = Model::new("eff", &["threshold", "width"], vec![18.0, 2.0], |x, p| {
///     0.5 * (1.0 + oxistats::erf((x - p[0]) / (2f64.sqrt() * p[1])))
/// })
/// .lower_limit("width", 0.1);
/// let fit = data.fit(&turn_on, &FitOptions::new());
/// assert!(fit.valid && (fit.params[0] - 20.0).abs() < 0.3);
/// # Ok::<(), oxistats::StatError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Efficiency {
    x: Vec<f64>,
    passed: Vec<f64>,
    total: Vec<f64>,
}

impl Efficiency {
    /// Binned efficiencies: `passed[i]` of `total[i]` trials at `x[i]`.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] for slices of different lengths.
    pub fn new(x: &[f64], passed: &[f64], total: &[f64]) -> Result<Efficiency, StatError> {
        if passed.len() != x.len() {
            return Err(StatError::LengthMismatch {
                left: x.len(),
                right: passed.len(),
            });
        }
        if total.len() != x.len() {
            return Err(StatError::LengthMismatch {
                left: x.len(),
                right: total.len(),
            });
        }
        Ok(Efficiency {
            x: x.to_vec(),
            passed: passed.to_vec(),
            total: total.to_vec(),
        })
    }

    /// Unbinned: one event per `x`, `accepted` or not.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] for slices of different lengths.
    pub fn events(x: &[f64], accepted: &[bool]) -> Result<Efficiency, StatError> {
        let passed: Vec<f64> = accepted.iter().map(|&a| f64::from(u8::from(a))).collect();
        Efficiency::new(x, &passed, &vec![1.0; x.len()])
    }

    /// The number of points (bins or events).
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether there are no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Fit the efficiency model `model` (values in [0, 1]) with the binomial
    /// likelihood, inside `opts.range` (the cost options' method and loss do
    /// not apply).
    #[must_use]
    pub fn fit(&self, model: &Model, opts: &FitOptions) -> FitResult {
        let cost = BinomialCost::new(self, model, opts);
        cost.fitter().fit(opts)
    }
}

/// The binomial `−2 ln L` of an efficiency model on [`Efficiency`] data, as a
/// [`Cost`] (on the deviance scale: zero for a perfect fit, so its minimum is
/// a goodness-of-fit statistic for binned data).
#[derive(Clone)]
pub struct BinomialCost {
    model: Model,
    x: Vec<f64>,
    passed: Vec<f64>,
    total: Vec<f64>,
    range: Option<(f64, f64)>,
}

impl std::fmt::Debug for BinomialCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinomialCost")
            .field("model", &self.model.name)
            .field("points", &self.x.len())
            .finish_non_exhaustive()
    }
}

/// The smallest and largest efficiency the likelihood takes.
const CLAMP: f64 = 1e-12;

impl BinomialCost {
    /// The cost of `model` on `data`, inside `opts.range` (if any).
    #[must_use]
    pub fn new(data: &Efficiency, model: &Model, opts: &FitOptions) -> BinomialCost {
        let keep: Vec<usize> = (0..data.len())
            .filter(|&i| {
                data.total[i] > 0.0
                    && opts
                        .range
                        .is_none_or(|(lo, hi)| (lo..=hi).contains(&data.x[i]))
            })
            .collect();
        BinomialCost {
            model: model.clone(),
            x: keep.iter().map(|&i| data.x[i]).collect(),
            passed: keep.iter().map(|&i| data.passed[i]).collect(),
            total: keep.iter().map(|&i| data.total[i]).collect(),
            range: opts.range,
        }
    }

    /// A [`Fitter`] with the model's seeds and settings.
    #[must_use]
    pub fn fitter(&self) -> Fitter<'_> {
        Fitter::with_model_params(self, &self.model)
    }

    fn with_passed(&self, passed: Vec<f64>) -> BinomialCost {
        BinomialCost {
            passed,
            ..self.clone()
        }
    }
}

/// `k·ln(k/m)`, zero at `k = 0`.
fn xlogx(k: f64, m: f64) -> f64 {
    if k > 0.0 { k * (k / m).ln() } else { 0.0 }
}

impl Cost for BinomialCost {
    fn param_names(&self) -> Vec<String> {
        self.model.param_names.clone()
    }

    fn value(&self, params: &[f64]) -> f64 {
        let f = &self.model.func;
        2.0 * self
            .x
            .iter()
            .zip(&self.passed)
            .zip(&self.total)
            .map(|((&x, &k), &n)| {
                let e = f(&[x], params).clamp(CLAMP, 1.0 - CLAMP);
                xlogx(k, n * e) + xlogx(n - k, n * (1.0 - e))
            })
            .sum::<f64>()
    }

    fn has_gradient(&self) -> bool {
        self.model.grad.is_some()
    }

    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        let Some(grad) = self.model.grad.as_deref() else {
            return;
        };
        let f = &self.model.func;
        let mut g = vec![0.0; out.len()];
        for ((&x, &k), &n) in self.x.iter().zip(&self.passed).zip(&self.total) {
            let raw = f(&[x], params);
            if !(CLAMP..=1.0 - CLAMP).contains(&raw) {
                continue;
            }
            // d/dε of 2[k·ln(k/nε) + (n−k)·ln((n−k)/(n(1−ε)))].
            let slope = 2.0 * (-k / raw + (n - k) / (1.0 - raw));
            g.iter_mut().for_each(|v| *v = 0.0);
            grad(&[x], params, &mut g);
            for (o, gk) in out.iter_mut().zip(&g) {
                *o += slope * gk;
            }
        }
    }

    fn n_measurements(&self) -> Option<usize> {
        Some(self.x.len())
    }

    fn is_gof(&self) -> bool {
        // Single events (n = 1) have no saturated model to compare with.
        self.total.iter().any(|&n| n > 1.0)
    }

    fn asimov(&self, params: &[f64]) -> Option<Box<dyn Cost + '_>> {
        let f = &self.model.func;
        let passed = self
            .x
            .iter()
            .zip(&self.total)
            .map(|(&x, &n)| n * f(&[x], params).clamp(0.0, 1.0))
            .collect();
        Some(Box::new(self.with_passed(passed)))
    }

    fn toy(&self, params: &[f64], seed: u64) -> Option<Box<dyn Cost + '_>> {
        let f = &self.model.func;
        let mut rng = Rng::stream(seed, 0);
        let passed = self
            .x
            .iter()
            .zip(&self.total)
            .map(|(&x, &n)| rng.binomial(n.round() as u64, f(&[x], params).clamp(0.0, 1.0)) as f64)
            .collect();
        Some(Box::new(self.with_passed(passed)))
    }
}

impl BinomialCost {
    /// The fit range the cost was built with.
    #[must_use]
    pub fn range(&self) -> Option<(f64, f64)> {
        self.range
    }
}
