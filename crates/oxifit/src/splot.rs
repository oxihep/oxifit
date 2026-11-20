//! sWeights (sPlot): per-event weights that unfold the species of an extended
//! unbinned fit in one (discriminating) variable, so another (control)
//! variable can be studied for each species alone — Pivk and Le Diberder,
//! NIM A 555 (2005) 356.

use crate::data::FitDataNd;
use crate::engine::{integrate_region, linalg};
use crate::model::Model;
use crate::result::FitOptions;
use crate::scan::FitError;

/// sWeights for every event, one set per species.
///
/// From the species' shapes in the discriminating variable (fixed at their
/// fitted values) and their fitted yields `Nₖ`:
///
/// `V⁻¹ₙⱼ = Σₑ pₙ(xₑ)·pⱼ(xₑ) / (Σₖ Nₖ·pₖ(xₑ))²`,
/// `sWₙ(e) = Σⱼ Vₙⱼ·pⱼ(xₑ) / Σₖ Nₖ·pₖ(xₑ)`,
///
/// `pₖ` each shape normalized to unit area over the fit range. A species'
/// weights sum to its yield; at the extended fit's minimum the weights of an
/// event sum to one. Fit the control variable with the weights (an
/// unbinned likelihood with
/// [`asymptotic_errors`](FitOptions::asymptotic_errors) for its
/// covariance) to measure a species alone. The control variable must be
/// independent of the discriminating one within each species.
///
/// ```
/// use oxifit::{Events, FitExt, FitMethod, FitOptions, Model, SWeights};
///
/// // Masses: a Gaussian peak on a flat background in [5.2, 5.4].
/// let masses: Vec<f64> = (0..600)
///     .map(|i| {
///         let u = (f64::from(i) * 0.618_034).fract();
///         if i % 4 == 0 { 5.3 + 0.01 * (u - 0.5) * 6.0 } else { 5.2 + 0.2 * u }
///     })
///     .collect();
/// let events = Events::new(&masses);
/// let opts = FitOptions::new().range(5.2, 5.4);
/// let gauss = |m: f64| (-0.5 * ((m - 5.3) / 0.01f64).powi(2)).exp() / (0.01 * 2.506_628);
/// // The extended fit of the yields, the shapes fixed…
/// let model = Model::new("m", &["ns", "nb"], vec![100.0, 400.0], move |m, p| {
///     p[0] * gauss(m) + p[1] / 0.2
/// });
/// let fit = events.fit_opts(&model, &opts.method(FitMethod::ExtendedUnbinnedLikelihood));
/// // …then the weights, from the same shapes and the fitted yields.
/// let signal = Model::new("sig", &[], vec![], move |m, _| gauss(m));
/// let background = Model::new("bkg", &[], vec![], |_, _| 1.0);
/// let sw = SWeights::compute(&events, &[&signal, &background], &fit.params, &opts)?;
/// let total: f64 = sw.weights(0).iter().sum();
/// assert!((total - fit.params[0]).abs() < 0.01 * fit.errors[0]);
/// # Ok::<(), oxifit::FitError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SWeights {
    weights: Vec<Vec<f64>>,
    covariance: Vec<Vec<f64>>,
    yields: Vec<f64>,
}

impl SWeights {
    /// sWeights for the events of `data`, from the species' `components`
    /// (shapes in the discriminating variable, at their fitted parameters)
    /// and fitted `yields`, normalized over `opts`' range (and `range_y`,
    /// `range_z`; else the events' span). Weights are given for every event,
    /// in the data's order: give only events inside the fit range.
    ///
    /// # Errors
    /// [`FitError::DimensionMismatch`] if a component's dimension is not the
    /// data's or the yields are not one per component;
    /// [`FitError::FitFailed`] if a shape does not normalize or the matrix
    /// `V⁻¹` is singular.
    pub fn compute(
        data: &(impl FitDataNd + ?Sized),
        components: &[&Model],
        yields: &[f64],
        opts: &FitOptions,
    ) -> Result<SWeights, FitError> {
        let dim = data.dim();
        if yields.len() != components.len() {
            return Err(FitError::DimensionMismatch {
                expected: components.len(),
                found: yields.len(),
            });
        }
        if let Some(c) = components.iter().find(|c| c.dim != dim) {
            return Err(FitError::DimensionMismatch {
                expected: dim,
                found: c.dim,
            });
        }
        let points = data.points_nd();
        let ranges = [opts.range, opts.range_y, opts.range_z];
        let region: Vec<(f64, f64)> = (0..dim)
            .map(|k| {
                ranges[k].unwrap_or_else(|| {
                    points
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                            (a.min(p.at[k]), b.max(p.at[k]))
                        })
                })
            })
            .collect();
        let norms: Vec<f64> = components
            .iter()
            .map(|c| integrate_region(&|x: &[f64]| (c.func)(x, &c.params), &region))
            .collect();
        if norms.iter().any(|n| !(n.is_finite() && *n > 0.0)) {
            return Err(FitError::FitFailed);
        }
        let n = components.len();
        // pₖ(xₑ) for every event.
        let densities: Vec<Vec<f64>> = points
            .iter()
            .map(|p| {
                components
                    .iter()
                    .zip(&norms)
                    .map(|(c, norm)| (c.func)(&p.at[..dim], &c.params) / norm)
                    .collect()
            })
            .collect();
        let mut vinv = vec![vec![0.0; n]; n];
        for d in &densities {
            let total: f64 = d.iter().zip(yields).map(|(p, y)| p * y).sum();
            if total <= 0.0 {
                continue;
            }
            for a in 0..n {
                for b in 0..n {
                    vinv[a][b] += d[a] * d[b] / (total * total);
                }
            }
        }
        let v = linalg::invert(&vinv).ok_or(FitError::FitFailed)?;
        let weights = (0..n)
            .map(|k| {
                densities
                    .iter()
                    .map(|d| {
                        let total: f64 = d.iter().zip(yields).map(|(p, y)| p * y).sum();
                        if total <= 0.0 {
                            return 0.0;
                        }
                        (0..n).map(|j| v[k][j] * d[j]).sum::<f64>() / total
                    })
                    .collect()
            })
            .collect();
        Ok(SWeights {
            weights,
            covariance: v,
            yields: yields.to_vec(),
        })
    }

    /// Species `k`'s weight for each event (in the data's order).
    ///
    /// # Panics
    /// If `k` is not a species index.
    #[must_use]
    pub fn weights(&self, k: usize) -> &[f64] {
        &self.weights[k]
    }

    /// The yields' covariance `V` the weights are built from (with the shapes
    /// fixed).
    #[must_use]
    pub fn covariance(&self) -> &[Vec<f64>] {
        &self.covariance
    }

    /// The yields the weights were computed for.
    #[must_use]
    pub fn yields(&self) -> &[f64] {
        &self.yields
    }
}
