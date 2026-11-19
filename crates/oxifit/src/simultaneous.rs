//! Simultaneous fits — RooFit's `RooSimultaneous`: several datasets, each with
//! its own model and cost, fitted together with the parameters they share.

use crate::data::{FitDataNd, PointNd};
use crate::engine::{self, CostFn, DataTerm, GradCostFn, Prepared};
use crate::model::{Constraint, Model};
use crate::result::{FitOptions, FitResult};

/// One dataset of a [`Simultaneous`] fit.
#[derive(Debug, Clone)]
struct Channel {
    name: String,
    points: Vec<PointNd>,
    dim: usize,
    model: Model,
    opts: FitOptions,
}

/// A simultaneous fit of several datasets ("channels"). Each channel has its
/// own data, model and cost options (method, range, robust loss); parameters
/// with the same name are the same parameter, fitted once across every
/// channel that uses it — a signal strength shared by several categories, a
/// mass seen in two decay modes, a calibration common to two samples.
///
/// The fit's parameters are every channel's, merged by name in order of first
/// appearance; a shared parameter's seed, limits, fixing, step and Gaussian
/// constraint come from the first channel that has it.
///
/// ```
/// # use oxifit::{FitExt, FitOptions, Model, Points, Simultaneous};
/// // Two samples of lines with a common slope "b" and their own intercepts.
/// let a = Points::new(&[0.0, 1.0, 2.0], &[1.0, 3.1, 4.9], &[0.1; 3])?;
/// let b = Points::new(&[0.0, 1.0, 2.0], &[5.0, 7.0, 9.1], &[0.1; 3])?;
/// let line = |c: &'static str| {
///     Model::new("line", &[c, "b"], vec![0.0, 1.0], |x, p| p[0] + p[1] * x)
/// };
/// let sim = Simultaneous::new()
///     .channel("A", &a, &line("c_a"), &FitOptions::new())
///     .channel("B", &b, &line("c_b"), &FitOptions::new());
/// assert_eq!(sim.param_names(), ["c_a", "b", "c_b"]);
/// let fit = sim.fit(&FitOptions::new());
/// assert!(fit.valid && (fit.params[1] - 2.0).abs() < 0.1);
/// # Ok::<(), oxifit::StatError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct Simultaneous {
    channels: Vec<Channel>,
}

impl Simultaneous {
    /// An empty simultaneous fit; add channels with [`channel`](Self::channel).
    #[must_use]
    pub fn new() -> Simultaneous {
        Simultaneous::default()
    }

    /// Add a channel: `data` fitted with `model` under `opts`'s cost settings
    /// (method, range, robust loss and its scale). The minimizer and its
    /// settings come from the options given to [`fit`](Self::fit).
    #[must_use]
    pub fn channel(
        mut self,
        name: &str,
        data: &(impl FitDataNd + ?Sized),
        model: &Model,
        opts: &FitOptions,
    ) -> Simultaneous {
        self.channels.push(Channel {
            name: name.to_string(),
            points: data.points_nd(),
            dim: data.dim(),
            model: model.clone(),
            opts: *opts,
        });
        self
    }

    /// The fit's parameter names: every channel's, merged by name in order of
    /// first appearance — the order of [`FitResult::params`].
    #[must_use]
    pub fn param_names(&self) -> Vec<String> {
        self.combined().0.param_names
    }

    /// The merged parameters (names, seeds, constraints) as a model whose
    /// function is unused, and each channel's map from its parameters to them.
    fn combined(&self) -> (Model, Vec<Vec<usize>>) {
        let mut names: Vec<String> = Vec::new();
        let mut params = Vec::new();
        let mut constraints: Vec<Constraint> = Vec::new();
        let maps = self
            .channels
            .iter()
            .map(|ch| {
                ch.model
                    .param_names
                    .iter()
                    .enumerate()
                    .map(|(k, name)| {
                        names.iter().position(|n| n == name).unwrap_or_else(|| {
                            names.push(name.clone());
                            params.push(ch.model.params.get(k).copied().unwrap_or(0.0));
                            constraints
                                .push(ch.model.constraints.get(k).copied().unwrap_or_default());
                            names.len() - 1
                        })
                    })
                    .collect()
            })
            .collect();
        let mut model = Model::new("simultaneous", &[], params, |_, _| 0.0);
        model.param_names = names;
        model.constraints = constraints;
        for ch in &self.channels {
            // A channel whose data and model differ in dimension.
            if ch.dim != ch.model.dim && model.dim_mismatch.is_none() {
                model.dim_mismatch = Some((ch.model.dim, ch.dim));
            }
            if model.dim_mismatch.is_none() {
                model.dim_mismatch = ch.model.dim_mismatch;
            }
            for u in &ch.model.unknown {
                if !model.unknown.contains(u) {
                    model.unknown.push(u.clone());
                }
            }
        }
        (model, maps)
    }

    /// Fit every channel together, with `opts`'s minimizer and its settings
    /// (strategy, tolerance, HESSE, MINOS, SumW2). The result's parameters are
    /// in [`param_names`](Self::param_names) order; its cost is the sum of the
    /// channels' (plus any constraints), `chi2` too when every channel's cost
    /// is a χ² or a binned likelihood (otherwise `NaN`), and `ndf` counts
    /// every channel's points.
    #[must_use]
    pub fn fit(&self, opts: &FitOptions) -> FitResult {
        let (model, maps) = self.combined();
        let preps: Vec<Prepared> = self
            .channels
            .iter()
            .map(|ch| Prepared::new(&ch.points, ch.dim, &ch.model, &ch.opts))
            .collect();
        let gather = |p: &[f64], map: &[usize]| map.iter().map(|&g| p[g]).collect::<Vec<f64>>();
        let cost = |p: &[f64]| {
            preps
                .iter()
                .zip(&maps)
                .map(|(prep, map)| prep.cost(&gather(p, map)))
                .sum::<f64>()
        };
        // Residuals only when every channel has them (no unbinned channel).
        let all_res = preps.iter().all(|p| p.n_residuals() == p.n_points());
        let residuals = |p: &[f64], out: &mut [f64]| {
            let mut start = 0;
            for (prep, map) in preps.iter().zip(&maps) {
                let n = prep.n_residuals();
                prep.residuals(&gather(p, map), &mut out[start..start + n]);
                start += n;
            }
        };
        let cost_w2 = |p: &[f64]| {
            preps
                .iter()
                .zip(&maps)
                .map(|(prep, map)| prep.cost_w2(&gather(p, map)))
                .sum::<f64>()
        };
        let sumw2 = opts.sumw2 && preps.iter().any(|p| p.unbinned() && p.weighted());
        // The joint gradient: each channel's, scattered onto the shared
        // parameters (a shared parameter collects every channel's term).
        let grad = |p: &[f64], out: &mut [f64]| {
            for (prep, map) in preps.iter().zip(&maps) {
                let mut g = vec![0.0; map.len()];
                prep.add_gradient(&gather(p, map), &mut g);
                for (&gi, v) in map.iter().zip(g) {
                    out[gi] += v;
                }
            }
        };
        let has_grad = preps.iter().all(Prepared::has_gradient);
        // The joint Jacobian: each channel's rows, its columns scattered onto
        // the shared parameters.
        let np = model.params.len();
        let jac = |p: &[f64], out: &mut [f64]| {
            let mut row0 = 0;
            for (prep, map) in preps.iter().zip(&maps) {
                let n = prep.n_residuals();
                let mut local = vec![0.0; n * map.len()];
                prep.residual_jacobian(&gather(p, map), &mut local);
                for r in 0..n {
                    for (c, &gi) in map.iter().enumerate() {
                        out[(row0 + r) * np + gi] += local[r * map.len() + c];
                    }
                }
                row0 += n;
            }
        };
        let n_points: usize = preps.iter().map(Prepared::n_points).sum();
        let term = DataTerm {
            cost: &cost,
            residuals: &residuals,
            n_residuals: if all_res { n_points } else { 0 },
            n_points,
            counted: true,
            gof: preps.iter().all(Prepared::gof),
            cost_w2: sumw2.then_some(&cost_w2 as CostFn<'_>),
            scores: None,
            grad: has_grad.then_some(&grad as GradCostFn<'_>),
            jacobian: (has_grad && all_res).then_some(&jac as GradCostFn<'_>),
        };
        engine::run_terms(&model, opts, &term)
    }

    /// Each channel's name, its cost at `result`'s parameters and its number
    /// of points — how the fit's cost splits across channels (each a χ² or
    /// `−2 ln L`). Constraint terms are not included.
    #[must_use]
    pub fn channel_costs(&self, result: &FitResult) -> Vec<(String, f64, usize)> {
        let (_, maps) = self.combined();
        self.channels
            .iter()
            .zip(&maps)
            .map(|(ch, map)| {
                let prep = Prepared::new(&ch.points, ch.dim, &ch.model, &ch.opts);
                let p: Vec<f64> = map.iter().map(|&g| result.params[g]).collect();
                (ch.name.clone(), prep.cost(&p), prep.n_points())
            })
            .collect()
    }
}
