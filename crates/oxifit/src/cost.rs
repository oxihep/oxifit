//! Costs as values: anything that maps named parameters to a `−2 ln L` (or a
//! χ²) — a data fit ([`ModelCost`]), a closure ([`FnCost`]), or a sum of
//! costs ([`CostSum`]) — minimized by a [`Fitter`](crate::Fitter).

use std::sync::Arc;

use crate::data::{FitDataNd, PointNd};
use crate::engine::Prepared;
use crate::model::{Constraint, Model};
use crate::result::FitOptions;

/// A fit cost: a function of named parameters whose minimum is the fit, on the
/// scale of a χ² — `−2 ln L` for a likelihood — so that a rise of 1 above the
/// minimum gives the 1σ errors, as everywhere in oxifit.
///
/// Implement it to minimize your own likelihood with every oxifit minimizer,
/// HESSE, MINOS, profiles and contours, through a [`Fitter`](crate::Fitter).
/// [`ModelCost`] is a data fit as a cost, [`FnCost`] a closure, and
/// [`CostSum`] adds costs, merging parameters by name.
///
/// ```
/// use oxifit::{Cost, Fitter, FitOptions};
///
/// /// −2 ln L of a Gaussian measurement of `mu`: x = 2.0 ± 0.5.
/// struct Measurement;
/// impl Cost for Measurement {
///     fn param_names(&self) -> Vec<String> {
///         vec!["mu".into()]
///     }
///     fn value(&self, p: &[f64]) -> f64 {
///         ((2.0 - p[0]) / 0.5).powi(2)
///     }
/// }
///
/// let fit = Fitter::new(&Measurement, vec![0.0]).fit(&FitOptions::new());
/// assert!((fit.params[0] - 2.0).abs() < 1e-6 && (fit.errors[0] - 0.5).abs() < 1e-6);
/// ```
pub trait Cost: Send + Sync {
    /// The parameter names, in the order [`value`](Self::value) takes them.
    fn param_names(&self) -> Vec<String>;

    /// The cost at `params` (one value per name): `−2 ln L` or a χ².
    fn value(&self, params: &[f64]) -> f64;

    /// Whether [`gradient`](Self::gradient) is implemented. With one, MIGRAD,
    /// HESSE and L-BFGS-B use it instead of finite differences.
    fn has_gradient(&self) -> bool {
        false
    }

    /// Write `∂cost/∂pₖ` at `params` into `out` (one entry per parameter).
    /// Called only when [`has_gradient`](Self::has_gradient) is `true`.
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        let _ = (params, out);
    }

    /// How many measurements the cost holds (data points, events, bins…), or
    /// `None` when it does not say. A count sets the fit's `ndf` and refuses a
    /// fit with fewer measurements than free parameters.
    fn n_measurements(&self) -> Option<usize> {
        None
    }

    /// Whether the cost at the minimum is a goodness-of-fit statistic (a χ²,
    /// or a binned Poisson likelihood ratio): then it is reported as the
    /// result's `chi2`.
    fn is_gof(&self) -> bool {
        false
    }
}

impl<C: Cost + ?Sized> Cost for &C {
    fn param_names(&self) -> Vec<String> {
        (**self).param_names()
    }
    fn value(&self, params: &[f64]) -> f64 {
        (**self).value(params)
    }
    fn has_gradient(&self) -> bool {
        (**self).has_gradient()
    }
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        (**self).gradient(params, out);
    }
    fn n_measurements(&self) -> Option<usize> {
        (**self).n_measurements()
    }
    fn is_gof(&self) -> bool {
        (**self).is_gof()
    }
}

impl<C: Cost + ?Sized> Cost for Box<C> {
    fn param_names(&self) -> Vec<String> {
        (**self).param_names()
    }
    fn value(&self, params: &[f64]) -> f64 {
        (**self).value(params)
    }
    fn has_gradient(&self) -> bool {
        (**self).has_gradient()
    }
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        (**self).gradient(params, out);
    }
    fn n_measurements(&self) -> Option<usize> {
        (**self).n_measurements()
    }
    fn is_gof(&self) -> bool {
        (**self).is_gof()
    }
}

type ValueFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;
type GradientFn = Arc<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

/// A cost from a closure over the parameter vector — iminuit's
/// `Minuit(fcn, …)`. Like every [`Cost`] it returns `−2 ln L` or a χ².
///
/// ```
/// use oxifit::{FitOptions, FnCost, Fitter};
///
/// // The Rosenbrock function, minimum at (1, 1).
/// let rosen = FnCost::new(&["x", "y"], |p| {
///     (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2)
/// });
/// let fit = Fitter::new(&rosen, vec![-1.0, 2.0]).fit(&FitOptions::new());
/// assert!((fit.params[0] - 1.0).abs() < 1e-3 && (fit.params[1] - 1.0).abs() < 1e-3);
/// ```
#[derive(Clone)]
pub struct FnCost {
    names: Vec<String>,
    value: ValueFn,
    gradient: Option<GradientFn>,
    n_measurements: Option<usize>,
    gof: bool,
}

impl std::fmt::Debug for FnCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnCost")
            .field("names", &self.names)
            .field("gradient", &self.gradient.is_some())
            .field("n_measurements", &self.n_measurements)
            .finish_non_exhaustive()
    }
}

impl FnCost {
    /// A cost of the named parameters from `value(params)`.
    pub fn new(names: &[&str], value: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> FnCost {
        FnCost {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            value: Arc::new(value),
            gradient: None,
            n_measurements: None,
            gof: false,
        }
    }

    /// Supply the exact gradient: `gradient(params, out)` writes
    /// `∂cost/∂pₖ` into `out[k]`.
    #[must_use]
    pub fn with_gradient(
        mut self,
        gradient: impl Fn(&[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> FnCost {
        self.gradient = Some(Arc::new(gradient));
        self
    }

    /// Say how many measurements the cost holds (see
    /// [`Cost::n_measurements`]), and whether its minimum is a
    /// goodness-of-fit statistic (see [`Cost::is_gof`]).
    #[must_use]
    pub fn measurements(mut self, n: usize, gof: bool) -> FnCost {
        self.n_measurements = Some(n);
        self.gof = gof;
        self
    }
}

impl Cost for FnCost {
    fn param_names(&self) -> Vec<String> {
        self.names.clone()
    }
    fn value(&self, params: &[f64]) -> f64 {
        (self.value)(params)
    }
    fn has_gradient(&self) -> bool {
        self.gradient.is_some()
    }
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        if let Some(g) = &self.gradient {
            g(params, out);
        }
    }
    fn n_measurements(&self) -> Option<usize> {
        self.n_measurements
    }
    fn is_gof(&self) -> bool {
        self.gof
    }
}

/// A data fit as a [`Cost`]: `data` fitted with `model` under `opts`'s cost
/// settings (method, range, robust loss) — the cost
/// [`FitExt::fit_opts`](crate::FitExt::fit_opts) minimizes, without the
/// model's Gaussian constraints (a [`Fitter`](crate::Fitter) adds its own; see
/// [`fitter`](Self::fitter)).
///
/// As a value it can be added to other costs ([`CostSum`]), evaluated
/// anywhere ([`Cost::value`]), and handed to the inference tools.
///
/// ```
/// use oxifit::{Cost, FitOptions, FnCost, CostSum, Model, ModelCost, Points};
///
/// let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.1, 2.9, 5.2, 6.8], &[0.2; 4])?;
/// let line = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
/// let chi2 = ModelCost::new(&data, &line, &FitOptions::new());
/// // An external measurement of the slope, 2.1 ± 0.1, added to the fit.
/// let slope = FnCost::new(&["p1"], |p| ((p[0] - 2.1) / 0.1).powi(2));
/// let total = CostSum::new().with(chi2).with(slope);
/// assert_eq!(total.param_names(), ["p0", "p1"]);
/// let fit = total.fitter(vec![0.0, 1.0]).fit(&FitOptions::new());
/// assert!(fit.valid && (fit.params[1] - 2.0).abs() < 0.1);
/// # Ok::<(), oxifit::StatError>(())
/// ```
#[derive(Clone)]
pub struct ModelCost {
    names: Vec<String>,
    model: Model,
    opts: FitOptions,
    points: Vec<PointNd>,
    dim: usize,
    prep: Prepared,
}

impl std::fmt::Debug for ModelCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelCost")
            .field("model", &self.model.name)
            .field("names", &self.names)
            .field("method", &self.opts.method)
            .field("points", &self.prep.n_points())
            .finish_non_exhaustive()
    }
}

impl ModelCost {
    /// The cost of fitting `data` with `model` under `opts` (its method,
    /// ranges and loss; the minimizer settings are the [`Fitter`](crate::Fitter)'s).
    pub fn new(data: &(impl FitDataNd + ?Sized), model: &Model, opts: &FitOptions) -> ModelCost {
        ModelCost::from_points(data.points_nd(), data.dim(), model, opts)
    }

    pub(crate) fn from_points(
        points: Vec<PointNd>,
        dim: usize,
        model: &Model,
        opts: &FitOptions,
    ) -> ModelCost {
        let prep = Prepared::new(&points, dim, model, opts);
        ModelCost {
            names: model.param_names.clone(),
            model: model.clone(),
            opts: *opts,
            points,
            dim,
            prep,
        }
    }

    /// The model.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The cost options (method, ranges, loss).
    #[must_use]
    pub fn options(&self) -> &FitOptions {
        &self.opts
    }

    /// The data points (before the options' range selection).
    #[must_use]
    pub fn points(&self) -> &[PointNd] {
        &self.points
    }

    /// The number of variables of the data.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// A [`Fitter`](crate::Fitter) on this cost with the model's seeds,
    /// limits, fixed parameters, steps and Gaussian constraints — the same fit
    /// as [`FitExt::fit_opts`](crate::FitExt::fit_opts).
    #[must_use]
    pub fn fitter(&self) -> crate::Fitter<'_> {
        crate::Fitter::with_model_params(self, &self.model)
    }

    /// The same model and options on other data — a toy or an Asimov dataset.
    #[must_use]
    pub fn with_points(&self, points: Vec<PointNd>) -> ModelCost {
        ModelCost::from_points(points, self.dim, &self.model, &self.opts)
    }
}

impl Cost for ModelCost {
    fn param_names(&self) -> Vec<String> {
        self.names.clone()
    }
    fn value(&self, params: &[f64]) -> f64 {
        self.prep.cost(params)
    }
    fn has_gradient(&self) -> bool {
        self.prep.has_gradient()
    }
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        self.prep.add_gradient(params, out);
    }
    fn n_measurements(&self) -> Option<usize> {
        Some(self.prep.n_points())
    }
    fn is_gof(&self) -> bool {
        self.prep.gof()
    }
}

/// A sum of costs, their parameters merged by name in order of first
/// appearance: parameters with the same name are one parameter. Adding a data
/// cost and an external measurement, or the costs of several datasets with
/// shared parameters (what [`Simultaneous`](crate::Simultaneous) does), is
/// this sum. See [`ModelCost`] for an example.
#[derive(Default)]
pub struct CostSum<'a> {
    names: Vec<String>,
    terms: Vec<(Box<dyn Cost + 'a>, Vec<usize>)>,
}

impl std::fmt::Debug for CostSum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostSum")
            .field("names", &self.names)
            .field("terms", &self.terms.len())
            .finish()
    }
}

impl<'a> CostSum<'a> {
    /// An empty sum (cost 0, no parameters).
    #[must_use]
    pub fn new() -> CostSum<'a> {
        CostSum {
            names: Vec::new(),
            terms: Vec::new(),
        }
    }

    /// Add a cost; its parameters join the sum's by name.
    #[must_use]
    pub fn with(mut self, cost: impl Cost + 'a) -> CostSum<'a> {
        let map = cost
            .param_names()
            .into_iter()
            .map(|name| {
                self.names
                    .iter()
                    .position(|n| *n == name)
                    .unwrap_or_else(|| {
                        self.names.push(name);
                        self.names.len() - 1
                    })
            })
            .collect();
        self.terms.push((Box::new(cost), map));
        self
    }

    /// The number of costs added.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether no cost was added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// A [`Fitter`](crate::Fitter) on the sum from `seeds` (one per name).
    #[must_use]
    pub fn fitter(&self, seeds: Vec<f64>) -> crate::Fitter<'_> {
        crate::Fitter::new(self, seeds)
    }
}

fn gather(p: &[f64], map: &[usize]) -> Vec<f64> {
    map.iter().map(|&g| p[g]).collect()
}

impl Cost for CostSum<'_> {
    fn param_names(&self) -> Vec<String> {
        self.names.clone()
    }
    fn value(&self, params: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|(c, map)| c.value(&gather(params, map)))
            .sum()
    }
    fn has_gradient(&self) -> bool {
        self.terms.iter().all(|(c, _)| c.has_gradient())
    }
    fn gradient(&self, params: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        for (c, map) in &self.terms {
            let mut g = vec![0.0; map.len()];
            c.gradient(&gather(params, map), &mut g);
            for (&i, v) in map.iter().zip(g) {
                out[i] += v;
            }
        }
    }
    fn n_measurements(&self) -> Option<usize> {
        self.terms.iter().map(|(c, _)| c.n_measurements()).sum()
    }
    fn is_gof(&self) -> bool {
        !self.terms.is_empty() && self.terms.iter().all(|(c, _)| c.is_gof())
    }
}

/// Merge a model's per-parameter settings into `(names, seeds, constraints)`
/// order: used by [`ModelCost::fitter`].
pub(crate) fn model_settings(model: &Model) -> (Vec<f64>, Vec<Constraint>) {
    let mut c = model.constraints.clone();
    c.resize(model.params.len(), Constraint::default());
    (model.params.clone(), c)
}
