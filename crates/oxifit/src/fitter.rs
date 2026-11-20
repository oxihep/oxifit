//! [`Fitter`]: minimize any [`Cost`] with oxifit's minimizers, errors, MINOS,
//! profiles and contours — iminuit's `Minuit` object.

use crate::cost::{Cost, model_settings};
use crate::engine::{self, DataTerm, Fit, GradCostFn};
use crate::model::Model;
use crate::result::{FitOptions, FitResult};
use crate::scan::{self, Contour, FitError, Profile, Refit};

/// Minimizes a [`Cost`]: the parameters' seeds and settings (limits, fixed
/// parameters, steps, Gaussian constraints), then [`fit`](Self::fit),
/// [`profile`](Self::profile) or [`contour`](Self::contour) with any
/// [`FitOptions`] minimizer — the same machinery as the data fits, on a cost
/// you wrote. The result's parameters are in the cost's
/// [`param_names`](Cost::param_names) order.
///
/// ```
/// use oxifit::{FitOptions, FnCost, Fitter};
///
/// // A Poisson count n = 7 on a known background b = 3.2: −2 ln L(s).
/// let (n, b) = (7.0_f64, 3.2);
/// let nll = FnCost::new(&["s"], move |p| 2.0 * ((p[0] + b) - n * (p[0] + b).ln()));
/// let fitter = Fitter::new(&nll, vec![1.0]).lower_limit("s", -b);
/// let fit = fitter.fit(&FitOptions::new().minos(true));
/// assert!((fit.params[0] - 3.8).abs() < 1e-3); // ŝ = n − b
/// let (lo, hi) = fit.minos.as_ref().unwrap()[0];
/// assert!(lo < 0.0 && hi > 0.0 && hi > -lo); // asymmetric Poisson interval
/// ```
#[derive(Clone)]
pub struct Fitter<'c> {
    pub(crate) cost: &'c dyn Cost,
    /// The parameters as a model whose function is unused: names, seeds and
    /// per-parameter settings, with the model setters' name checking.
    pub(crate) params: Model,
}

impl std::fmt::Debug for Fitter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fitter")
            .field("names", &self.params.param_names)
            .field("seeds", &self.params.params)
            .finish_non_exhaustive()
    }
}

impl<'c> Fitter<'c> {
    /// Minimize `cost` from `seeds`, one per [`param_names`](Cost::param_names)
    /// entry (a fit with a different number of seeds is invalid).
    #[must_use]
    pub fn new(cost: &'c dyn Cost, seeds: Vec<f64>) -> Fitter<'c> {
        let names = cost.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Fitter {
            cost,
            params: Model::new("cost", &refs, seeds, |_, _| 0.0),
        }
    }

    /// A fitter whose seeds and settings are `model`'s (its names must be the
    /// cost's).
    pub(crate) fn with_model_params(cost: &'c dyn Cost, model: &Model) -> Fitter<'c> {
        let (seeds, constraints) = model_settings(model);
        let mut f = Fitter::new(cost, seeds);
        f.params.constraints = constraints;
        f.params.unknown.clone_from(&model.unknown);
        f
    }

    /// The parameter names, in result order.
    #[must_use]
    pub fn param_names(&self) -> &[String] {
        &self.params.param_names
    }

    /// The seeds.
    #[must_use]
    pub fn seeds(&self) -> &[f64] {
        &self.params.params
    }

    /// Replace every seed.
    #[must_use]
    pub fn with_params(mut self, seeds: Vec<f64>) -> Fitter<'c> {
        self.params = self.params.with_params(seeds);
        self
    }

    /// Set one parameter's seed (or, with [`fix`](Self::fix), its value).
    #[must_use]
    pub fn set(mut self, name: &str, value: f64) -> Fitter<'c> {
        match self.params.param_names.iter().position(|n| n == name) {
            Some(i) if i < self.params.params.len() => self.params.params[i] = value,
            _ => self.params.unknown.push(name.to_string()),
        }
        self
    }

    /// Limit a parameter to `[lower, upper]`.
    #[must_use]
    pub fn limit(mut self, name: &str, lower: f64, upper: f64) -> Fitter<'c> {
        self.params = self.params.limit(name, lower, upper);
        self
    }

    /// Give a parameter a lower limit.
    #[must_use]
    pub fn lower_limit(mut self, name: &str, lower: f64) -> Fitter<'c> {
        self.params = self.params.lower_limit(name, lower);
        self
    }

    /// Give a parameter an upper limit.
    #[must_use]
    pub fn upper_limit(mut self, name: &str, upper: f64) -> Fitter<'c> {
        self.params = self.params.upper_limit(name, upper);
        self
    }

    /// Hold a parameter at its seed.
    #[must_use]
    pub fn fix(mut self, name: &str) -> Fitter<'c> {
        self.params = self.params.fix(name);
        self
    }

    /// Let a fixed parameter vary again.
    #[must_use]
    pub fn release(mut self, name: &str) -> Fitter<'c> {
        match self.params.param_names.iter().position(|n| n == name) {
            Some(i) => {
                if let Some(c) = self.params.constraints.get_mut(i) {
                    c.fixed = false;
                }
            }
            None => self.params.unknown.push(name.to_string()),
        }
        self
    }

    /// The minimizer's initial step for a parameter.
    #[must_use]
    pub fn step(mut self, name: &str, step: f64) -> Fitter<'c> {
        self.params = self.params.step(name, step);
        self
    }

    /// Add a Gaussian constraint `((p − mean)/sigma)²` on a parameter to the
    /// cost — an external measurement, or a nuisance parameter's prior.
    #[must_use]
    pub fn constrain(mut self, name: &str, mean: f64, sigma: f64) -> Fitter<'c> {
        self.params = self.params.constrain(name, mean, sigma);
        self
    }

    /// Whether the named parameter is fixed (`None`: no such parameter).
    #[must_use]
    pub fn is_fixed(&self, name: &str) -> Option<bool> {
        let i = self.params.param_names.iter().position(|n| n == name)?;
        Some(self.params.constraints.get(i).is_some_and(|c| c.fixed))
    }

    /// The cost at `params`, including the Gaussian constraints.
    #[must_use]
    pub fn value(&self, params: &[f64]) -> f64 {
        let penalty: f64 = self
            .params
            .constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.fixed)
            .filter_map(|(i, c)| c.gauss.map(|(m, s)| ((params[i] - m) / s).powi(2)))
            .sum();
        self.cost.value(params) + penalty
    }

    /// Minimize under `opts` (minimizer, strategy, tolerance, HESSE, MINOS,
    /// gradient use; the cost options — method, range, loss — belong to the
    /// costs themselves).
    #[must_use]
    pub fn fit(&self, opts: &FitOptions) -> FitResult {
        CostRefit(self.cost).run(&self.params, opts)
    }

    /// The profile of the cost in `param` over `range`, refitting the other
    /// parameters at each of `n` points. See [`Profile`].
    ///
    /// # Errors
    /// [`FitError`] for an unknown or fixed parameter, or a failed fit.
    pub fn profile(
        &self,
        param: &str,
        range: (f64, f64),
        n: usize,
        opts: &FitOptions,
    ) -> Result<Profile, FitError> {
        scan::profile(&CostRefit(self.cost), &self.params, opts, param, range, n)
    }

    /// The 2-D contour of `x` and `y` at `Δcost = delta`, with `npoints`
    /// points. See [`Contour`].
    ///
    /// # Errors
    /// [`FitError`] for an unknown or fixed parameter, or a failed fit.
    pub fn contour(
        &self,
        (x, y): (&str, &str),
        delta: f64,
        npoints: usize,
        opts: &FitOptions,
    ) -> Result<Contour, FitError> {
        scan::contour(
            &CostRefit(self.cost),
            &self.params,
            opts,
            (x, y),
            delta,
            npoints,
        )
    }
}

/// Refit a [`Cost`] under a parameter spec.
pub(crate) struct CostRefit<'c>(pub(crate) &'c dyn Cost);

impl CostRefit<'_> {
    /// Build the [`Fit`] for the cost under `params`' settings and run `body`.
    fn with_fit<R>(
        &self,
        params: &Model,
        opts: &FitOptions,
        body: impl FnOnce(&Fit) -> R,
    ) -> Option<(R, usize)> {
        let c = self.0;
        if params.params.len() != c.param_names().len() {
            return None;
        }
        let cost = |p: &[f64]| c.value(p);
        let residuals = |_: &[f64], _: &mut [f64]| {};
        let grad = |p: &[f64], out: &mut [f64]| c.gradient(p, out);
        let term = DataTerm {
            cost: &cost,
            residuals: &residuals,
            n_residuals: 0,
            n_points: c.n_measurements().unwrap_or(0),
            counted: c.n_measurements().is_some(),
            gof: c.is_gof(),
            cost_w2: None,
            scores: None,
            grad: c.has_gradient().then_some(&grad as GradCostFn<'_>),
            jacobian: None,
        };
        engine::with_terms(params, opts, &term, body)
    }
}

impl Refit for CostRefit<'_> {
    fn run(&self, params: &Model, opts: &FitOptions) -> FitResult {
        if params.check().is_err() {
            return engine::failed(params);
        }
        match self.with_fit(params, opts, |fit| engine::dispatch(fit, opts)) {
            Some((mut r, calls)) => {
                r.nfcn = calls;
                r
            }
            None => engine::failed(params),
        }
    }

    fn value(&self, params: &Model, opts: &FitOptions) -> f64 {
        self.with_fit(params, opts, |fit| (fit.cost)(&fit.seed()))
            .map_or(f64::NAN, |(f, _)| f)
    }
}
