//! [`Model`] — a named parametric fit function `f(x, params)` with optional
//! per-parameter limits, fixing, and step hints, plus the built-in shapes
//! (Gaussian / exponential / polynomial).

use crate::data::{FitData, Point};

/// The most intervals one convolution integral may use.
const CONV_INTERVALS: usize = 200;

/// A model evaluator: `f(coords, params) -> y`, `coords` holding the model's
/// 1–3 variables. `Send + Sync` so a [`Model`] can cross thread boundaries
/// (e.g. be shared by a parallel fit); an `Arc` so [`Model`] is cheaply `Clone`
/// (clones share the immutable closure).
pub(crate) type ModelFn = std::sync::Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>;

/// A model's gradient: `g(coords, params, out)` writes `∂f/∂pₖ` into `out[k]`.
pub(crate) type GradFn = std::sync::Arc<dyn Fn(&[f64], &[f64], &mut [f64]) + Send + Sync>;

/// A model's value and gradient in one evaluation: `vg(coords, params, out)`
/// writes `∂f/∂pₖ` into `out[k]` and returns `f` — what a gradient pass needs,
/// without computing the shared factors (an exponential, a power) twice.
pub(crate) type ValGradFn = std::sync::Arc<dyn Fn(&[f64], &[f64], &mut [f64]) -> f64 + Send + Sync>;

/// A model's integral over a region — one `(lo, hi)` per variable — at the
/// given parameters, when it is known without integrating numerically (a
/// product of 1-D factors, a sum of such).
pub(crate) type IntegralFn = std::sync::Arc<dyn Fn(&[f64], &[(f64, f64)]) -> f64 + Send + Sync>;

/// Optional fit constraints on one parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Constraint {
    pub(crate) lower: Option<f64>,
    pub(crate) upper: Option<f64>,
    pub(crate) fixed: bool,
    pub(crate) step: Option<f64>,
    /// A Gaussian constraint `(mean, sigma)`: the fit adds `((p − mean)/sigma)²`.
    pub(crate) gauss: Option<(f64, f64)>,
}

/// A parametric fit function: a closure `f(x, params)` plus named parameters
/// with their current/initial values and optional per-parameter limits, fixing,
/// and step hints. Build one from a closure ([`new`](Model::new)), a built-in
/// shape, or a formula string ([`from_formula`](Model::from_formula)).
#[derive(Clone)]
pub struct Model {
    /// Function name.
    pub name: String,
    /// Parameter names (used as the Minuit2 parameter labels).
    pub param_names: Vec<String>,
    /// Current parameter values; these seed the fit.
    pub params: Vec<f64>,
    /// Per-parameter constraints (limits / fixed / step), one per parameter.
    pub(crate) constraints: Vec<Constraint>,
    pub(crate) func: ModelFn,
    /// The analytic gradient `∂f/∂p`, when known
    /// ([`with_gradient`](Model::with_gradient)).
    pub(crate) grad: Option<GradFn>,
    /// The value and gradient fused, for the built-in shapes and sums of
    /// models with gradients; `None` otherwise (the engine then calls
    /// `func` and `grad`). Cleared whenever `func` or `grad` is replaced.
    pub(crate) value_grad: Option<ValGradFn>,
    /// For a sum ([`Model::sum`]): its parts' names, in order, so `+` can
    /// append to it rather than nest it.
    pub(crate) parts: Option<Vec<String>>,
    /// Names given to [`limit`](Model::limit), [`fix`](Model::fix) and the
    /// other per-parameter setters that match no parameter — a typo would
    /// otherwise be silently ignored. See [`check`](Model::check).
    pub(crate) unknown: Vec<String>,
    /// The number of variables, 1–3.
    pub(crate) dim: usize,
    /// A part of another dimension where one was required (in a sum, a
    /// product or a convolution): `(expected, found)`. See [`check`](Model::check).
    pub(crate) dim_mismatch: Option<(usize, usize)>,
    /// The integral over a region, when known without integrating
    /// numerically (products, sums of known integrals).
    pub(crate) integral: Option<IntegralFn>,
}

/// A model's numerical integral over a region, remembered for the last
/// parameters and region it was asked for (shared by clones of the model;
/// thread-safe, so parallel fits only lose hits).
fn remembered_integral(func: ModelFn) -> IntegralFn {
    type Memo = Option<(Vec<f64>, Vec<(f64, f64)>, f64)>;
    let memo: std::sync::Arc<std::sync::Mutex<Memo>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    std::sync::Arc::new(move |p: &[f64], region: &[(f64, f64)]| {
        if let Ok(m) = memo.lock()
            && let Some((mp, mr, v)) = m.as_ref()
            && mp.as_slice() == p
            && mr.as_slice() == region
        {
            return *v;
        }
        let v = crate::engine::integrate_region(&|x: &[f64]| func(x, p), region);
        if let Ok(mut m) = memo.lock() {
            *m = Some((p.to_vec(), region.to_vec(), v));
        }
        v
    })
}

/// `Φ(b) − Φ(a)` of the standard normal, without cancellation in the tails.
fn normal_mass(a: f64, b: f64) -> f64 {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    if a >= 0.0 {
        0.5 * (oxistats::erfc(a * r) - oxistats::erfc(b * r))
    } else if b <= 0.0 {
        0.5 * (oxistats::erfc(-b * r) - oxistats::erfc(-a * r))
    } else {
        0.5 * (oxistats::erf(b * r) - oxistats::erf(a * r))
    }
}

/// `∫ (B − t)^(−n) dt` from `t1` to `t2` (`t < B`), scaled by `A`: a
/// Crystal Ball power-law tail.
fn power_tail(big_a: f64, big_b: f64, n: f64, t1: f64, t2: f64) -> f64 {
    if t2 <= t1 {
        return 0.0;
    }
    let (u1, u2) = (big_b - t1, big_b - t2);
    if (n - 1.0).abs() < 1e-9 {
        big_a * (u1 / u2).ln()
    } else {
        big_a / (n - 1.0) * (u2.powf(1.0 - n) - u1.powf(1.0 - n))
    }
}

/// The area of `oxistats::crystal_ball` (unit peak, tail on the low side for
/// `alpha > 0`) between `lo` and `hi`.
fn crystal_ball_area(lo: f64, hi: f64, mean: f64, sigma: f64, alpha: f64, n: f64) -> f64 {
    let s = if sigma == 0.0 {
        f64::MIN_POSITIVE
    } else {
        sigma
    };
    let (mut t1, mut t2) = ((lo - mean) / s, (hi - mean) / s);
    if alpha < 0.0 {
        (t1, t2) = (-t2, -t1);
    }
    let a = alpha.abs().max(f64::MIN_POSITIVE);
    let big_a = (n / a).powf(n) * (-0.5 * a * a).exp();
    let big_b = n / a - a;
    let tail = power_tail(big_a, big_b, n, t1, t2.min(-a));
    let core = normal_mass(t1.max(-a), t2.max(-a)) * std::f64::consts::TAU.sqrt();
    s * (tail + core)
}

/// The area of `oxistats::double_crystal_ball` between `lo` and `hi`.
#[allow(clippy::too_many_arguments)]
fn double_crystal_ball_area(
    lo: f64,
    hi: f64,
    mean: f64,
    sigma: f64,
    alo: f64,
    nlo: f64,
    ahi: f64,
    nhi: f64,
) -> f64 {
    let s = if sigma == 0.0 {
        f64::MIN_POSITIVE
    } else {
        sigma
    };
    let (t1, t2) = ((lo - mean) / s, (hi - mean) / s);
    let (alo, ahi) = (
        alo.abs().max(f64::MIN_POSITIVE),
        ahi.abs().max(f64::MIN_POSITIVE),
    );
    let low = {
        let big_a = (nlo / alo).powf(nlo) * (-0.5 * alo * alo).exp();
        power_tail(big_a, nlo / alo - alo, nlo, t1, t2.min(-alo))
    };
    // The high tail A·(B + t)^(−n) is the low tail's mirror: t → −t.
    let high = {
        let big_a = (nhi / ahi).powf(nhi) * (-0.5 * ahi * ahi).exp();
        power_tail(big_a, nhi / ahi - ahi, nhi, -t2, (-t1).min(-ahi))
    };
    let core =
        normal_mass(t1.max(-alo).min(ahi), t2.max(-alo).min(ahi)) * std::f64::consts::TAU.sqrt();
    s * (low + high + core)
}

impl Model {
    /// Attach the model's integral over a region — one `(lo, hi)` per
    /// variable — known without integrating numerically: unbinned fits then
    /// normalize with it, and bin-integrated fits average with it, instead of
    /// quadrature. The built-in shapes have theirs; sums and products of
    /// models with one have one too.
    ///
    /// ```
    /// # use oxifit::Model;
    /// let ramp = Model::new("ramp", &["a"], vec![2.0], |x, p| p[0] * x)
    ///     .with_integral(|p, region| {
    ///         let (lo, hi) = region[0];
    ///         0.5 * p[0] * (hi * hi - lo * lo)
    ///     });
    /// assert_eq!(ramp.integral(&[(0.0, 3.0)]), Some(9.0));
    /// ```
    #[must_use]
    pub fn with_integral(
        mut self,
        integral: impl Fn(&[f64], &[(f64, f64)]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        self.integral = Some(std::sync::Arc::new(integral));
        self
    }

    /// The model's integral over `region` at its current parameters, when it
    /// knows it ([`with_integral`](Self::with_integral)); `None` otherwise.
    #[must_use]
    pub fn integral(&self, region: &[(f64, f64)]) -> Option<f64> {
        self.integral.as_ref().map(|i| i(&self.params, region))
    }
}

impl Model {
    /// Build a function from named parameters and a closure `f(x, params)`.
    pub fn new(
        name: &str,
        param_names: &[&str],
        params: Vec<f64>,
        func: impl Fn(f64, &[f64]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        Model::with_dim(name, param_names, params, 1, move |c, p| func(c[0], p))
    }

    /// A model of two variables, `f(x, y, params)` — for 2-D histograms,
    /// [`Points2`](crate::Points2) and [`Events2`](crate::Events2).
    ///
    /// ```
    /// # use oxifit::Model;
    /// let plane = Model::new2("plane", &["a", "b", "c"], vec![1.0, 2.0, 3.0], |x, y, p| {
    ///     p[0] + p[1] * x + p[2] * y
    /// });
    /// assert_eq!(plane.eval_at(&[1.0, 1.0]), 6.0);
    /// ```
    pub fn new2(
        name: &str,
        param_names: &[&str],
        params: Vec<f64>,
        func: impl Fn(f64, f64, &[f64]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        Model::with_dim(name, param_names, params, 2, move |c, p| {
            func(c[0], c[1], p)
        })
    }

    /// A model of three variables, `f(x, y, z, params)` — for 3-D histograms,
    /// [`Points3`](crate::Points3) and [`Events3`](crate::Events3).
    pub fn new3(
        name: &str,
        param_names: &[&str],
        params: Vec<f64>,
        func: impl Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        Model::with_dim(name, param_names, params, 3, move |c, p| {
            func(c[0], c[1], c[2], p)
        })
    }

    /// A model of `dim` variables from a closure over the coordinates.
    fn with_dim(
        name: &str,
        param_names: &[&str],
        params: Vec<f64>,
        dim: usize,
        func: impl Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        let constraints = vec![Constraint::default(); params.len()];
        Model {
            name: name.to_string(),
            param_names: param_names.iter().map(|s| s.to_string()).collect(),
            params,
            constraints,
            func: std::sync::Arc::new(func),
            grad: None,
            value_grad: None,
            parts: None,
            unknown: Vec::new(),
            dim,
            dim_mismatch: None,
            integral: None,
        }
    }

    /// The number of variables the model takes: 1, 2 or 3.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Index of the parameter named `name`.
    fn index_of(&self, name: &str) -> Option<usize> {
        self.param_names.iter().position(|n| n == name)
    }

    /// Index of the parameter named `name` for a setter, recording the name
    /// as unknown when there is none.
    fn setter_index(&mut self, name: &str) -> Option<usize> {
        let i = self.index_of(name);
        if i.is_none() && !self.unknown.iter().any(|u| u == name) {
            self.unknown.push(name.to_string());
        }
        i
    }

    /// Check that every parameter name given to the per-parameter setters
    /// ([`limit`](Self::limit), [`lower_limit`](Self::lower_limit),
    /// [`upper_limit`](Self::upper_limit), [`fix`](Self::fix),
    /// [`step`](Self::step), [`constrain`](Self::constrain)) names a parameter
    /// of the model.
    ///
    /// A name that matches none — usually a typo — would make the setter do
    /// nothing, and the fit a different one than intended. Such a model fits
    /// as **invalid** (`FitResult::valid` is `false`); this tells you why.
    ///
    /// ```
    /// # use oxifit::{FitError, Model};
    /// let m = Model::gaussian("g").lower_limit("sigam", 0.0);
    /// assert_eq!(m.check(), Err(FitError::UnknownParameter("sigam".into())));
    /// assert!(Model::gaussian("g").lower_limit("sigma", 0.0).check().is_ok());
    /// ```
    ///
    /// It also reports a sum, product or convolution built from parts of the
    /// wrong dimension.
    ///
    /// # Errors
    /// [`FitError::DimensionMismatch`](crate::FitError::DimensionMismatch) for
    /// parts of the wrong dimension, else
    /// [`FitError::UnknownParameter`](crate::FitError::UnknownParameter) with the
    /// first unknown name.
    pub fn check(&self) -> Result<(), crate::FitError> {
        if let Some((expected, found)) = self.dim_mismatch {
            return Err(crate::FitError::DimensionMismatch { expected, found });
        }
        match self.unknown.first() {
            Some(name) => Err(crate::FitError::UnknownParameter(name.clone())),
            None => Ok(()),
        }
    }

    /// Mutable access to a parameter's constraints, growing the vector if a
    /// later `with_params` shrank it.
    fn constraint_mut(&mut self, i: usize) -> &mut Constraint {
        if self.constraints.len() < self.params.len() {
            self.constraints
                .resize(self.params.len(), Constraint::default());
        }
        &mut self.constraints[i]
    }

    /// Constrain parameter `name` to `[lower, upper]` during the fit. A name
    /// that matches no parameter makes the fit invalid (see
    /// [`check`](Self::check)), as for the other per-parameter setters.
    #[must_use]
    pub fn limit(mut self, name: &str, lower: f64, upper: f64) -> Model {
        if let Some(i) = self.setter_index(name) {
            let c = self.constraint_mut(i);
            c.lower = Some(lower);
            c.upper = Some(upper);
        }
        self
    }

    /// Constrain parameter `name` to be `>= lower` (e.g. a positive width).
    #[must_use]
    pub fn lower_limit(mut self, name: &str, lower: f64) -> Model {
        if let Some(i) = self.setter_index(name) {
            self.constraint_mut(i).lower = Some(lower);
        }
        self
    }

    /// Constrain parameter `name` to be `<= upper`.
    #[must_use]
    pub fn upper_limit(mut self, name: &str, upper: f64) -> Model {
        if let Some(i) = self.setter_index(name) {
            self.constraint_mut(i).upper = Some(upper);
        }
        self
    }

    /// Hold parameter `name` fixed at its current value during the fit.
    #[must_use]
    pub fn fix(mut self, name: &str) -> Model {
        if let Some(i) = self.setter_index(name) {
            self.constraint_mut(i).fixed = true;
        }
        self
    }

    /// Set the initial Minuit2 step for parameter `name` (default: 10 % of the
    /// value, with a small floor).
    #[must_use]
    pub fn step(mut self, name: &str, step: f64) -> Model {
        if let Some(i) = self.setter_index(name) {
            self.constraint_mut(i).step = Some(step);
        }
        self
    }

    /// Constrain parameter `name` with a Gaussian of `mean` and `sigma` — an
    /// external measurement of it, or a nuisance parameter's prior. The fit
    /// adds `((p − mean)/sigma)²` to its cost (a χ² term, or `−2 ln` of the
    /// Gaussian for a likelihood) and counts the constraint as one more
    /// measurement in `ndf`. A non-positive `sigma` removes the constraint; a
    /// fixed parameter's constraint has no effect.
    ///
    /// ```
    /// # use oxifit::{FitExt, Model, Points};
    /// // The data say 10 ± 1; an earlier measurement said 12 ± 1.
    /// let data = Points::new(&[0.0], &[10.0], &[1.0])?;
    /// let c = Model::polynomial("c", 0).with_params(vec![0.0]).constrain("p0", 12.0, 1.0);
    /// let fit = data.fit(&c);
    /// assert!((fit.params[0] - 11.0).abs() < 1e-3); // the weighted average
    /// # Ok::<(), oxifit::StatError>(())
    /// ```
    #[must_use]
    pub fn constrain(mut self, name: &str, mean: f64, sigma: f64) -> Model {
        if let Some(i) = self.setter_index(name) {
            self.constraint_mut(i).gauss = (sigma > 0.0).then_some((mean, sigma));
        }
        self
    }

    /// The sum of several models, `f(x) = Σ fₖ(x)` — a signal on a background,
    /// several peaks. Its parameters are every part's, in order, each named
    /// `part.param` after its part's [`name`](Self::name) (`"sig.mean"`,
    /// `"bkg.slope"`); a part whose name repeats an earlier one's is numbered
    /// (`"g"`, `"g_2"`, …). Each part keeps its seeds, limits, fixed
    /// parameters, steps and constraints. `a + b` is the same as
    /// `Model::sum("a+b", &[&a, &b])`.
    ///
    /// ```
    /// # use oxifit::Model;
    /// let sig = Model::gaussian("sig").with_params(vec![50.0, 91.2, 2.5]);
    /// let bkg = Model::exponential("bkg").with_params(vec![3.0, -0.02]);
    /// let model = Model::sum("z", &[&sig, &bkg]).lower_limit("sig.sigma", 0.0);
    /// assert_eq!(model.param_names[1], "sig.mean");
    /// assert_eq!(model.param_names[4], "bkg.slope");
    /// assert!((model.eval(91.2) - (50.0 + bkg.eval(91.2))).abs() < 1e-12);
    /// ```
    pub fn sum(name: &str, parts: &[&Model]) -> Model {
        let empty = Model {
            name: name.to_string(),
            param_names: Vec::new(),
            params: Vec::new(),
            constraints: Vec::new(),
            func: std::sync::Arc::new(|_, _| 0.0),
            grad: Some(std::sync::Arc::new(|_, _, _| {})),
            value_grad: Some(std::sync::Arc::new(|_, _, _| 0.0)),
            parts: Some(Vec::new()),
            unknown: Vec::new(),
            dim: parts.first().map_or(1, |p| p.dim),
            dim_mismatch: None,
            integral: Some(std::sync::Arc::new(|_, _| 0.0)),
        };
        parts.iter().fold(empty, |sum, part| sum.push_part(part))
    }

    /// Append `part` to this sum: its parameters (prefixed with its name,
    /// numbered when the name repeats) and its term.
    fn push_part(mut self, part: &Model) -> Model {
        let names = self.parts.get_or_insert_with(Vec::new);
        let base = part.name.clone();
        let count = names.iter().filter(|n| **n == base).count();
        names.push(base.clone());
        let prefix = if count == 0 {
            base
        } else {
            format!("{base}_{}", count + 1)
        };
        let offset = self.params.len();
        let n = part.params.len();
        // Constraints line up with the parameters (a `with_params` may have
        // left the vector shorter).
        self.constraints.resize(offset, Constraint::default());
        for k in 0..n {
            let pname = part
                .param_names
                .get(k)
                .map_or_else(|| format!("p{k}"), Clone::clone);
            self.param_names.push(format!("{prefix}.{pname}"));
            self.constraints
                .push(part.constraints.get(k).copied().unwrap_or_default());
        }
        self.params.extend_from_slice(&part.params);
        for u in &part.unknown {
            if !self.unknown.contains(u) {
                self.unknown.push(u.clone());
            }
        }
        if part.dim != self.dim && self.dim_mismatch.is_none() {
            self.dim_mismatch = Some((self.dim, part.dim));
        }
        if self.dim_mismatch.is_none() {
            self.dim_mismatch = part.dim_mismatch;
        }
        let (head, tail) = (self.func.clone(), part.func.clone());
        let (head_f, tail_f) = (head.clone(), tail.clone());
        self.func = std::sync::Arc::new(move |c, p| {
            head(c, &p[..offset]) + tail(c, &p[offset..offset + n])
        });
        // The sum has a gradient when every part does, and always an integral:
        // each part's known one, or its own numerical one, remembered for the
        // part's last parameters — so a step in one part's parameters (most of
        // a minimizer's) integrates that part alone.
        self.value_grad = match (self.value_grad.take(), part.fused_value_grad()) {
            (Some(vh), Some(vt)) => Some(std::sync::Arc::new(move |c, p, out: &mut [f64]| {
                vh(c, &p[..offset], &mut out[..offset])
                    + vt(c, &p[offset..offset + n], &mut out[offset..offset + n])
            })),
            _ => None,
        };
        self.grad = match (self.grad.take(), part.grad.clone()) {
            (Some(gh), Some(gt)) => Some(std::sync::Arc::new(move |c, p, out: &mut [f64]| {
                gh(c, &p[..offset], &mut out[..offset]);
                gt(c, &p[offset..offset + n], &mut out[offset..offset + n]);
            })),
            _ => None,
        };
        let ih = self
            .integral
            .take()
            .unwrap_or_else(|| remembered_integral(head_f));
        let it = part
            .integral
            .clone()
            .unwrap_or_else(|| remembered_integral(tail_f));
        self.integral = Some(std::sync::Arc::new(move |p, region: &[(f64, f64)]| {
            ih(&p[..offset], region) + it(&p[offset..offset + n], region)
        }));
        self
    }

    /// The convolution `(f ⊗ g)(x) = ∫ f(x − t)·g(t) dt` of `signal` with
    /// `resolution` — a lifetime smeared by vertex resolution, a resonance by
    /// detector resolution (RooFit's `RooFFTConvPdf`). `window` is the range
    /// of `t` the resolution covers (±5–8 of its widths); `g` should be
    /// normalized to unit area (e.g. [`normal`](Self::normal)) so the
    /// convolution keeps `signal`'s area.
    ///
    /// Each evaluation integrates numerically over the window (bounded,
    /// adaptive Gauss–Kronrod, so a step such as a decay starting at `t = 0` is
    /// handled), which costs a few hundred evaluations of the two models. The
    /// parameters are `signal`'s then `resolution`'s, named after them as in a
    /// [`sum`](Self::sum) (`decay.tau`, `res.sigma`); each keeps its seeds,
    /// limits and constraints. With analytic gradients on both, the
    /// convolution has one too.
    ///
    /// ```
    /// # use oxifit::Model;
    /// let decay = Model::new("decay", &["n", "tau"], vec![1.0, 1.5], |t, p| {
    ///     if t < 0.0 { 0.0 } else { p[0] / p[1] * (-t / p[1]).exp() }
    /// });
    /// let res = Model::normal("res").with_params(vec![0.0, 0.1]);
    /// let smeared = Model::convolve("measured", &decay, &res, (-1.0, 1.0));
    /// assert_eq!(smeared.param_names, ["decay.n", "decay.tau", "res.mean", "res.sigma"]);
    /// // Smearing spreads the decay below t = 0.
    /// assert!(smeared.eval(-0.05) > 0.0 && decay.eval(-0.05) == 0.0);
    /// ```
    pub fn convolve(name: &str, signal: &Model, resolution: &Model, window: (f64, f64)) -> Model {
        // The parameters, names and constraints as for a two-part sum.
        let mut m = Model::sum(name, &[signal, resolution]);
        m.parts = None; // a convolution is not a sum: `+` must not append to it
        m.integral = None;
        for part in [signal, resolution] {
            if part.dim != 1 && m.dim_mismatch.is_none() {
                m.dim_mismatch = Some((1, part.dim));
            }
        }
        m.dim = 1;
        let (lo, hi) = if window.0 <= window.1 {
            window
        } else {
            (window.1, window.0)
        };
        let ns = signal.params.len();
        let nr = resolution.params.len();
        let (f, g) = (signal.func.clone(), resolution.func.clone());
        let (fv, gv) = (f.clone(), g.clone());
        m.func = std::sync::Arc::new(move |c, p| {
            let (ps, pr) = p.split_at(ns);
            let x = c[0];
            let integrand = |t: f64| fv(&[x - t], ps) * gv(&[t], pr);
            oxifit_formula::integrate_adaptive(integrand, lo, hi, 1e-10, CONV_INTERVALS).0
        });
        m.value_grad = None;
        m.grad = match (signal.grad.clone(), resolution.grad.clone()) {
            (Some(fg), Some(gg)) => Some(std::sync::Arc::new(move |c, p, out: &mut [f64]| {
                let (ps, pr) = p.split_at(ns);
                let x = c[0];
                for (k, o) in out.iter_mut().enumerate().take(ns + nr) {
                    let integrand = |t: f64| {
                        if k < ns {
                            let mut d = vec![0.0; ns];
                            fg(&[x - t], ps, &mut d);
                            d[k] * g(&[t], pr)
                        } else {
                            let mut d = vec![0.0; nr];
                            gg(&[t], pr, &mut d);
                            f(&[x - t], ps) * d[k - ns]
                        }
                    };
                    *o = oxifit_formula::integrate_adaptive(
                        integrand,
                        lo,
                        hi,
                        1e-10,
                        CONV_INTERVALS,
                    )
                    .0;
                }
            })),
            _ => None,
        };
        m
    }

    /// The product of 1-D models, one per variable: `f(x, y) = fx(x)·fy(y)`
    /// (or three for `f(x, y, z)`) — RooFit's `RooProdPdf`, the usual way to
    /// build a 2-D or 3-D shape from 1-D ones. Parameters are the factors',
    /// named after them as in a [`sum`](Self::sum); keep only one overall
    /// scale free (multiply a [`gaussian`](Self::gaussian) by a unit-area
    /// [`normal`](Self::normal), say). It has a gradient when every factor
    /// does, and its integral over a region is the product of the factors' —
    /// so an unbinned fit normalizes it with 1-D integrals only.
    ///
    /// ```
    /// # use oxifit::Model;
    /// let blob = Model::product(
    ///     "blob",
    ///     &[&Model::gaussian("x").with_params(vec![10.0, 0.0, 1.0]), &Model::normal("y")],
    /// );
    /// assert_eq!(blob.dim(), 2);
    /// assert_eq!(blob.param_names, ["x.constant", "x.mean", "x.sigma", "y.mean", "y.sigma"]);
    /// let peak = blob.eval_at(&[0.0, 0.0]);
    /// assert!((peak - 10.0 / std::f64::consts::TAU.sqrt()).abs() < 1e-12);
    /// ```
    pub fn product(name: &str, factors: &[&Model]) -> Model {
        let mut m = Model::sum(name, factors);
        m.parts = None; // a product is not a sum: `+` must not append to it
        m.dim = factors.len().clamp(1, 3);
        m.dim_mismatch = None;
        if factors.len() > 3 {
            m.dim_mismatch = Some((3, factors.len()));
        }
        for f in factors {
            if f.dim != 1 && m.dim_mismatch.is_none() {
                m.dim_mismatch = Some((1, f.dim));
            }
        }
        // Each factor's offset and parameter count, its function, gradient
        // and integral.
        let mut layout = Vec::new();
        let mut offset = 0;
        for f in factors.iter().take(3) {
            let n = f.params.len();
            layout.push((
                offset,
                n,
                f.func.clone(),
                f.grad.clone(),
                f.integral.clone(),
            ));
            offset += n;
        }
        let funcs: Vec<(usize, usize, ModelFn)> = layout
            .iter()
            .map(|(o, n, f, _, _)| (*o, *n, f.clone()))
            .collect();
        m.func = std::sync::Arc::new(move |c, p| {
            funcs
                .iter()
                .enumerate()
                .map(|(k, (o, n, f))| f(&[c[k]], &p[*o..o + n]))
                .product()
        });
        let grads: Option<Vec<(usize, usize, ModelFn, GradFn)>> = layout
            .iter()
            .map(|(o, n, f, g, _)| g.clone().map(|g| (*o, *n, f.clone(), g)))
            .collect();
        m.value_grad = None;
        m.grad = grads.map(|grads| {
            std::sync::Arc::new(move |c: &[f64], p: &[f64], out: &mut [f64]| {
                let values: Vec<f64> = grads
                    .iter()
                    .enumerate()
                    .map(|(k, (o, n, f, _))| f(&[c[k]], &p[*o..o + n]))
                    .collect();
                for (k, (o, n, _, g)) in grads.iter().enumerate() {
                    let others: f64 = values
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != k)
                        .map(|(_, v)| v)
                        .product();
                    g(&[c[k]], &p[*o..o + n], &mut out[*o..o + n]);
                    out[*o..o + n].iter_mut().for_each(|v| *v *= others);
                }
            }) as GradFn
        });
        // ∫∫ fx·fy = ∫fx · ∫fy: each factor's own integral, or a 1-D one.
        let factors_int: Vec<(usize, usize, ModelFn, Option<IntegralFn>)> = layout
            .into_iter()
            .map(|(o, n, f, _, i)| (o, n, f, i))
            .collect();
        m.integral = Some(std::sync::Arc::new(move |p, region: &[(f64, f64)]| {
            factors_int
                .iter()
                .enumerate()
                .map(|(k, (o, n, f, i))| {
                    let (lo, hi) = region[k];
                    let pk = &p[*o..o + n];
                    match i {
                        Some(i) => i(pk, &[(lo, hi)]),
                        None => {
                            oxifit_formula::integrate_adaptive(
                                |x| f(&[x], pk),
                                lo,
                                hi,
                                1e-10,
                                CONV_INTERVALS,
                            )
                            .0
                        }
                    }
                })
                .product()
        }));
        m
    }

    /// A unit-area Gaussian density `exp(−½·((x − mean)/sigma)²)/(sigma·√(2π))`
    /// — a resolution function for [`convolve`](Self::convolve), or a shape for
    /// the unbinned likelihoods. Parameters `mean`, `sigma` (keep it positive:
    /// `.lower_limit("sigma", 0.0)`); it has an analytic gradient.
    pub fn normal(name: &str) -> Model {
        let norm = 1.0 / std::f64::consts::TAU.sqrt();
        Model::new(name, &["mean", "sigma"], vec![0.0, 1.0], move |x, p| {
            let u = (x - p[0]) / p[1];
            norm / p[1] * (-0.5 * u * u).exp()
        })
        .with_gradient(move |x, p, g| {
            let u = (x - p[0]) / p[1];
            let f = norm / p[1] * (-0.5 * u * u).exp();
            g[0] = f * u / p[1];
            g[1] = f * (u * u - 1.0) / p[1];
        })
        .with_value_grad(move |x, p, g| {
            let u = (x - p[0]) / p[1];
            let f = norm / p[1] * (-0.5 * u * u).exp();
            g[0] = f * u / p[1];
            g[1] = f * (u * u - 1.0) / p[1];
            f
        })
        .with_integral(|p, r| normal_mass((r[0].0 - p[0]) / p[1], (r[0].1 - p[0]) / p[1]))
    }

    /// A Gaussian `[0]·exp(-½·((x-[1])/[2])²)` (ROOT's `"gaus"`); parameters are
    /// `constant`, `mean`, `sigma`. Seed sensible initials with
    /// [`with_params`](Self::with_params) or [`estimate_from`](Self::estimate_from),
    /// and add `.lower_limit("sigma", 0.0)` to keep the width positive.
    pub fn gaussian(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "mean", "sigma"],
            vec![1.0, 0.0, 1.0],
            |x, p| {
                let s = if p[2] == 0.0 { f64::EPSILON } else { p[2] };
                p[0] * (-0.5 * ((x - p[1]) / s).powi(2)).exp()
            },
        )
        .with_gradient(|x, p, g| {
            let s = if p[2] == 0.0 { f64::EPSILON } else { p[2] };
            let u = (x - p[1]) / s;
            let e = (-0.5 * u * u).exp();
            g[0] = e;
            g[1] = p[0] * e * u / s;
            g[2] = p[0] * e * u * u / s;
        })
        .with_value_grad(|x, p, g| {
            let s = if p[2] == 0.0 { f64::EPSILON } else { p[2] };
            let u = (x - p[1]) / s;
            let e = (-0.5 * u * u).exp();
            g[0] = e;
            g[1] = p[0] * e * u / s;
            g[2] = p[0] * e * u * u / s;
            p[0] * e
        })
        .with_integral(|p, r| {
            let s = if p[2] == 0.0 { f64::EPSILON } else { p[2] };
            p[0] * s.abs()
                * std::f64::consts::TAU.sqrt()
                * normal_mass((r[0].0 - p[1]) / s.abs(), (r[0].1 - p[1]) / s.abs())
        })
    }

    /// An exponential `exp([0] + [1]·x)` (ROOT's `"expo"`).
    pub fn exponential(name: &str) -> Model {
        Model::new(name, &["constant", "slope"], vec![0.0, -1.0], |x, p| {
            (p[0] + p[1] * x).exp()
        })
        .with_gradient(|x, p, g| {
            let e = (p[0] + p[1] * x).exp();
            g[0] = e;
            g[1] = e * x;
        })
        .with_value_grad(|x, p, g| {
            let e = (p[0] + p[1] * x).exp();
            g[0] = e;
            g[1] = e * x;
            e
        })
        .with_integral(|p, r| {
            let (lo, hi) = r[0];
            let base = (p[0] + p[1] * lo).exp();
            if p[1].abs() * (hi - lo).abs() < 1e-12 {
                base * (hi - lo)
            } else {
                base * (p[1] * (hi - lo)).exp_m1() / p[1]
            }
        })
    }

    /// A degree-`n` polynomial `Σ p[k]·x^k` (ROOT's `"polN"`).
    pub fn polynomial(name: &str, degree: usize) -> Model {
        let names: Vec<String> = (0..=degree).map(|k| format!("p{k}")).collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Model::new(name, &name_refs, vec![0.0; degree + 1], |x, p| {
            p.iter().rev().fold(0.0, |acc, &c| acc * x + c)
        })
        .with_gradient(|x, _p, g| {
            let mut xk = 1.0;
            for gk in g.iter_mut() {
                *gk = xk;
                xk *= x;
            }
        })
        .with_value_grad(|x, p, g| {
            let (mut xk, mut f) = (1.0, 0.0);
            for (gk, c) in g.iter_mut().zip(p) {
                *gk = xk;
                f += c * xk;
                xk *= x;
            }
            f
        })
        .with_integral(|p, r| {
            let (lo, hi) = r[0];
            let antiderivative = |x: f64| {
                p.iter()
                    .enumerate()
                    .rev()
                    .fold(0.0, |acc, (k, &c)| acc * x + c / (k as f64 + 1.0))
                    * x
            };
            antiderivative(hi) - antiderivative(lo)
        })
    }

    /// A **Crystal Ball** peak `constant · CB(x; mean, sigma, alpha, n)` — a
    /// Gaussian core with a power-law low-side tail (RooFit's `RooCBShape`).
    /// Parameters `constant`, `mean`, `sigma`, `alpha`, `n`; a negative `alpha`
    /// moves the tail to the high side. Seed the core with
    /// [`estimate_from`](Self::estimate_from) and keep the width/tail positive
    /// with `.lower_limit("sigma", 0.0).lower_limit("n", 1.0)`.
    pub fn crystal_ball(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "mean", "sigma", "alpha", "n"],
            vec![1.0, 0.0, 1.0, 1.5, 2.0],
            |x, p| p[0] * oxistats::crystal_ball(x, p[1], p[2], p[3], p[4]),
        )
        .with_integral(|p, r| p[0] * crystal_ball_area(r[0].0, r[0].1, p[1], p[2], p[3], p[4]))
    }

    /// A **double-sided Crystal Ball** `constant · DSCB(x)` — a Gaussian core with
    /// an independent power-law tail on each side. Parameters `constant`, `mean`,
    /// `sigma`, `alpha_lo`, `n_lo`, `alpha_hi`, `n_hi`.
    pub fn double_crystal_ball(name: &str) -> Model {
        Model::new(
            name,
            &[
                "constant", "mean", "sigma", "alpha_lo", "n_lo", "alpha_hi", "n_hi",
            ],
            vec![1.0, 0.0, 1.0, 1.5, 2.0, 1.5, 2.0],
            |x, p| p[0] * oxistats::double_crystal_ball(x, p[1], p[2], p[3], p[4], p[5], p[6]),
        )
        .with_integral(|p, r| {
            p[0] * double_crystal_ball_area(r[0].0, r[0].1, p[1], p[2], p[3], p[4], p[5], p[6])
        })
    }

    /// A **Breit–Wigner** resonance `constant · BW(x; mean, gamma)` (a Lorentzian
    /// of full width `gamma`). Parameters `constant`, `mean`, `gamma`.
    pub fn breit_wigner(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "mean", "gamma"],
            vec![1.0, 0.0, 1.0],
            |x, p| p[0] * oxistats::breit_wigner(x, p[1], p[2]),
        )
        .with_integral(|p, r| {
            let hw = p[2].abs().max(f64::MIN_POSITIVE) / 2.0;
            p[0] / std::f64::consts::PI
                * (((r[0].1 - p[1]) / hw).atan() - ((r[0].0 - p[1]) / hw).atan())
        })
    }

    /// A **Voigt profile** `constant · V(x; mean, sigma, gamma)` — a Lorentzian of
    /// half-width `gamma` convolved with a Gaussian of width `sigma` (a
    /// resolution-broadened resonance). Parameters `constant`, `mean`, `sigma`,
    /// `gamma`.
    pub fn voigtian(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "mean", "sigma", "gamma"],
            vec![1.0, 0.0, 1.0, 1.0],
            |x, p| p[0] * oxistats::voigtian(x, p[1], p[2], p[3]),
        )
    }

    /// A **Novosibirsk** peak `constant · N(x; peak, sigma, tail)` — an asymmetric
    /// Gaussian whose skew is set by `tail`. Parameters `constant`, `peak`,
    /// `sigma`, `tail`.
    pub fn novosibirsk(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "peak", "sigma", "tail"],
            vec![1.0, 0.0, 1.0, 0.1],
            |x, p| p[0] * oxistats::novosibirsk(x, p[1], p[2], p[3]),
        )
    }

    /// An **ARGUS** endpoint background `constant · ARGUS(x; m0, c, p)`. Parameters
    /// `constant`, `m0` (kinematic endpoint), `c` (curvature), `power` (the `√`
    /// exponent, ROOT's default `0.5`).
    pub fn argus(name: &str) -> Model {
        Model::new(
            name,
            &["constant", "m0", "c", "power"],
            vec![1.0, 10.0, -1.0, 0.5],
            |x, p| p[0] * oxistats::argus(x, p[1], p[2], p[3]),
        )
    }

    /// Build a model from an arbitrary ROOT [`TFormula`](oxifit_formula) string
    /// (`"[0]+[1]*x"`, `"gaus"`, `"[0]*exp(-[1]*x)+[2]"`, …). Parameters are
    /// `[0], [1], …` and the variable is `x`; the parameter count is inferred
    /// from the formula. All parameters start at zero — seed them with
    /// [`with_params`](Self::with_params). The model has an exact gradient,
    /// from the formula's symbolic derivatives (unless it uses `lgamma`), so
    /// MIGRAD, HESSE and L-BFGS-B do not need finite differences.
    ///
    /// # Errors
    /// Returns a [`ParseError`](oxifit_formula::ParseError) if the formula does
    /// not parse.
    pub fn from_formula(name: &str, formula: &str) -> Result<Model, oxifit_formula::ParseError> {
        let f = oxifit_formula::Formula::parse(formula)?;
        let npar = f.npar();
        // Own the parameter names so the closure can move `f`.
        let names = f.param_names().to_vec();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        // The exact gradient, from the formula's symbolic derivatives, when
        // every one exists.
        let derivatives: Option<Vec<oxifit_formula::Formula>> =
            (0..npar).map(|k| f.derivative(k)).collect();
        let model = Model::new(name, &name_refs, vec![0.0; npar], move |x, p| f.eval1(x, p));
        Ok(match derivatives {
            Some(d) => model.with_gradient(move |x, p, g| {
                for (gk, dk) in g.iter_mut().zip(&d) {
                    *gk = dk.eval1(x, p);
                }
            }),
            None => model,
        })
    }

    /// Replace the (initial) parameter values.
    #[must_use]
    pub fn with_params(mut self, params: Vec<f64>) -> Model {
        self.constraints.resize(params.len(), Constraint::default());
        self.params = params;
        self
    }

    /// Evaluate a 1-D model at `x` using its current parameters (`NaN` for a
    /// model of more variables: use [`eval_at`](Self::eval_at)).
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        self.eval_at(&[x])
    }

    /// Evaluate the model at the point `coords` (one value per variable)
    /// using its current parameters; `NaN` if `coords` does not have
    /// [`dim`](Self::dim) values.
    #[must_use]
    pub fn eval_at(&self, coords: &[f64]) -> f64 {
        if coords.len() == self.dim {
            (self.func)(coords, &self.params)
        } else {
            f64::NAN
        }
    }

    /// Give the model its analytic gradient: `grad(x, params, out)` writes
    /// `∂f(x; params)/∂pₖ` into `out[k]` for every parameter (`out` is zeroed
    /// before each call). The fit then uses exact derivatives of the cost:
    /// MIGRAD's steps (falling back to finite differences if it stalls far
    /// from the minimum), Levenberg–Marquardt's Jacobian,
    /// [`Minimizer::LBfgsB`](crate::Minimizer::LBfgsB), and the error matrix —
    /// a central difference of the gradient, `2n` gradient calls instead of
    /// HESSE's `~2n²` cost calls, which is where models with many parameters
    /// gain most. The Gaussian,
    /// exponential and polynomial shapes have theirs built in, and a sum of
    /// models has one when every part does. Verify a hand-written gradient
    /// with [`check_gradient`](Self::check_gradient).
    ///
    /// ```
    /// # use oxifit::Model;
    /// let line = Model::new("line", &["a", "b"], vec![1.0, 2.0], |x, p| p[0] + p[1] * x)
    ///     .with_gradient(|x, _p, g| {
    ///         g[0] = 1.0;
    ///         g[1] = x;
    ///     });
    /// assert!(line.check_gradient(&[0.0, 1.5, 3.0]) < 1e-8);
    /// ```
    #[must_use]
    pub fn with_gradient(
        mut self,
        grad: impl Fn(f64, &[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> Model {
        self.grad = Some(std::sync::Arc::new(move |c, p, out| grad(c[0], p, out)));
        self.value_grad = None;
        self
    }

    /// Give a built-in shape its fused value and gradient (see
    /// [`ValGradFn`]); after [`with_gradient`](Self::with_gradient), which
    /// clears it.
    fn with_value_grad(
        mut self,
        vg: impl Fn(f64, &[f64], &mut [f64]) -> f64 + Send + Sync + 'static,
    ) -> Model {
        self.value_grad = Some(std::sync::Arc::new(move |c, p, out| vg(c[0], p, out)));
        self
    }

    /// The fused value and gradient: the model's own, or one made from its
    /// function and gradient; `None` without a gradient.
    pub(crate) fn fused_value_grad(&self) -> Option<ValGradFn> {
        if let Some(vg) = &self.value_grad {
            return Some(vg.clone());
        }
        let (f, g) = (self.func.clone(), self.grad.clone()?);
        Some(std::sync::Arc::new(move |c, p, out| {
            g(c, p, out);
            f(c, p)
        }))
    }

    /// Give a model of any dimension its analytic gradient:
    /// `grad(coords, params, out)` writes `∂f/∂pₖ` at the point `coords` into
    /// `out[k]` (see [`with_gradient`](Self::with_gradient), its 1-D form).
    #[must_use]
    pub fn with_gradient_at(
        mut self,
        grad: impl Fn(&[f64], &[f64], &mut [f64]) + Send + Sync + 'static,
    ) -> Model {
        self.grad = Some(std::sync::Arc::new(grad));
        self.value_grad = None;
        self
    }

    /// Whether the model has an analytic gradient.
    #[must_use]
    pub fn has_gradient(&self) -> bool {
        self.grad.is_some()
    }

    /// The gradient `∂f/∂p` of a 1-D model at `x` with the current
    /// parameters: analytic when the model has one, else a central difference.
    #[must_use]
    pub fn gradient(&self, x: f64) -> Vec<f64> {
        self.gradient_at(&[x])
    }

    /// The gradient `∂f/∂p` at the point `coords`, as [`gradient`](Self::gradient).
    #[must_use]
    pub fn gradient_at(&self, coords: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.params.len()];
        match &self.grad {
            Some(g) => g(coords, &self.params, &mut out),
            None => out = self.numerical_gradient(coords),
        }
        out
    }

    /// `∂f/∂p` at `x` by central differences (step `1e-6·max(|p|, 1)`).
    fn numerical_gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut p = self.params.clone();
        (0..p.len())
            .map(|k| {
                let h = 1e-6 * p[k].abs().max(1.0);
                let v = p[k];
                p[k] = v + h;
                let up = (self.func)(x, &p);
                p[k] = v - h;
                let down = (self.func)(x, &p);
                p[k] = v;
                (up - down) / (2.0 * h)
            })
            .collect()
    }

    /// The largest disagreement between a 1-D model's analytic gradient and a
    /// central difference over the points `xs`, at the current parameters —
    /// relative to the larger of the two (or absolute below 1): about `1e-8`
    /// for a correct gradient. `NaN` when the model has no gradient.
    #[must_use]
    pub fn check_gradient(&self, xs: &[f64]) -> f64 {
        let points: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
        self.check_gradient_at(&points)
    }

    /// [`check_gradient`](Self::check_gradient) at points of any dimension.
    #[must_use]
    pub fn check_gradient_at(&self, points: &[Vec<f64>]) -> f64 {
        let Some(g) = &self.grad else {
            return f64::NAN;
        };
        let mut out = vec![0.0; self.params.len()];
        points
            .iter()
            .flat_map(|x| {
                out.iter_mut().for_each(|o| *o = 0.0);
                g(x, &self.params, &mut out);
                let num = self.numerical_gradient(x);
                out.iter()
                    .zip(num)
                    .map(|(&a, n)| (a - n).abs() / a.abs().max(n.abs()).max(1.0))
                    .collect::<Vec<_>>()
            })
            .fold(0.0, f64::max)
    }

    /// Whether the model's first three parameters are the Gaussian-core triple
    /// `constant`/`mean`/`sigma` — the Gaussian and the peaks built on a Gaussian
    /// core (Crystal Ball, double Crystal Ball, Voigt), the shapes
    /// [`estimate_from`](Self::estimate_from) can seed.
    fn has_gaussian_core(&self) -> bool {
        self.param_names.len() >= 3
            && self.param_names[0] == "constant"
            && self.param_names[1] == "mean"
            && self.param_names[2] == "sigma"
    }

    /// Seed the parameters from a dataset, for a good starting point even when a
    /// histogram's stored moment sums are zero (e.g. one built with
    /// `set_bin_content` rather than `fill`). Works for any [`FitData`] —
    /// histogram, graph, or raw points.
    ///
    /// For shapes with a Gaussian core (the [`gaussian`](Self::gaussian),
    /// [`crystal_ball`](Self::crystal_ball),
    /// [`double_crystal_ball`](Self::double_crystal_ball), and
    /// [`voigtian`](Self::voigtian) peaks) this sets the leading `(constant, mean,
    /// sigma)` to the peak height and the `y`-weighted mean and standard deviation
    /// of the data, leaving any tail parameters at their defaults. For other
    /// shapes it is a no-op (seed those with [`with_params`](Self::with_params)).
    ///
    /// ```
    /// # use oxifit::{FitExt, Model, Points};
    /// let data = Points::new(&[-1.0, 0.0, 1.0], &[1.0, 4.0, 1.0], &[0.2; 3])?;
    /// let model = Model::gaussian("g").estimate_from(&data); // no manual seed
    /// let fit = data.fit(&model);
    /// # assert!(fit.params[2] > 0.0);
    /// # Ok::<(), oxifit::StatError>(())
    /// ```
    #[must_use]
    pub fn estimate_from(mut self, data: &impl FitData) -> Model {
        if self.has_gaussian_core() {
            let (constant, mean, sigma) = gaussian_seed(&data.points());
            self.params[0] = constant;
            self.params[1] = mean;
            self.params[2] = sigma;
        }
        self
    }
}

/// `a + b` is [`Model::sum`]`("a+b", &[&a, &b])`, the parts' parameters
/// named `a.…` and `b.…`. Adding to a sum appends a part, so `a + b + c` is
/// `Model::sum("a+b+c", &[&a, &b, &c])`.
impl std::ops::Add for Model {
    type Output = Model;
    fn add(self, rhs: Model) -> Model {
        let lhs = if self.parts.is_some() {
            self
        } else {
            Model::sum(&self.name.clone(), &[&self])
        };
        let mut sum = lhs.push_part(&rhs);
        sum.name = format!("{}+{}", sum.name, rhs.name);
        sum
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("param_names", &self.param_names)
            .field("params", &self.params)
            .field("constraints", &self.constraints)
            .finish_non_exhaustive() // the model closure is not printable
    }
}

/// `y`-weighted `(peak, mean, sigma)` of the data points — the Gaussian seed.
/// Robust when the weights carry no spread (falls back to a unit width) or are
/// all zero (mean 0, width 1, the peak height).
fn gaussian_seed(points: &[Point]) -> (f64, f64, f64) {
    let (mut sw, mut swx, mut swx2) = (0.0, 0.0, 0.0);
    let mut peak = f64::NEG_INFINITY;
    for p in points {
        sw += p.y;
        swx += p.y * p.x;
        swx2 += p.y * p.x * p.x;
        peak = peak.max(p.y);
    }
    let peak = if points.is_empty() { 0.0 } else { peak };
    if sw <= 0.0 {
        return (peak, 0.0, 1.0);
    }
    let mean = swx / sw;
    let sigma = (swx2 / sw - mean * mean).max(0.0).sqrt();
    (peak, mean, if sigma > 0.0 { sigma } else { 1.0 })
}

#[cfg(test)]
mod tests {
    use super::Model;

    /// The fused value and gradient agree with `func` and `grad`, for the
    /// shapes that have one, for sums mixing fused and unfused parts, and
    /// after a user gradient replaces a shape's.
    #[test]
    fn fused_value_gradient_matches_the_separate_ones() {
        let custom = Model::new("c", &["a"], vec![0.7], |x, p| p[0] * x.sin())
            .with_gradient(|x, _, g| g[0] = x.sin());
        let models = [
            Model::gaussian("g").with_params(vec![3.0, 0.4, 1.3]),
            Model::normal("n").with_params(vec![-0.2, 0.8]),
            Model::exponential("e").with_params(vec![0.5, -0.7]),
            Model::polynomial("p", 3).with_params(vec![1.0, -2.0, 0.5, 0.1]),
            Model::gaussian("g").with_params(vec![3.0, 0.4, 1.3])
                + Model::exponential("e").with_params(vec![0.5, -0.7])
                + custom.clone(),
            Model::gaussian("g")
                .with_params(vec![3.0, 0.4, 1.3])
                .with_gradient(|_, _, g| g.iter_mut().for_each(|v| *v = 7.0)),
        ];
        for m in &models {
            let vg = m.fused_value_grad().expect("a gradient");
            let grad = m.grad.as_deref().expect("a gradient");
            for x in [-1.3, 0.2, 2.9] {
                let mut a = vec![0.0; m.params.len()];
                let mut b = vec![0.0; m.params.len()];
                let v = vg(&[x], &m.params, &mut a);
                grad(&[x], &m.params, &mut b);
                let f = (m.func)(&[x], &m.params);
                assert!(
                    (v - f).abs() <= 1e-12 * f.abs().max(1.0),
                    "{}: {v} vs {f}",
                    m.name
                );
                for (ga, gb) in a.iter().zip(&b) {
                    assert!((ga - gb).abs() <= 1e-12 * gb.abs().max(1.0), "{}", m.name);
                }
            }
        }
        assert!(custom.value_grad.is_none() && models[5].value_grad.is_none());
        assert!(models[4].value_grad.is_some());
    }
}
