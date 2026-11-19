//! Fit configuration ([`FitMethod`], [`FitOptions`]) and outcome ([`FitResult`]).

/// Which cost a fit minimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FitMethod {
    /// Neyman chi-square `Σ (y − f)² / σ²` over points with an error, the
    /// per-point error being the *observed* `σ` (ROOT's default fit). For a
    /// histogram this is `√Sumw2`/`√content`; empty (`σ ≤ 0`) points are dropped.
    #[default]
    Chi2,
    /// Pearson chi-square (ROOT's `"P"`): like [`Chi2`](Self::Chi2) but the
    /// per-point variance is the *expected* (model) value `Σ (y − f)² / f` over
    /// every point — less biased than Neyman at low counts.
    PearsonChi2,
    /// Binned Poisson maximum likelihood (ROOT's `"L"`): minimize the
    /// likelihood-ratio `2·Σ [f − y + y·ln(y/f)]` over every point (which it
    /// treats as a count). Assumes a non-negative model `f`; a model that dips
    /// below zero is clamped to a tiny positive value (heavily penalised, not
    /// rejected). Meaningful for binned counts (histograms), not arbitrary `y`.
    Likelihood,
    /// Unbinned maximum likelihood on individual events ([`Events`](crate::Events)):
    /// minimize `−2·Σ wᵢ·ln(f(xᵢ)/∫f)`, the model normalized numerically over
    /// the fit [`range`](FitOptions::range) (default: the events' span). Only
    /// the model's *shape* matters, so its overall scale is free: fix one
    /// normalization parameter (or fit the yield too with
    /// [`ExtendedUnbinnedLikelihood`](Self::ExtendedUnbinnedLikelihood)).
    /// There is no goodness of fit: `chi2` is `NaN`.
    UnbinnedLikelihood,
    /// Extended unbinned maximum likelihood: minimize
    /// `−2·[Σ wᵢ·ln f(xᵢ) − ∫f]`, the model `f` being the expected event
    /// *density* (events per unit `x`) over the fit range, so `∫f` is the
    /// expected number of events and is fitted too — a sum of yields times
    /// normalized shapes (RooFit's extended `RooAddPdf`). `chi2` is `NaN`.
    ExtendedUnbinnedLikelihood,
}

/// Which optimizer minimizes the cost.
///
/// The local minimizers ([`Minuit2`](Self::Minuit2),
/// [`Minuit2Combined`](Self::Minuit2Combined), [`Simplex`](Self::Simplex)) start
/// from the model's parameters and find the nearest minimum. The global searches
/// ([`MultiStart`](Self::MultiStart),
/// [`DifferentialEvolution`](Self::DifferentialEvolution),
/// [`CmaEs`](Self::CmaEs)) look over a box of
/// parameter values first — useful when the seed may sit in the wrong basin
/// (a peak whose position is unknown, a multimodal model) — and then polish
/// the best point with MIGRAD, so every Minuit-based result carries MIGRAD +
/// HESSE values, errors, covariance and validity.
///
/// The global box is each free parameter's limits when it has both; give the
/// parameters you want searched a [`limit`](crate::Model::limit). A parameter
/// with an open side is searched 10 initial steps beyond its seed on that side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Minimizer {
    /// The pure-Rust [Minuit2](https://crates.io/crates/minuit2) port (MIGRAD) —
    /// ROOT's algorithm. Gives parabolic errors, the covariance matrix, and (on
    /// request) asymmetric MINOS errors. The default, always available.
    #[default]
    Minuit2,
    /// MIGRAD, falling back to SIMPLEX (and MIGRAD again from its result) when
    /// MIGRAD fails — ROOT's `"Minimize"` algorithm. Slower only on failure.
    Minuit2Combined,
    /// Minuit2's SIMPLEX method alone: gradient-free and robust far from the
    /// minimum but slow to converge; HESSE supplies the errors.
    Simplex,
    /// A local minimization (MIGRAD, falling back to SIMPLEX as in
    /// [`Minuit2Combined`](Self::Minuit2Combined)) from the model's seed and
    /// from `starts − 1` more points spread over the search box (a Latin
    /// hypercube); the lowest minimum wins. Costs about `starts` local fits.
    MultiStart(MultiStart),
    /// Differential evolution — a population-based global search (in the
    /// genetic-algorithm family) over the search box, then MIGRAD from the best
    /// point. Costs thousands of cost evaluations, but finds the global
    /// minimum of a multimodal cost far more reliably than one local fit.
    DifferentialEvolution(DifferentialEvolution),
    /// CMA-ES — the covariance matrix adaptation evolution strategy — over the
    /// search box, then MIGRAD from the best point. It learns the shape of the
    /// cost's valley as it searches, so it copes with strongly correlated or
    /// badly scaled parameters better than differential evolution, usually in
    /// fewer evaluations.
    CmaEs(CmaEs),
    /// Levenberg–Marquardt from the model's seed: damped Gauss–Newton steps on
    /// the residuals (each cost is a sum of squared residuals; the likelihood
    /// as signed deviance residuals), limits enforced by projection, then
    /// MIGRAD + HESSE. A local method like MIGRAD — `scipy.optimize.curve_fit`'s
    /// default — that often needs fewer evaluations on least-squares fits.
    LevenbergMarquardt,
    /// L-BFGS-B from the model's seed: a limited-memory quasi-Newton method
    /// with bounds (SciPy's and zfit's gradient minimizer), here in projected
    /// form, then MIGRAD + HESSE. It pays off with an analytic gradient
    /// ([`Model::with_gradient`](crate::Model::with_gradient)) and many
    /// parameters; without one it uses central differences.
    LBfgsB,
    /// Gradient-free Nelder–Mead simplex via the
    /// [`argmin`](https://crates.io/crates/argmin) crate (requires the `argmin`
    /// feature). It moves in Minuit's internal coordinates, so limits hold
    /// exactly; parameter errors come from a numerical Hessian at the minimum,
    /// and MINOS is not available. A useful independent cross-check of Minuit2.
    #[cfg(feature = "argmin")]
    NelderMead,
}

/// Settings for [`Minimizer::MultiStart`].
///
/// ```
/// # use oxifit::{FitOptions, Minimizer, MultiStart};
/// let opts = FitOptions::new().minimizer(Minimizer::MultiStart(MultiStart::new(30).seed(7)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MultiStart {
    /// Number of local minimizations, the model's own seed included (default 20).
    pub starts: usize,
    /// Seed of the generator placing the starts: the same seed gives the same
    /// fit.
    pub seed: u64,
}

impl MultiStart {
    /// `starts` local minimizations (at least one: from the model's seed).
    #[must_use]
    pub fn new(starts: usize) -> MultiStart {
        MultiStart {
            starts,
            seed: DEFAULT_SEED,
        }
    }
    /// Set the generator seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> MultiStart {
        self.seed = seed;
        self
    }
}

impl Default for MultiStart {
    fn default() -> MultiStart {
        MultiStart::new(20)
    }
}

/// Settings for [`Minimizer::DifferentialEvolution`].
///
/// ```
/// # use oxifit::{DifferentialEvolution, FitOptions, Minimizer};
/// let de = DifferentialEvolution::new().population(40).seed(1);
/// let opts = FitOptions::new().minimizer(Minimizer::DifferentialEvolution(de));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DifferentialEvolution {
    /// Population size; `None` (the default) is 15 per free parameter,
    /// between 20 and 300.
    pub population: Option<usize>,
    /// The most generations to evolve (default 1000). The search stops earlier
    /// once every member's cost is within 0.01 of the best.
    pub generations: usize,
    /// Seed of the generator: the same seed gives the same fit.
    pub seed: u64,
}

impl DifferentialEvolution {
    /// The default settings.
    #[must_use]
    pub fn new() -> DifferentialEvolution {
        DifferentialEvolution {
            population: None,
            generations: 1000,
            seed: DEFAULT_SEED,
        }
    }
    /// Set the population size (at least 4).
    #[must_use]
    pub fn population(mut self, n: usize) -> DifferentialEvolution {
        self.population = Some(n);
        self
    }
    /// Set the most generations to evolve.
    #[must_use]
    pub fn generations(mut self, n: usize) -> DifferentialEvolution {
        self.generations = n;
        self
    }
    /// Set the generator seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> DifferentialEvolution {
        self.seed = seed;
        self
    }
}

impl Default for DifferentialEvolution {
    fn default() -> DifferentialEvolution {
        DifferentialEvolution::new()
    }
}

/// Settings for [`Minimizer::CmaEs`].
///
/// ```
/// # use oxifit::{CmaEs, FitOptions, Minimizer};
/// let opts = FitOptions::new().minimizer(Minimizer::CmaEs(CmaEs::new().seed(3)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CmaEs {
    /// Offspring per generation, `λ`; `None` (the default) is 8 per free
    /// parameter, between 16 and 200 — larger than the textbook
    /// `4 + ⌊3·ln n⌋`, which is tuned for local refinement rather than a
    /// global search. A larger population searches more globally, at more
    /// evaluations per generation.
    pub population: Option<usize>,
    /// The most generations (default 1000). The search stops earlier once the
    /// distribution has collapsed or the population's costs agree within 0.01.
    pub generations: usize,
    /// Seed of the generator: the same seed gives the same fit.
    pub seed: u64,
}

impl CmaEs {
    /// The default settings.
    #[must_use]
    pub fn new() -> CmaEs {
        CmaEs {
            population: None,
            generations: 1000,
            seed: DEFAULT_SEED,
        }
    }
    /// Set the population size `λ` (at least 4).
    #[must_use]
    pub fn population(mut self, n: usize) -> CmaEs {
        self.population = Some(n);
        self
    }
    /// Set the most generations.
    #[must_use]
    pub fn generations(mut self, n: usize) -> CmaEs {
        self.generations = n;
        self
    }
    /// Set the generator seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> CmaEs {
        self.seed = seed;
        self
    }
}

impl Default for CmaEs {
    fn default() -> CmaEs {
        CmaEs::new()
    }
}

/// The global searches' default generator seed.
const DEFAULT_SEED: u64 = 0x6F78_6966_6974; // "oxifit"

/// A robust loss `ρ` applied to each squared residual of a least-squares fit
/// ([`Chi2`](FitMethod::Chi2) / [`PearsonChi2`](FitMethod::PearsonChi2)) — the
/// `loss` of [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
/// [`Linear`](Self::Linear) (the default) is an ordinary least-squares fit; the
/// others grow sub-quadratically, so a far-out point (an outlier) pulls on the
/// fit far less than its squared residual would suggest. Set the residual scale
/// at which a point starts to count as an outlier with
/// [`FitOptions::f_scale`](FitOptions::f_scale).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Loss {
    /// Ordinary least squares, `ρ(z) = z` (no down-weighting). The default.
    #[default]
    Linear,
    /// Smooth-L1, `ρ(z) = 2(√(1 + z) − 1)` — a gentle, everywhere-smooth robustness.
    SoftL1,
    /// Huber, `ρ(z) = z` for `z ≤ 1`, else `2√z − 1` — a quadratic core with
    /// linear tails.
    Huber,
    /// Cauchy / Lorentzian, `ρ(z) = ln(1 + z)` — strong outlier suppression.
    Cauchy,
    /// Arctan, `ρ(z) = arctan(z)` — the most aggressive (bounded influence).
    Arctan,
}

/// Options controlling a fit ([`FitExt::fit_opts`](crate::FitExt::fit_opts)).
/// Construct with [`new`](Self::new) and the chainable setters; the defaults are
/// a full-range chi-square fit minimized with [`Minimizer::Minuit2`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct FitOptions {
    /// The cost to minimize.
    pub method: FitMethod,
    /// Restrict the fit to points whose `x` lies in `[lo, hi]`.
    pub range: Option<(f64, f64)>,
    /// For 2-D and 3-D data: restrict the fit to points whose `y` lies in
    /// `[lo, hi]` (and normalize unbinned fits over it).
    pub range_y: Option<(f64, f64)>,
    /// For 3-D data: restrict the fit to points whose `z` lies in `[lo, hi]`.
    pub range_z: Option<(f64, f64)>,
    /// Also compute asymmetric [MINOS](https://root.cern/doc/master/classTMinuit.html)
    /// errors for each free parameter (a likelihood scan — more accurate than the
    /// parabolic errors near a non-quadratic minimum, but extra work).
    /// Ignored by the [`NelderMead`](Minimizer::NelderMead) backend.
    pub minos: bool,
    /// Run HESSE after MIGRAD (default `true`, as ROOT's `Fit` does): the
    /// errors and covariance then come from a finite-difference Hessian at the
    /// minimum rather than MIGRAD's running estimate. Turn off only for speed
    /// (e.g. many toy fits where only the best-fit values matter). Ignored by
    /// the [`NelderMead`](Minimizer::NelderMead) backend, which always computes
    /// a numerical Hessian.
    pub hesse: bool,
    /// Which optimizer backend to use.
    pub minimizer: Minimizer,
    /// Correct the covariance of a fit to *weighted* events for the weights
    /// (RooFit's `SumW2Error`): `V = H⁻¹·W·H⁻¹`, `H` the Hessian of the
    /// weighted likelihood and `W` that of the same likelihood with squared
    /// weights. Without it a weighted fit's errors are those of `Σw` unit
    /// events. Default `false`; applies to the unbinned likelihoods (MINOS
    /// errors are not corrected).
    pub sumw2: bool,
    /// Use the model's analytic gradient ([`Model::with_gradient`](crate::Model::with_gradient))
    /// when it has one (default `true`): MIGRAD takes exact cost derivatives,
    /// Levenberg–Marquardt an exact Jacobian, L-BFGS-B exact gradients. Turn
    /// off to compare with finite differences.
    pub gradient: bool,
    /// Minuit2 strategy: `0` (fast, fewer derivative calls), `1` (default),
    /// `2` (careful: more accurate derivatives and covariance, slower) — ROOT's
    /// `SetStrategy`. Applies to MIGRAD, HESSE and MINOS.
    pub strategy: u32,
    /// Minuit2 convergence tolerance: MIGRAD stops once the estimated distance
    /// to the minimum is below `0.002 × tolerance` (default `0.1`, ROOT's
    /// default). Lower it to locate the minimum more precisely.
    pub tolerance: f64,
    /// Cap on the cost evaluations of one Minuit2 minimization; `None` (the
    /// default) is Minuit2's own `200 + 100·n + 5·n²` for `n` free parameters.
    pub max_fcn: Option<usize>,
    /// Robust [`Loss`] applied to each squared residual (the χ² costs only;
    /// ignored by the likelihoods).
    pub loss: Loss,
    /// The residual scale `C` for a robust [`loss`](Self::loss): a point begins to
    /// be treated as an outlier once its residual exceeds ~`C`. Default `1.0`;
    /// irrelevant to [`Loss::Linear`].
    pub f_scale: f64,
}

impl Default for FitOptions {
    fn default() -> FitOptions {
        FitOptions {
            method: FitMethod::default(),
            range: None,
            range_y: None,
            range_z: None,
            minos: false,
            hesse: true,
            sumw2: false,
            gradient: true,
            minimizer: Minimizer::default(),
            strategy: 1,
            tolerance: 0.1,
            max_fcn: None,
            loss: Loss::default(),
            f_scale: 1.0,
        }
    }
}

impl FitOptions {
    /// Default options: a full-range chi-square fit.
    #[must_use]
    pub fn new() -> FitOptions {
        FitOptions::default()
    }
    /// Set the fit cost ([`FitMethod`]).
    #[must_use]
    pub fn method(mut self, method: FitMethod) -> FitOptions {
        self.method = method;
        self
    }
    /// Fit only the points whose `x` lies in `[lo, hi]`.
    #[must_use]
    pub fn range(mut self, lo: f64, hi: f64) -> FitOptions {
        self.range = Some((lo, hi));
        self
    }
    /// For 2-D and 3-D data: fit only the points whose `y` lies in `[lo, hi]`.
    #[must_use]
    pub fn range_y(mut self, lo: f64, hi: f64) -> FitOptions {
        self.range_y = Some((lo, hi));
        self
    }
    /// For 3-D data: fit only the points whose `z` lies in `[lo, hi]`.
    #[must_use]
    pub fn range_z(mut self, lo: f64, hi: f64) -> FitOptions {
        self.range_z = Some((lo, hi));
        self
    }
    /// Also compute asymmetric MINOS errors (see [`minos`](Self::minos) field).
    #[must_use]
    pub fn with_minos(mut self, on: bool) -> FitOptions {
        self.minos = on;
        self
    }
    /// Run HESSE after MIGRAD (see the [`hesse`](Self::hesse) field; on by default).
    #[must_use]
    pub fn with_hesse(mut self, on: bool) -> FitOptions {
        self.hesse = on;
        self
    }
    /// Correct a weighted fit's covariance for the weights (see the
    /// [`sumw2`](Self::sumw2) field).
    #[must_use]
    pub fn with_sumw2(mut self, on: bool) -> FitOptions {
        self.sumw2 = on;
        self
    }
    /// Use the model's analytic gradient when it has one (see the
    /// [`gradient`](Self::gradient) field; on by default).
    #[must_use]
    pub fn with_gradient(mut self, on: bool) -> FitOptions {
        self.gradient = on;
        self
    }
    /// Choose the optimizer backend ([`Minimizer`]).
    #[must_use]
    pub fn minimizer(mut self, minimizer: Minimizer) -> FitOptions {
        self.minimizer = minimizer;
        self
    }
    /// Set the Minuit2 strategy (see [`strategy`](Self::strategy)).
    #[must_use]
    pub fn strategy(mut self, level: u32) -> FitOptions {
        self.strategy = level;
        self
    }
    /// Set the Minuit2 convergence tolerance (see [`tolerance`](Self::tolerance)).
    #[must_use]
    pub fn tolerance(mut self, tolerance: f64) -> FitOptions {
        self.tolerance = tolerance;
        self
    }
    /// Cap the cost evaluations of one Minuit2 minimization.
    #[must_use]
    pub fn max_fcn(mut self, max: usize) -> FitOptions {
        self.max_fcn = Some(max);
        self
    }
    /// Use a robust [`Loss`] to down-weight outliers (least-squares costs only).
    #[must_use]
    pub fn loss(mut self, loss: Loss) -> FitOptions {
        self.loss = loss;
        self
    }
    /// Set the robust-loss residual scale (see [`f_scale`](Self::f_scale)).
    #[must_use]
    pub fn f_scale(mut self, f_scale: f64) -> FitOptions {
        self.f_scale = f_scale;
        self
    }
}

/// The outcome of a fit ([`FitExt::fit`](crate::FitExt::fit)).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FitResult {
    /// Best-fit parameter values (in the model's parameter order).
    pub params: Vec<f64>,
    /// Parabolic uncertainties on each parameter (from HESSE unless it was
    /// turned off with [`FitOptions::with_hesse`]); `0` for a fixed parameter.
    pub errors: Vec<f64>,
    /// Asymmetric MINOS errors `(lower, upper)` per parameter (`lower ≤ 0 ≤ upper`),
    /// when requested via [`FitOptions::minos`]; `None` otherwise. A fixed
    /// parameter reports `(0.0, 0.0)`.
    pub minos: Option<Vec<(f64, f64)>>,
    /// Covariance matrix of the *free* (non-fixed) parameters, in their parameter
    /// order (row-major), when Minuit2 produced one; `None` otherwise. With no
    /// fixed parameters this is the full parameter covariance.
    pub covariance: Option<Vec<Vec<f64>>>,
    /// Goodness-of-fit statistic at the minimum: the χ² (Neyman or Pearson) or,
    /// for [`FitMethod::Likelihood`], the Poisson likelihood ratio
    /// `2·Σ [f − y + y·ln(y/f)]` (χ²-distributed for large counts). `NaN` under
    /// a robust [`Loss`], whose cost is no χ² — read [`fval`](Self::fval) then.
    pub chi2: f64,
    /// The minimized cost itself (equal to [`chi2`](Self::chi2) without a
    /// robust loss).
    pub fval: f64,
    /// Minuit2's estimated distance to the minimum (EDM) — how far, in cost,
    /// the true minimum may still lie; `NaN` from a backend without one.
    pub edm: f64,
    /// Cost evaluations the fit took in all: the minimization (a global
    /// search's included), HESSE and MINOS.
    pub nfcn: usize,
    /// Degrees of freedom: fitted points + Gaussian constraints
    /// ([`Model::constrain`](crate::Model::constrain)) − free parameters.
    pub ndf: usize,
    /// Whether the minimizer reported a valid minimum.
    pub valid: bool,
}

impl FitResult {
    /// A result from its values — e.g. a fit stored on disk, or one built by
    /// hand for a plot. The cost [`fval`](Self::fval) is `chi2`; there are no
    /// MINOS errors or covariance, the EDM is `NaN` and `nfcn` is `0` (set
    /// the public fields to fill them in).
    #[must_use]
    pub fn new(
        params: Vec<f64>,
        errors: Vec<f64>,
        chi2: f64,
        ndf: usize,
        valid: bool,
    ) -> FitResult {
        FitResult {
            params,
            errors,
            minos: None,
            covariance: None,
            chi2,
            fval: chi2,
            edm: f64::NAN,
            nfcn: 0,
            ndf,
            valid,
        }
    }

    /// Reduced chi-square `chi2 / ndf`, or `NaN` when `ndf == 0` (an
    /// under-determined fit has no meaningful reduced chi-square).
    #[must_use]
    pub fn chi2_per_ndf(&self) -> f64 {
        if self.ndf == 0 {
            f64::NAN
        } else {
            self.chi2 / self.ndf as f64
        }
    }

    /// Goodness-of-fit p-value: the probability of a chi-square at least this
    /// large for `ndf` degrees of freedom (a good fit is near 1, a poor one near
    /// 0). For a likelihood fit this is the asymptotic value via the
    /// likelihood-ratio (Wilks' theorem). `NaN` for an invalid fit, and under a
    /// robust loss (no χ²).
    #[must_use]
    pub fn p_value(&self) -> f64 {
        if !self.valid || self.ndf == 0 || !self.chi2.is_finite() {
            f64::NAN
        } else {
            oxistats::chi_square_prob(self.chi2, self.ndf)
        }
    }
}
