//! Parametric curve fitting for **any 1-D data** — not just histograms.
//!
//! A fit needs only, per data point, an independent value `x`, a measured value
//! `y`, and a Gaussian uncertainty `σ`. Anything that can produce those — a
//! histogram's bins, a graph's points, or raw measurements — implements the
//! [`FitData`] trait and gains the [`FitExt`] fit methods:
//!
//! ```
//! use oxifit::{FitExt, Model, Points};
//!
//! // Raw (x, y, σ) measurements; the same API works for a Hist1D or a Graph.
//! let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0], &[0.1; 4])?;
//! let line = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
//! let fit = data.fit(&line); // χ² by default
//! assert!(fit.valid);
//! assert!((fit.params[0] - 1.0).abs() < 1e-6); // intercept ≈ 1
//! assert!((fit.params[1] - 2.0).abs() < 1e-6); // slope ≈ 2
//! # Ok::<(), oxifit::StatError>(())
//! ```
//!
//! The minimizer is the pure-Rust [Minuit2](https://crates.io/crates/minuit2)
//! port (the algorithm ROOT uses). [`Model`] is a named parametric fit function
//! with optional per-parameter limits/fixing/steps — built from a closure, a
//! built-in shape ([`Model::gaussian`]/[`exponential`](Model::exponential)/
//! [`polynomial`](Model::polynomial)), or an arbitrary formula string
//! ([`Model::from_formula`]). Costs: Neyman χ², Pearson χ², or binned Poisson
//! likelihood ([`FitMethod`]). Outlier-resistant fits use a robust [`Loss`]
//! (`Huber`/`SoftL1`/`Cauchy`/`Arctan` — `scipy.optimize.least_squares`'s `loss`)
//! set via [`FitOptions::loss`].
//!
//! Any other `−2 ln L` or χ² — your own likelihood, a data fit plus external
//! measurements ([`CostSum`]) — is a [`Cost`], minimized by a [`Fitter`] with
//! the same minimizers, errors, MINOS, profiles and contours. [`HypoTest`]
//! turns one into discovery significances, CLs values and upper limits.
//!
//! The evaluable, persistable ROOT function objects (`TF1`/`TF2`/`TF3`) live in
//! [`oxiroot_hist_func`](https://crates.io/crates/oxiroot-hist-func) (re-exported
//! as `oxiroot::hist::{Func1D, Func2D, Func3D}`); with that crate's `fit` feature,
//! `Func1D::to_model` converts one to a `Model` for fitting.

mod cost;
mod data;
mod efficiency;
mod engine;
mod fitter;
mod inference;
mod mcmc;
mod model;
mod result;
mod rng;
mod scan;
mod simultaneous;
mod splot;
mod template;
mod toys;

pub use cost::{Cost, CostSum, FnCost, ModelCost};
pub use data::{
    Events, Events2, Events3, FitData, FitDataNd, FitExt, Point, PointNd, Points, Points2, Points3,
};
pub use efficiency::{BinomialCost, Efficiency};
pub use fitter::Fitter;
pub use inference::{
    Cls, Discovery, HypoTest, TestStatistic, ToyCls, ToyDiscovery, UpperLimit, global_significance,
    upcrossings,
};
pub use mcmc::{Chain, Sampler};
pub use model::Model;
/// The error [`Points::new`] and [`Points::unweighted`] return for inputs of
/// different lengths (from `oxistats`).
pub use oxistats::StatError;
pub use result::{
    CmaEs, DifferentialEvolution, FitMethod, FitOptions, FitResult, Loss, Minimizer, MultiStart,
};
pub use scan::{Contour, FitError, Profile};
pub use simultaneous::Simultaneous;
pub use splot::SWeights;
#[cfg(feature = "pyhf")]
pub use template::Workspace;
pub use template::{
    Channel, ChannelYields, HistoInterp, Impact, NormInterp, Sample, TemplateError, TemplateModel,
};
pub use toys::{PullSummary, ToyStudy, Toys};
