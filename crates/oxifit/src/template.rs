//! Binned template fits with systematic uncertainties — the HistFactory model
//! of ROOT/RooStats and pyhf.
//!
//! A [`TemplateModel`] is a set of [`Channel`]s, each a histogram of observed
//! counts and the [`Sample`]s expected in it (signal, backgrounds — templates
//! from simulation), each sample scaled and deformed by its modifiers:
//!
//! | Modifier | Effect | Constraint |
//! |---|---|---|
//! | [`norm_factor`](Sample::norm_factor) | `× μ` (free) | — |
//! | [`norm_sys`](Sample::norm_sys) | `× κ(α)`, interpolating the ±1σ factors | Gaussian on `α` |
//! | [`histo_sys`](Sample::histo_sys) | `+ Δ(α)` per bin, interpolating the ±1σ templates | Gaussian on `α` |
//! | [`stat_error`](Sample::stat_error) | `× γ_b` per bin (finite simulation statistics, Barlow–Beeston-lite) | Gaussian on each `γ_b` |
//! | [`shape_sys`](Sample::shape_sys) | `× γ_b` per bin, uncorrelated | Poisson on each `γ_b` |
//! | [`lumi`](Sample::lumi) | `× λ` | Gaussian on `λ` |
//! | [`shape_factor`](Sample::shape_factor) | `× γ_b` per bin (free) | — |
//!
//! Modifiers with the same name are the same parameter, across samples and
//! channels. The model is a [`Cost`]: `−2 ln L` over every bin (Poisson) and
//! every auxiliary measurement, so it is fitted by a [`Fitter`] (with
//! [`fitter`](TemplateModel::fitter)'s standard seeds and bounds), tested with
//! [`HypoTest`](crate::HypoTest), and summarized by [`pulls`](TemplateModel::pulls),
//! [`impacts`](TemplateModel::impacts) and [`yields`](TemplateModel::yields).
//! The interpolations are pyhf's (and ROOT's) codes; the defaults are pyhf's.

use crate::cost::Cost;
use crate::engine::linalg;
use crate::fitter::Fitter;
use crate::result::{FitOptions, FitResult};
use crate::rng::Rng;
use crate::scan::FitError;

/// How a [`norm_sys`](Sample::norm_sys) factor is interpolated between its
/// ±1σ values `hi` and `lo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum NormInterp {
    /// Exponential: `hi^α` for `α ≥ 0`, `lo^(−α)` below (code 1).
    Exponential,
    /// Exponential beyond `|α| = 1`, a sixth-order polynomial inside, matching
    /// value and first two derivatives at ±1 (code 4, pyhf's default).
    #[default]
    Polynomial,
}

/// How a [`histo_sys`](Sample::histo_sys) template is interpolated between
/// its ±1σ templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HistoInterp {
    /// Piecewise linear (code 0).
    Linear,
    /// Linear beyond `|α| = 1`, a polynomial inside with matching first
    /// derivatives (code 4p, pyhf's default).
    #[default]
    Polynomial,
}

/// One modifier of a sample.
#[derive(Debug, Clone, PartialEq)]
enum Modifier {
    NormFactor(String),
    NormSys {
        name: String,
        lo: f64,
        hi: f64,
    },
    HistoSys {
        name: String,
        lo: Vec<f64>,
        hi: Vec<f64>,
    },
    StatError {
        name: String,
        uncertainty: Vec<f64>,
    },
    ShapeSys {
        name: String,
        uncertainty: Vec<f64>,
    },
    Lumi(String),
    ShapeFactor(String),
}

/// A sample: a template of expected counts per bin and its modifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    name: String,
    nominal: Vec<f64>,
    modifiers: Vec<Modifier>,
}

impl Sample {
    /// A sample with its nominal expected counts per bin.
    #[must_use]
    pub fn new(name: &str, nominal: &[f64]) -> Sample {
        Sample {
            name: name.to_string(),
            nominal: nominal.to_vec(),
            modifiers: Vec::new(),
        }
    }

    /// Scale by a free normalization (a signal strength): seed 1, bounds [0, 10].
    #[must_use]
    pub fn norm_factor(mut self, name: &str) -> Sample {
        self.modifiers.push(Modifier::NormFactor(name.to_string()));
        self
    }

    /// A normalization uncertainty: the sample scales by `hi` at `α = +1` and
    /// by `lo` at `α = −1` (e.g. 1.05 and 0.95 for ±5 %).
    #[must_use]
    pub fn norm_sys(mut self, name: &str, lo: f64, hi: f64) -> Sample {
        self.modifiers.push(Modifier::NormSys {
            name: name.to_string(),
            lo,
            hi,
        });
        self
    }

    /// A shape uncertainty: the templates at `α = −1` and `α = +1`.
    #[must_use]
    pub fn histo_sys(mut self, name: &str, lo: &[f64], hi: &[f64]) -> Sample {
        self.modifiers.push(Modifier::HistoSys {
            name: name.to_string(),
            lo: lo.to_vec(),
            hi: hi.to_vec(),
        });
        self
    }

    /// The template's statistical uncertainty per bin (absolute): one factor
    /// `γ_b` per bin, shared by every sample of the channel with the same
    /// `name`, constrained by their combined relative uncertainty.
    #[must_use]
    pub fn stat_error(mut self, name: &str, uncertainty: &[f64]) -> Sample {
        self.modifiers.push(Modifier::StatError {
            name: name.to_string(),
            uncertainty: uncertainty.to_vec(),
        });
        self
    }

    /// An uncorrelated per-bin uncertainty (absolute), with a Poisson
    /// constraint.
    #[must_use]
    pub fn shape_sys(mut self, name: &str, uncertainty: &[f64]) -> Sample {
        self.modifiers.push(Modifier::ShapeSys {
            name: name.to_string(),
            uncertainty: uncertainty.to_vec(),
        });
        self
    }

    /// Scale by the luminosity parameter (its uncertainty set by
    /// [`TemplateModel::lumi_uncertainty`]).
    #[must_use]
    pub fn lumi(mut self, name: &str) -> Sample {
        self.modifiers.push(Modifier::Lumi(name.to_string()));
        self
    }

    /// A free factor per bin (a data-driven shape).
    #[must_use]
    pub fn shape_factor(mut self, name: &str) -> Sample {
        self.modifiers.push(Modifier::ShapeFactor(name.to_string()));
        self
    }
}

/// A channel: observed counts per bin and the samples expected in them.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    name: String,
    observed: Vec<f64>,
    samples: Vec<Sample>,
}

impl Channel {
    /// A channel with its observed counts.
    #[must_use]
    pub fn new(name: &str, observed: &[f64]) -> Channel {
        Channel {
            name: name.to_string(),
            observed: observed.to_vec(),
            samples: Vec::new(),
        }
    }

    /// Add a sample.
    #[must_use]
    pub fn sample(mut self, sample: Sample) -> Channel {
        self.samples.push(sample);
        self
    }

    /// The channel's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number of bins.
    #[must_use]
    pub fn bins(&self) -> usize {
        self.observed.len()
    }
}

/// What kind of parameter, and how it is constrained.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// Free normalization or per-bin factor.
    Free,
    /// A Gaussian-constrained nuisance parameter of width `sigma` around its
    /// auxiliary measurement (α: 0 ± 1; γ: 1 ± σ; λ: 1 ± σ).
    Gauss { sigma: f64 },
    /// A Poisson-constrained factor: auxiliary count `tau` at `γ = 1`.
    Poisson { tau: f64 },
}

#[derive(Debug, Clone, PartialEq)]
struct Param {
    name: String,
    kind: Kind,
    init: f64,
    bounds: (f64, f64),
    /// Fixed (a γ for a bin with no expected events).
    fixed: bool,
}

/// One modifier, resolved to parameter indices.
#[derive(Debug, Clone, PartialEq)]
enum Resolved {
    Factor(usize),
    NormSys {
        p: usize,
        lo: f64,
        hi: f64,
        poly: Option<[f64; 6]>,
    },
    HistoSys {
        p: usize,
        lo: Vec<f64>,
        hi: Vec<f64>,
    },
    PerBin(Vec<usize>),
}

/// A binned likelihood of channels, samples and modifiers — a [`Cost`]; the
/// HistFactory model of ROOT/RooStats and pyhf.
///
/// A [`TemplateModel`] is a set of [`Channel`]s, each a histogram of observed
/// counts and the [`Sample`]s expected in it (signal, backgrounds — templates
/// from simulation), each sample scaled and deformed by its modifiers:
///
/// | Modifier | Effect | Constraint |
/// |---|---|---|
/// | [`norm_factor`](Sample::norm_factor) | `× μ` (free) | — |
/// | [`norm_sys`](Sample::norm_sys) | `× κ(α)`, interpolating the ±1σ factors | Gaussian on `α` |
/// | [`histo_sys`](Sample::histo_sys) | `+ Δ(α)` per bin, interpolating the ±1σ templates | Gaussian on `α` |
/// | [`stat_error`](Sample::stat_error) | `× γ_b` per bin (finite simulation statistics, Barlow–Beeston-lite) | Gaussian on each `γ_b` |
/// | [`shape_sys`](Sample::shape_sys) | `× γ_b` per bin, uncorrelated | Poisson on each `γ_b` |
/// | [`lumi`](Sample::lumi) | `× λ` | Gaussian on `λ` |
/// | [`shape_factor`](Sample::shape_factor) | `× γ_b` per bin (free) | — |
///
/// Modifiers with the same name are the same parameter, across samples and
/// channels. The model is a [`Cost`]: `−2 ln L` over every bin (Poisson) and
/// every auxiliary measurement, so it is fitted by a [`Fitter`] (with
/// [`fitter`](TemplateModel::fitter)'s standard seeds and bounds), tested with
/// [`HypoTest`](crate::HypoTest), and summarized by [`pulls`](TemplateModel::pulls),
/// [`impacts`](TemplateModel::impacts) and [`yields`](TemplateModel::yields).
/// The interpolations are pyhf's (and ROOT's) codes; the defaults are pyhf's.
///
/// ```
/// use oxifit::{Channel, FitOptions, Sample, TemplateModel};
///
/// let model = TemplateModel::new(vec![Channel::new("SR", &[52.0, 64.0])
///     .sample(Sample::new("signal", &[5.0, 10.0]).norm_factor("mu"))
///     .sample(
///         Sample::new("background", &[50.0, 52.0])
///             .norm_sys("bkg_norm", 0.9, 1.1)
///             .stat_error("mc_stat", &[3.0, 4.0]),
///     )])?;
/// assert_eq!(model.param_names(), ["mu", "bkg_norm", "mc_stat[0]", "mc_stat[1]"]);
/// let fit = model.fitter().fit(&FitOptions::new());
/// assert!(fit.valid);
/// # Ok::<(), oxifit::FitError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateModel {
    channels: Vec<Channel>,
    params: Vec<Param>,
    /// Per channel, per sample, the resolved modifiers.
    resolved: Vec<Vec<Vec<Resolved>>>,
    /// The auxiliary measurements, one per constrained parameter (by index).
    aux: Vec<f64>,
    norm_interp: NormInterp,
    histo_interp: HistoInterp,
    lumi_sigma: f64,
}

/// The polynomial of interpolation code 4 between `lo` and `hi`: the
/// coefficients `a₁…a₆` of `κ(α) − 1 = Σ aᵢ αⁱ` on `|α| < 1`.
fn code4_poly(lo: f64, hi: f64) -> Option<[f64; 6]> {
    if !(lo > 0.0 && hi > 0.0) {
        return None;
    }
    let (lh, ll) = (hi.ln(), lo.ln());
    // Rows: f(1), f(−1), f'(1), f'(−1), f''(1), f''(−1) of Σ aᵢ αⁱ.
    let mut a = vec![vec![0.0; 6]; 6];
    for i in 1..=6 {
        let fi = i as f64;
        let s = if i % 2 == 0 { 1.0 } else { -1.0 };
        a[0][i - 1] = 1.0;
        a[1][i - 1] = s;
        a[2][i - 1] = fi;
        a[3][i - 1] = -s * fi;
        a[4][i - 1] = fi * (fi - 1.0);
        a[5][i - 1] = s * fi * (fi - 1.0);
    }
    let b = [
        hi - 1.0,
        lo - 1.0,
        hi * lh,
        -lo * ll,
        hi * lh * lh,
        lo * ll * ll,
    ];
    let x = linalg::solve(&a, &b)?;
    Some([x[0], x[1], x[2], x[3], x[4], x[5]])
}

impl TemplateModel {
    /// A model of these channels.
    ///
    /// # Errors
    /// [`FitError`] for a template of the wrong number of bins, or one
    /// name used for modifiers of different kinds.
    pub fn new(channels: Vec<Channel>) -> Result<TemplateModel, FitError> {
        let mut m = TemplateModel {
            channels,
            params: Vec::new(),
            resolved: Vec::new(),
            aux: Vec::new(),
            norm_interp: NormInterp::default(),
            histo_interp: HistoInterp::default(),
            lumi_sigma: 0.0,
        };
        m.resolve()?;
        Ok(m)
    }

    /// The interpolation codes (default: pyhf's, polynomial for both).
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn interpolation(
        mut self,
        norm: NormInterp,
        histo: HistoInterp,
    ) -> Result<TemplateModel, FitError> {
        self.norm_interp = norm;
        self.histo_interp = histo;
        self.resolve()?;
        Ok(self)
    }

    /// The luminosity parameter's relative uncertainty (a Gaussian
    /// constraint `λ = 1 ± sigma`; without one, λ is free).
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn lumi_uncertainty(mut self, sigma: f64) -> Result<TemplateModel, FitError> {
        self.lumi_sigma = sigma;
        self.resolve()?;
        Ok(self)
    }

    /// Register parameter `name` (or check an existing one has the same
    /// kind) and return its index.
    fn param(
        params: &mut Vec<Param>,
        name: String,
        kind: Kind,
        init: f64,
        bounds: (f64, f64),
        tag: &str,
        tags: &mut Vec<(String, String)>,
    ) -> Result<usize, FitError> {
        if let Some(i) = params.iter().position(|p| p.name == name) {
            let same = tags.iter().any(|(n, t)| *n == name && t == tag);
            // normsys and histosys may share an α.
            let compatible = matches!(tag, "normsys" | "histosys")
                && tags
                    .iter()
                    .any(|(n, t)| *n == name && (t == "normsys" || t == "histosys"));
            if !same && !compatible {
                return Err(FitError::KindConflict(name));
            }
            return Ok(i);
        }
        params.push(Param {
            name: name.clone(),
            kind,
            init,
            bounds,
            fixed: false,
        });
        tags.push((name, tag.to_string()));
        Ok(params.len() - 1)
    }

    /// Resolve every modifier to parameters, and set the auxiliary data.
    fn resolve(&mut self) -> Result<(), FitError> {
        let mut params: Vec<Param> = Vec::new();
        let mut tags: Vec<(String, String)> = Vec::new();
        let mut resolved = Vec::new();
        for ch in &self.channels {
            let nb = ch.bins();
            let mismatch = |s: &Sample| FitError::BinMismatch {
                channel: ch.name.clone(),
                sample: s.name.clone(),
            };
            // The combined relative statistical uncertainty per staterror name.
            let mut stat: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new(); // (name, Σδ², Σν)
            for s in &ch.samples {
                if s.nominal.len() != nb {
                    return Err(mismatch(s));
                }
                for m in &s.modifiers {
                    if let Modifier::StatError { name, uncertainty } = m {
                        if uncertainty.len() != nb {
                            return Err(mismatch(s));
                        }
                        let k = stat.iter().position(|e| e.0 == *name).unwrap_or_else(|| {
                            stat.push((name.clone(), vec![0.0; nb], vec![0.0; nb]));
                            stat.len() - 1
                        });
                        for b in 0..nb {
                            stat[k].1[b] += uncertainty[b] * uncertainty[b];
                            stat[k].2[b] += s.nominal[b];
                        }
                    }
                }
            }
            let mut ch_res = Vec::new();
            for s in &ch.samples {
                let mut s_res = Vec::new();
                for m in &s.modifiers {
                    let r = match m {
                        Modifier::NormFactor(n) => Resolved::Factor(Self::param(
                            &mut params,
                            n.clone(),
                            Kind::Free,
                            1.0,
                            (0.0, 10.0),
                            "normfactor",
                            &mut tags,
                        )?),
                        Modifier::NormSys { name, lo, hi } => {
                            let p = Self::param(
                                &mut params,
                                name.clone(),
                                Kind::Gauss { sigma: 1.0 },
                                0.0,
                                (-5.0, 5.0),
                                "normsys",
                                &mut tags,
                            )?;
                            let poly = match self.norm_interp {
                                NormInterp::Polynomial => code4_poly(*lo, *hi),
                                NormInterp::Exponential => None,
                            };
                            Resolved::NormSys {
                                p,
                                lo: *lo,
                                hi: *hi,
                                poly,
                            }
                        }
                        Modifier::HistoSys { name, lo, hi } => {
                            if lo.len() != nb || hi.len() != nb {
                                return Err(mismatch(s));
                            }
                            let p = Self::param(
                                &mut params,
                                name.clone(),
                                Kind::Gauss { sigma: 1.0 },
                                0.0,
                                (-5.0, 5.0),
                                "histosys",
                                &mut tags,
                            )?;
                            Resolved::HistoSys {
                                p,
                                lo: lo.clone(),
                                hi: hi.clone(),
                            }
                        }
                        Modifier::StatError { name, .. } => {
                            let (_, var, nom) =
                                stat.iter().find(|e| e.0 == *name).expect("collected above");
                            let mut idx = Vec::with_capacity(nb);
                            for b in 0..nb {
                                let sigma = if nom[b] > 0.0 {
                                    var[b].sqrt() / nom[b]
                                } else {
                                    0.0
                                };
                                let i = Self::param(
                                    &mut params,
                                    format!("{name}[{b}]"),
                                    Kind::Gauss { sigma },
                                    1.0,
                                    (1e-10, 10.0),
                                    "staterror",
                                    &mut tags,
                                )?;
                                if sigma.is_nan() || sigma <= 0.0 {
                                    params[i].fixed = true;
                                }
                                idx.push(i);
                            }
                            Resolved::PerBin(idx)
                        }
                        Modifier::ShapeSys { name, uncertainty } => {
                            if uncertainty.len() != nb {
                                return Err(mismatch(s));
                            }
                            let mut idx = Vec::with_capacity(nb);
                            for b in 0..nb {
                                let rel = if s.nominal[b] > 0.0 {
                                    uncertainty[b] / s.nominal[b]
                                } else {
                                    0.0
                                };
                                let tau = if rel > 0.0 { 1.0 / (rel * rel) } else { 0.0 };
                                let i = Self::param(
                                    &mut params,
                                    format!("{name}[{b}]"),
                                    Kind::Poisson { tau },
                                    1.0,
                                    (1e-10, 10.0),
                                    "shapesys",
                                    &mut tags,
                                )?;
                                if tau.is_nan() || tau <= 0.0 {
                                    params[i].fixed = true;
                                }
                                idx.push(i);
                            }
                            Resolved::PerBin(idx)
                        }
                        Modifier::Lumi(n) => {
                            let kind = if self.lumi_sigma > 0.0 {
                                Kind::Gauss {
                                    sigma: self.lumi_sigma,
                                }
                            } else {
                                Kind::Free
                            };
                            Resolved::Factor(Self::param(
                                &mut params,
                                n.clone(),
                                kind,
                                1.0,
                                (0.0, 10.0),
                                "lumi",
                                &mut tags,
                            )?)
                        }
                        Modifier::ShapeFactor(n) => {
                            let mut idx = Vec::with_capacity(nb);
                            for b in 0..nb {
                                idx.push(Self::param(
                                    &mut params,
                                    format!("{n}[{b}]"),
                                    Kind::Free,
                                    1.0,
                                    (0.0, 10.0),
                                    "shapefactor",
                                    &mut tags,
                                )?);
                            }
                            Resolved::PerBin(idx)
                        }
                    };
                    s_res.push(r);
                }
                ch_res.push(s_res);
            }
            resolved.push(ch_res);
        }
        self.aux = params
            .iter()
            .map(|p| match p.kind {
                Kind::Free => f64::NAN,
                Kind::Gauss { .. } => p.init,
                Kind::Poisson { tau } => tau,
            })
            .collect();
        self.params = params;
        self.resolved = resolved;
        Ok(())
    }

    /// The parameter names, in [`Cost`] order.
    #[must_use]
    pub fn param_names(&self) -> Vec<String> {
        self.params.iter().map(|p| p.name.clone()).collect()
    }

    /// The channels.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// The standard seeds (1 for factors, 0 for α).
    #[must_use]
    pub fn inits(&self) -> Vec<f64> {
        self.params.iter().map(|p| p.init).collect()
    }

    /// A [`Fitter`] with the standard seeds and bounds: factors in [0, 10],
    /// α in [−5, 5], γ in (0, 10]; a γ of an empty bin fixed at 1.
    #[must_use]
    pub fn fitter(&self) -> Fitter<'_> {
        let mut f = Fitter::new(self, self.inits());
        for p in &self.params {
            f = f.limit(&p.name, p.bounds.0, p.bounds.1);
            if p.fixed {
                f = f.fix(&p.name);
            }
        }
        f
    }

    /// `κ(α)` of a normsys.
    fn kappa(&self, alpha: f64, lo: f64, hi: f64, poly: Option<&[f64; 6]>) -> f64 {
        match poly {
            Some(a) if alpha.abs() < 1.0 => {
                let mut v = 0.0;
                for c in a.iter().rev() {
                    v = (v + c) * alpha;
                }
                1.0 + v
            }
            _ => {
                if alpha >= 0.0 {
                    hi.powf(alpha)
                } else {
                    lo.powf(-alpha)
                }
            }
        }
    }

    /// `Δ(α)` of a histosys in one bin.
    fn delta(&self, alpha: f64, nominal: f64, lo: f64, hi: f64) -> f64 {
        let (up, down) = (hi - nominal, nominal - lo);
        match self.histo_interp {
            HistoInterp::Polynomial if alpha.abs() <= 1.0 => {
                let s = 0.5 * (up + down);
                let a = 0.0625 * (up - down);
                alpha * (s + alpha * a * (15.0 + alpha * alpha * (3.0 * alpha * alpha - 10.0)))
            }
            _ => {
                if alpha >= 0.0 {
                    alpha * up
                } else {
                    alpha * down
                }
            }
        }
    }

    /// The expected counts at `params`: `[channel][sample][bin]`.
    #[must_use]
    pub fn expected(&self, params: &[f64]) -> Vec<Vec<Vec<f64>>> {
        self.channels
            .iter()
            .zip(&self.resolved)
            .map(|(ch, ch_res)| {
                ch.samples
                    .iter()
                    .zip(ch_res)
                    .map(|(s, mods)| {
                        let mut nu = s.nominal.clone();
                        // Additive first (shape templates), then factors.
                        for m in mods {
                            if let Resolved::HistoSys { p, lo, hi } = m {
                                for (b, v) in nu.iter_mut().enumerate() {
                                    *v += self.delta(params[*p], s.nominal[b], lo[b], hi[b]);
                                }
                            }
                        }
                        let mut scale = 1.0;
                        for m in mods {
                            match m {
                                Resolved::Factor(p) => scale *= params[*p],
                                Resolved::NormSys { p, lo, hi, poly } => {
                                    scale *= self.kappa(params[*p], *lo, *hi, poly.as_ref());
                                }
                                Resolved::PerBin(idx) => {
                                    for (v, &i) in nu.iter_mut().zip(idx) {
                                        *v *= params[i];
                                    }
                                }
                                Resolved::HistoSys { .. } => {}
                            }
                        }
                        nu.iter_mut().for_each(|v| *v *= scale);
                        nu
                    })
                    .collect()
            })
            .collect()
    }

    /// The total expected counts per channel and bin.
    fn totals(&self, params: &[f64]) -> Vec<Vec<f64>> {
        self.expected(params)
            .into_iter()
            .map(|samples| {
                let nb = samples.first().map_or(0, Vec::len);
                (0..nb)
                    .map(|b| samples.iter().map(|s| s[b]).sum())
                    .collect()
            })
            .collect()
    }

    /// The constraint terms of `−2 ln L` at `params` given auxiliary data.
    fn constraint_cost(&self, params: &[f64], aux: &[f64]) -> f64 {
        self.params
            .iter()
            .enumerate()
            .map(|(i, p)| match p.kind {
                Kind::Free => 0.0,
                Kind::Gauss { sigma } if sigma > 0.0 => ((params[i] - aux[i]) / sigma).powi(2),
                Kind::Gauss { .. } => 0.0,
                Kind::Poisson { tau } if tau > 0.0 => {
                    let rate = (params[i] * tau).max(1e-300);
                    let n = aux[i];
                    2.0 * (rate - n + if n > 0.0 { n * (n / rate).ln() } else { 0.0 })
                }
                Kind::Poisson { .. } => 0.0,
            })
            .sum()
    }

    /// The same model on other data (observed counts and auxiliary data).
    fn with_data(&self, observed: Vec<Vec<f64>>, aux: Vec<f64>) -> TemplateModel {
        let mut m = self.clone();
        for (ch, obs) in m.channels.iter_mut().zip(observed) {
            ch.observed = obs;
        }
        m.aux = aux;
        m
    }

    /// The constrained nuisance parameters' pulls after `fit`: `(name,
    /// (θ̂ − θ₀)/σ₀, σ̂/σ₀)` — the shift in units of the prior width and the
    /// post-fit error relative to it (1: the data add nothing).
    #[must_use]
    pub fn pulls(&self, fit: &FitResult) -> Vec<(String, f64, f64)> {
        self.params
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let sigma = match p.kind {
                    Kind::Gauss { sigma } if sigma > 0.0 => sigma,
                    Kind::Poisson { tau } if tau > 0.0 => 1.0 / tau.sqrt(),
                    _ => return None,
                };
                if p.fixed {
                    return None;
                }
                Some((
                    p.name.clone(),
                    (fit.params[i] - p.init) / sigma,
                    fit.errors[i] / sigma,
                ))
            })
            .collect()
    }

    /// The impact of each constrained nuisance parameter on the parameter of
    /// interest `poi`: refit with the parameter fixed at `θ̂ ± σ̂` (post-fit)
    /// and at `θ̂ ± σ₀` (pre-fit), and record the shift of the POI — ranked by
    /// the largest post-fit shift. `fitter` is the fit's, `fit` its result.
    #[must_use]
    pub fn impacts(&self, fitter: &Fitter, fit: &FitResult, poi: &str) -> Vec<Impact> {
        let Some(ip) = self.params.iter().position(|p| p.name == poi) else {
            return Vec::new();
        };
        let quiet = FitOptions::new().hesse(false);
        let shift = |name: &str, value: f64| -> f64 {
            let r = fitter
                .clone()
                .with_params(fit.params.clone())
                .set(name, value)
                .fix(name)
                .fit(&quiet);
            if r.valid {
                r.params[ip] - fit.params[ip]
            } else {
                f64::NAN
            }
        };
        let mut out: Vec<Impact> = self
            .pulls(fit)
            .into_iter()
            .map(|(name, _, _)| {
                let i = self
                    .params
                    .iter()
                    .position(|p| p.name == name)
                    .expect("a parameter");
                let prior = match self.params[i].kind {
                    Kind::Gauss { sigma } => sigma,
                    Kind::Poisson { tau } => 1.0 / tau.sqrt(),
                    Kind::Free => 0.0,
                };
                let (v, e) = (fit.params[i], fit.errors[i]);
                Impact {
                    postfit: (shift(&name, v + e), shift(&name, v - e)),
                    prefit: (shift(&name, v + prior), shift(&name, v - prior)),
                    name,
                }
            })
            .collect();
        out.sort_by(|a, b| b.size().total_cmp(&a.size()));
        out
    }

    /// The expected yields at `fit`, per channel: each sample's and the
    /// total's counts per bin, the total's error propagated from the fit's
    /// covariance (a numerical Jacobian), and the observed counts.
    #[must_use]
    pub fn yields(&self, fit: &FitResult) -> Vec<ChannelYields> {
        let p = &fit.params;
        let free: Vec<usize> = (0..self.params.len())
            .filter(|&i| fit.errors.get(i).is_some_and(|e| e.is_finite() && *e > 0.0))
            .collect();
        let base = self.totals(p);
        // ∂ν/∂pₖ by central differences of a fraction of each error.
        let jac: Vec<Vec<Vec<f64>>> = free
            .iter()
            .map(|&k| {
                let h = 0.01 * fit.errors[k];
                let (mut up, mut down) = (p.clone(), p.clone());
                up[k] += h;
                down[k] -= h;
                let (tu, td) = (self.totals(&up), self.totals(&down));
                tu.iter()
                    .zip(&td)
                    .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y) / (2.0 * h)).collect())
                    .collect()
            })
            .collect();
        let cov = fit.covariance.as_ref();
        let expected = self.expected(p);
        self.channels
            .iter()
            .enumerate()
            .map(|(c, ch)| {
                let error = (0..ch.bins())
                    .map(|b| {
                        let Some(cov) = cov else { return f64::NAN };
                        if cov.len() != free.len() {
                            return f64::NAN;
                        }
                        let mut v = 0.0;
                        for (x, row) in cov.iter().enumerate() {
                            for (y, cxy) in row.iter().enumerate() {
                                v += jac[x][c][b] * cxy * jac[y][c][b];
                            }
                        }
                        v.max(0.0).sqrt()
                    })
                    .collect();
                ChannelYields {
                    channel: ch.name.clone(),
                    samples: ch
                        .samples
                        .iter()
                        .zip(&expected[c])
                        .map(|(s, nu)| (s.name.clone(), nu.clone()))
                        .collect(),
                    total: base[c].clone(),
                    total_error: error,
                    observed: ch.observed.clone(),
                }
            })
            .collect()
    }
}

/// One nuisance parameter's impact on the parameter of interest.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Impact {
    /// The nuisance parameter.
    pub name: String,
    /// The POI shift with the parameter at `θ̂ + σ̂` and `θ̂ − σ̂`.
    pub postfit: (f64, f64),
    /// The POI shift with the parameter at `θ̂ ± σ₀` (its prior width).
    pub prefit: (f64, f64),
}

impl Impact {
    /// The larger post-fit shift, in absolute value — the ranking.
    #[must_use]
    pub fn size(&self) -> f64 {
        self.postfit.0.abs().max(self.postfit.1.abs())
    }
}

/// A channel's yields after a fit: see [`TemplateModel::yields`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ChannelYields {
    /// The channel.
    pub channel: String,
    /// Each sample's expected counts per bin.
    pub samples: Vec<(String, Vec<f64>)>,
    /// The total expected counts per bin.
    pub total: Vec<f64>,
    /// The total's uncertainty per bin, from the fit's covariance.
    pub total_error: Vec<f64>,
    /// The observed counts.
    pub observed: Vec<f64>,
}

impl Cost for TemplateModel {
    fn param_names(&self) -> Vec<String> {
        TemplateModel::param_names(self)
    }

    fn value(&self, params: &[f64]) -> f64 {
        let main: f64 = self
            .totals(params)
            .iter()
            .zip(&self.channels)
            .map(|(nu, ch)| {
                nu.iter()
                    .zip(&ch.observed)
                    .map(|(&f, &n)| {
                        let f = f.max(1e-300);
                        2.0 * if n > 0.0 { f - n + n * (n / f).ln() } else { f }
                    })
                    .sum::<f64>()
            })
            .sum();
        main + self.constraint_cost(params, &self.aux)
    }

    fn n_measurements(&self) -> Option<usize> {
        let bins: usize = self.channels.iter().map(Channel::bins).sum();
        let aux = self
            .params
            .iter()
            .filter(|p| !matches!(p.kind, Kind::Free) && !p.fixed)
            .count();
        Some(bins + aux)
    }

    fn is_gof(&self) -> bool {
        true
    }

    fn asimov(&self, params: &[f64]) -> Option<Box<dyn Cost + '_>> {
        let aux = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| match p.kind {
                Kind::Free => f64::NAN,
                Kind::Gauss { .. } => params[i],
                Kind::Poisson { tau } => params[i] * tau,
            })
            .collect();
        Some(Box::new(self.with_data(self.totals(params), aux)))
    }

    fn toy(&self, params: &[f64], seed: u64) -> Option<Box<dyn Cost + '_>> {
        let mut rng = Rng::stream(seed, 0);
        let observed = self
            .totals(params)
            .into_iter()
            .map(|nu| {
                nu.into_iter()
                    .map(|m| rng.poisson(m.max(0.0)) as f64)
                    .collect()
            })
            .collect();
        let aux = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| match p.kind {
                Kind::Free => f64::NAN,
                Kind::Gauss { sigma } => params[i] + sigma * rng.gauss(),
                Kind::Poisson { tau } => rng.poisson(params[i] * tau) as f64,
            })
            .collect();
        Some(Box::new(self.with_data(observed, aux)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code4_matches_code1_at_and_beyond_one_sigma() {
        let poly = code4_poly(0.9, 1.15).unwrap();
        let m = TemplateModel::new(vec![]).unwrap();
        for a in [-1.0, 1.0, -2.0, 1.7] {
            let p4 = m.kappa(a, 0.9, 1.15, Some(&poly));
            let p1 = m.kappa(a, 0.9, 1.15, None);
            assert!((p4 - p1).abs() < 1e-12, "{a}: {p4} vs {p1}");
        }
        // Smooth through zero: κ(0) = 1, and the derivative is continuous at 1.
        assert!((m.kappa(0.0, 0.9, 1.15, Some(&poly)) - 1.0).abs() < 1e-15);
        let h = 1e-6;
        let inside = (m.kappa(1.0 - h, 0.9, 1.15, Some(&poly))
            - m.kappa(1.0 - 2.0 * h, 0.9, 1.15, Some(&poly)))
            / h;
        let outside = (m.kappa(1.0 + 2.0 * h, 0.9, 1.15, Some(&poly))
            - m.kappa(1.0 + h, 0.9, 1.15, Some(&poly)))
            / h;
        assert!((inside - outside).abs() < 1e-4, "{inside} vs {outside}");
    }

    #[test]
    fn code4p_is_continuous_at_one_sigma() {
        let m = TemplateModel::new(vec![]).unwrap();
        for (a, want) in [(1.0, 13.0 - 10.0), (-1.0, -(10.0 - 8.5))] {
            let d = m.delta(a, 10.0, 8.5, 13.0);
            assert!((d - want).abs() < 1e-12, "{a}: {d}");
        }
        assert_eq!(m.delta(0.0, 10.0, 8.5, 13.0), 0.0);
    }
}

/// A measurement's setting for one parameter: name, bounds, seed, fixed.
#[cfg(feature = "pyhf")]
type Setting = (String, Option<(f64, f64)>, Option<f64>, bool);

/// A pyhf JSON workspace read into a [`TemplateModel`], with its
/// measurement's parameter of interest and parameter settings (`pyhf`
/// feature).
///
/// ```
/// # #[cfg(feature = "pyhf")] {
/// use oxifit::{FitOptions, HypoTest, Workspace};
///
/// let json = r#"{
///   "channels": [{"name": "SR", "samples": [
///     {"name": "signal", "data": [5.0, 10.0],
///      "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]},
///     {"name": "background", "data": [50.0, 52.0],
///      "modifiers": [{"name": "unc", "type": "shapesys", "data": [3.0, 7.0]}]}]}],
///   "observations": [{"name": "SR", "data": [51.0, 48.0]}],
///   "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
///   "version": "1.0.0"
/// }"#;
/// let ws = Workspace::from_json(json, None)?;
/// assert_eq!(ws.poi(), "mu");
/// let test = HypoTest::new(ws.fitter(), ws.poi())?;
/// let limit = test.upper_limit(0.95, (0.0, 5.0))?;
/// assert!(limit.observed > 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "pyhf")]
#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    model: TemplateModel,
    poi: String,
    /// Per parameter name: bounds, seed, fixed.
    settings: Vec<Setting>,
}

#[cfg(feature = "pyhf")]
impl Workspace {
    /// Read a pyhf workspace (`channels`, `observations`, `measurements`),
    /// using the named measurement or, with `None`, the first.
    ///
    /// # Errors
    /// [`FitError::Workspace`] for JSON that is not a workspace this
    /// reader understands (and the model's own errors).
    pub fn from_json(json: &str, measurement: Option<&str>) -> Result<Workspace, FitError> {
        use serde_json::Value;
        let bad = |why: &str| FitError::Workspace(why.to_string());
        let v: Value = serde_json::from_str(json).map_err(|e| bad(&e.to_string()))?;
        let nums = |x: &Value, what: &str| -> Result<Vec<f64>, FitError> {
            x.as_array()
                .ok_or_else(|| bad(&format!("{what}: expected an array")))?
                .iter()
                .map(|n| {
                    n.as_f64()
                        .ok_or_else(|| bad(&format!("{what}: expected numbers")))
                })
                .collect()
        };
        let text = |x: &Value, key: &str| -> Result<String, FitError> {
            x.get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| bad(&format!("missing \"{key}\"")))
        };
        // The measurement: its POI and parameter settings.
        let measurements = v
            .get("measurements")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("no measurements"))?;
        let meas = match measurement {
            Some(name) => measurements
                .iter()
                .find(|m| m.get("name").and_then(Value::as_str) == Some(name))
                .ok_or_else(|| bad(&format!("no measurement {name:?}")))?,
            None => measurements.first().ok_or_else(|| bad("no measurements"))?,
        };
        let config = meas
            .get("config")
            .ok_or_else(|| bad("measurement without config"))?;
        let poi = text(config, "poi")?;
        let mut lumi_sigma = 0.0;
        let mut settings = Vec::new();
        for p in config
            .get("parameters")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
        {
            let name = text(p, "name")?;
            if let Some(s) = p.get("sigmas")
                && let Some(&first) = nums(s, "sigmas")?.first()
            {
                lumi_sigma = first;
            }
            let bounds = match p.get("bounds") {
                Some(b) => b
                    .as_array()
                    .and_then(|a| a.first())
                    .map(|pair| nums(pair, "bounds"))
                    .transpose()?
                    .and_then(|pair| (pair.len() == 2).then(|| (pair[0], pair[1]))),
                None => None,
            };
            let init = match p.get("inits") {
                Some(i) => nums(i, "inits")?.first().copied(),
                None => None,
            };
            let fixed = p.get("fixed").and_then(Value::as_bool).unwrap_or(false);
            settings.push((name, bounds, init, fixed));
        }
        // Observations by channel name.
        let observations = v
            .get("observations")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("no observations"))?;
        let mut channels = Vec::new();
        for ch in v
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("no channels"))?
        {
            let name = text(ch, "name")?;
            let observed = observations
                .iter()
                .find(|o| o.get("name").and_then(Value::as_str) == Some(name.as_str()))
                .ok_or_else(|| bad(&format!("no observation for channel {name:?}")))?;
            let mut channel = Channel::new(&name, &nums(&observed["data"], "observation")?);
            for s in ch
                .get("samples")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("channel without samples"))?
            {
                let mut sample = Sample::new(&text(s, "name")?, &nums(&s["data"], "sample data")?);
                for m in s
                    .get("modifiers")
                    .and_then(Value::as_array)
                    .map_or(&[][..], Vec::as_slice)
                {
                    let (mname, kind) = (text(m, "name")?, text(m, "type")?);
                    let data = &m["data"];
                    sample = match kind.as_str() {
                        "normfactor" => sample.norm_factor(&mname),
                        "normsys" => {
                            let f = |k: &str| {
                                data.get(k)
                                    .and_then(Value::as_f64)
                                    .ok_or_else(|| bad("normsys without hi/lo"))
                            };
                            sample.norm_sys(&mname, f("lo")?, f("hi")?)
                        }
                        "histosys" => sample.histo_sys(
                            &mname,
                            &nums(&data["lo_data"], "histosys lo_data")?,
                            &nums(&data["hi_data"], "histosys hi_data")?,
                        ),
                        "staterror" => sample.stat_error(&mname, &nums(data, "staterror")?),
                        "shapesys" => sample.shape_sys(&mname, &nums(data, "shapesys")?),
                        "lumi" => sample.lumi(&mname),
                        "shapefactor" => sample.shape_factor(&mname),
                        other => return Err(bad(&format!("modifier type {other:?}"))),
                    };
                }
                channel = channel.sample(sample);
            }
            channels.push(channel);
        }
        let model = TemplateModel::new(channels)?.lumi_uncertainty(lumi_sigma)?;
        if !model.param_names().contains(&poi) {
            return Err(bad(&format!("the POI {poi:?} is not a parameter")));
        }
        Ok(Workspace {
            model,
            poi,
            settings,
        })
    }

    /// The model.
    #[must_use]
    pub fn model(&self) -> &TemplateModel {
        &self.model
    }

    /// The measurement's parameter of interest.
    #[must_use]
    pub fn poi(&self) -> &str {
        &self.poi
    }

    /// A [`Fitter`] with the model's standard settings and the measurement's
    /// bounds, seeds and fixed parameters (a setting for a multi-bin
    /// parameter applies to each bin).
    #[must_use]
    pub fn fitter(&self) -> Fitter<'_> {
        let mut f = self.model.fitter();
        let names = self.model.param_names();
        for (name, bounds, init, fixed) in &self.settings {
            let targets: Vec<&String> = names
                .iter()
                .filter(|n| {
                    *n == name
                        || n.strip_prefix(name.as_str())
                            .is_some_and(|r| r.starts_with('['))
                })
                .collect();
            for t in targets {
                if let Some((lo, hi)) = bounds {
                    f = f.limit(t, *lo, *hi);
                }
                if let Some(v) = init {
                    f = f.set(t, *v);
                }
                if *fixed {
                    f = f.fix(t);
                }
            }
        }
        f
    }
}
