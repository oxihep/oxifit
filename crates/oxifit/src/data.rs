//! [`FitData`] — the data abstraction every fit consumes — and [`FitExt`], the
//! blanket trait that gives every dataset the `.fit(...)` methods.

use oxistats::StatError;

use crate::engine::run_fit;
use crate::model::Model;
use crate::result::{FitOptions, FitResult};
use crate::scan::{self, Contour, FitError, Profile};

/// One data point a model is fit against: an independent value `x`, a measured
/// value `y`, and a Gaussian uncertainty `sigma` on `y`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Independent variable.
    pub x: f64,
    /// Measured value at `x`.
    pub y: f64,
    /// Gaussian uncertainty on `y`. A value `<= 0` means "no usable error":
    /// χ² (Neyman) fits skip such points, the other costs keep them.
    pub sigma: f64,
}

impl Point {
    /// A point `(x, y)` with uncertainty `sigma`.
    #[must_use]
    pub fn new(x: f64, y: f64, sigma: f64) -> Point {
        Point { x, y, sigma }
    }
}

/// A 1-D dataset a parametric [`Model`] can be fit to. Implement it for any data
/// source — a histogram's bins, a graph's points, raw measurements — and the
/// blanket [`FitExt`] gives it `.fit(...)` for free.
///
/// `points` returns every candidate point; the fit engine applies the range
/// filter and (for Neyman χ²) the empty-point cut itself.
pub trait FitData {
    /// The data as `(x, y, sigma)` points.
    fn points(&self) -> Vec<Point>;

    /// Each point's bin `(lo, hi)`, for data that are bins of a histogram —
    /// used by bin-integrated fits
    /// ([`FitOptions::with_bin_integration`](crate::FitOptions::with_bin_integration)).
    /// `None` (the default): the bins are taken halfway between neighbouring
    /// points.
    fn bin_edges(&self) -> Option<Vec<(f64, f64)>> {
        None
    }
}

/// A standalone dataset of raw `(x, y, sigma)` points — for fitting data that is
/// neither a histogram nor a graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Points {
    points: Vec<Point>,
    edges: Option<Vec<(f64, f64)>>,
}

impl Points {
    /// Build from parallel `x`, `y`, and `sigma` slices.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `y` or `sigma` is not as long as `x`
    /// (`left` is the length of `x`).
    pub fn new(x: &[f64], y: &[f64], sigma: &[f64]) -> Result<Points, StatError> {
        paired(x.len(), y.len())?;
        paired(x.len(), sigma.len())?;
        let points = x
            .iter()
            .zip(y)
            .zip(sigma)
            .map(|((&x, &y), &s)| Point::new(x, y, s))
            .collect();
        Ok(Points {
            points,
            edges: None,
        })
    }

    /// Build from `x`/`y` with unit (unweighted) errors — an ordinary
    /// least-squares fit.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `y` is not as long as `x`.
    pub fn unweighted(x: &[f64], y: &[f64]) -> Result<Points, StatError> {
        paired(x.len(), y.len())?;
        let points = x
            .iter()
            .zip(y)
            .map(|(&x, &y)| Point::new(x, y, 1.0))
            .collect();
        Ok(Points {
            points,
            edges: None,
        })
    }

    /// Build directly from [`Point`]s.
    #[must_use]
    pub fn from_points(points: Vec<Point>) -> Points {
        Points {
            points,
            edges: None,
        }
    }

    /// Histogram bins: `edges` (one more than the points) are the bins'
    /// boundaries, for bin-integrated fits
    /// ([`FitOptions::with_bin_integration`](crate::FitOptions::with_bin_integration))
    /// of variable-width bins.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] unless there is one more edge than points.
    pub fn with_bin_edges(mut self, edges: &[f64]) -> Result<Points, StatError> {
        paired(self.points.len() + 1, edges.len())?;
        self.edges = Some(edges.windows(2).map(|w| (w[0], w[1])).collect());
        Ok(self)
    }

    /// The points.
    #[must_use]
    pub fn as_points(&self) -> &[Point] {
        &self.points
    }
}

impl FitData for Points {
    fn points(&self) -> Vec<Point> {
        self.points.clone()
    }
    fn bin_edges(&self) -> Option<Vec<(f64, f64)>> {
        self.edges.clone()
    }
}

/// Unbinned data: individual events (a value `x` each, with an optional
/// weight) for the unbinned likelihoods
/// ([`FitMethod::UnbinnedLikelihood`](crate::FitMethod::UnbinnedLikelihood),
/// [`FitMethod::ExtendedUnbinnedLikelihood`](crate::FitMethod::ExtendedUnbinnedLikelihood)).
/// As [`FitData`], each event is a point `(x, weight, 0)`.
///
/// ```
/// # use oxifit::{Events, FitExt, FitMethod, FitOptions, Model};
/// let events = Events::new(&[0.3, 1.1, 0.7, 2.4, 0.9, 1.6]);
/// let decay = Model::exponential("decay").with_params(vec![0.0, -1.0]).fix("constant");
/// let opts = FitOptions::new().method(FitMethod::UnbinnedLikelihood).range(0.0, 5.0);
/// let fit = events.fit_opts(&decay, &opts);
/// assert!(fit.valid && fit.params[1] < 0.0);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Events {
    x: Vec<f64>,
    w: Option<Vec<f64>>,
}

impl Events {
    /// Unweighted events at `x`.
    #[must_use]
    pub fn new(x: &[f64]) -> Events {
        Events {
            x: x.to_vec(),
            w: None,
        }
    }

    /// Weighted events: `w[i]` is event `i`'s weight (e.g. an efficiency
    /// correction or an sWeight). Fit with
    /// [`FitOptions::with_sumw2`](crate::FitOptions::with_sumw2) for errors
    /// that reflect the weights.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `w` is not as long as `x`.
    pub fn weighted(x: &[f64], w: &[f64]) -> Result<Events, StatError> {
        paired(x.len(), w.len())?;
        Ok(Events {
            x: x.to_vec(),
            w: Some(w.to_vec()),
        })
    }

    /// The event values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.x
    }

    /// The event weights, if the events are weighted.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.w.as_deref()
    }

    /// The number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

impl FitData for Events {
    fn points(&self) -> Vec<Point> {
        match &self.w {
            Some(w) => self
                .x
                .iter()
                .zip(w)
                .map(|(&x, &w)| Point::new(x, w, 0.0))
                .collect(),
            None => self.x.iter().map(|&x| Point::new(x, 1.0, 0.0)).collect(),
        }
    }
}

/// Fit a bare collection of points. Because the [`FitExt`] blanket is `?Sized`,
/// implementing `FitData` on the unsized `[Point]` also gives `.fit(...)` to
/// `&[Point]`, `[Point; N]`, and `Vec<Point>` (via deref/unsizing).
impl FitData for [Point] {
    fn points(&self) -> Vec<Point> {
        self.to_vec()
    }
}

/// Fit methods for any [`FitData`]. Implemented for all of them by a blanket
/// impl, so `data.fit(&model)` works for a `Hist1D`, a `Graph`, or [`Points`]
/// alike. Bring it into scope (`use oxifit::FitExt;`, or the `oxiroot`
/// prelude) to call these.
pub trait FitExt: FitDataNd {
    /// Fit `model` by chi-square minimization over the full range (ROOT's
    /// default `Fit`) — shorthand for [`fit_opts`](Self::fit_opts) with the
    /// default [`FitOptions`]. To pick a different cost, a range, or a robust
    /// loss, build a [`FitOptions`] and use [`fit_opts`](Self::fit_opts).
    #[must_use]
    fn fit(&self, model: &Model) -> FitResult {
        self.fit_opts(model, &FitOptions::new())
    }

    /// Fit `model` with full control over the cost and range ([`FitOptions`]),
    /// seeded from `model.params` and its per-parameter constraints. The points
    /// entering the fit are those whose `x` is in `opts.range` (default: all);
    /// Neyman chi-square additionally drops points with no error (`sigma <= 0`).
    /// `FitResult::chi2` is the cost at the minimum.
    #[must_use]
    fn fit_opts(&self, model: &Model, opts: &FitOptions) -> FitResult {
        run_fit(&engine_points(self), self.dim(), model, opts)
    }

    /// Fit `model` (per `opts`) and write the best-fit parameters back into it,
    /// so `model.eval(x)` evaluates the *fitted* curve afterwards. Use
    /// [`fit_opts`](Self::fit_opts) to keep the model's initial parameters.
    fn fit_into(&self, model: &mut Model, opts: &FitOptions) -> FitResult {
        let result = self.fit_opts(model, opts);
        model.params.clone_from(&result.params);
        result
    }

    /// The profile of the cost in parameter `param`: fit once (per `opts`),
    /// then step `param` over `n` values evenly spaced across `range`, refitting
    /// the other parameters (from the best fit) at each, and report
    /// `Δcost` against the minimum. [`Profile::interval`] reads off the 1σ
    /// (`Δ = 1`) or 2σ (`Δ = 4`) interval, valid where the cost is not
    /// parabolic. Costs `n` fits.
    ///
    /// ```
    /// # use oxifit::{FitExt, Model, Points};
    /// let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.1, 4.9, 7.0], &[0.2; 4])?;
    /// let line = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
    /// let fit = data.fit(&line);
    /// let profile = data
    ///     .profile(&line, &Default::default(), "p1", (fit.params[1] - 0.3, fit.params[1] + 0.3), 31)
    ///     .unwrap();
    /// let (lo, hi) = profile.interval(1.0).unwrap();
    /// // A linear fit's χ² is parabolic: the interval is the parabolic error.
    /// assert!(((hi - lo) / 2.0 / fit.errors[1] - 1.0).abs() < 0.01);
    /// # Ok::<(), oxifit::StatError>(())
    /// ```
    ///
    /// # Errors
    /// [`FitError::UnknownParameter`] / [`FitError::FixedParameter`] if `param`
    /// is not a free parameter of `model`; [`FitError::FitFailed`] if the fit
    /// at the minimum is not valid.
    fn profile(
        &self,
        model: &Model,
        opts: &FitOptions,
        param: &str,
        range: (f64, f64),
        n: usize,
    ) -> Result<Profile, FitError> {
        scan::check_dim(model, self.dim())?;
        let data = engine_points(self);
        let refit = scan::DataRefit {
            data: &data,
            dim: self.dim(),
        };
        scan::profile(&refit, model, opts, param, range, n)
    }

    /// The 2-D confidence contour of parameters `x` and `y` at `Δcost = delta`
    /// (`2.30` for a 68.3 % joint region, `5.99` for 95 %; `1.0` for the
    /// contour whose projections are the 1σ errors), traced by Minuit2's
    /// `MnContours` with `npoints` points (at least 4) around the minimum
    /// found per `opts`.
    ///
    /// # Errors
    /// [`FitError::UnknownParameter`] / [`FitError::FixedParameter`] if `x` or
    /// `y` is not a free parameter of `model`; [`FitError::FitFailed`] if the
    /// fit at the minimum is not valid or the contour could not be traced.
    fn contour(
        &self,
        model: &Model,
        opts: &FitOptions,
        x: &str,
        y: &str,
        delta: f64,
        npoints: usize,
    ) -> Result<Contour, FitError> {
        scan::check_dim(model, self.dim())?;
        let data = engine_points(self);
        let refit = scan::DataRefit {
            data: &data,
            dim: self.dim(),
        };
        scan::contour(&refit, model, opts, (x, y), delta, npoints)
    }
}

impl<T: FitDataNd + ?Sized> FitExt for T {}

/// The points as the engine takes them: with their bins attached, when the
/// data have bin edges.
pub(crate) fn engine_points(data: &(impl FitDataNd + ?Sized)) -> Vec<PointNd> {
    crate::engine::attach_edges(data.points_nd(), data.bin_edges_nd())
}

/// A data point in one to three dimensions: the variables `at` (unused
/// trailing entries are 0), a value, and its uncertainty — as
/// [`Point`] for 1-D data. For unbinned events the value is the weight and
/// `sigma` is unused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointNd {
    /// The variables: `[x, y, z]`, the first [`FitDataNd::dim`] of them used.
    pub at: [f64; 3],
    /// The measured value at `at` (an event's weight for unbinned data).
    pub value: f64,
    /// The uncertainty on `value`; `<= 0` means "no usable error".
    pub sigma: f64,
}

impl PointNd {
    /// A point of `dim` variables: the first `dim` entries of `at` are used.
    #[must_use]
    pub fn new(at: [f64; 3], value: f64, sigma: f64) -> PointNd {
        PointNd { at, value, sigma }
    }
}

/// A dataset of one to three variables a model of the same dimension can be
/// fitted to — 2-D and 3-D histograms ([`Points2`], [`Points3`]) and events
/// ([`Events2`], [`Events3`]), and every 1-D [`FitData`] (by a blanket
/// implementation). The [`FitExt`] methods (`fit`, `fit_opts`, `profile`, …)
/// work on all of them; the model's [`dim`](Model::dim) must match the data's.
pub trait FitDataNd {
    /// The number of variables, 1–3.
    fn dim(&self) -> usize;
    /// The data as points.
    fn points_nd(&self) -> Vec<PointNd>;
    /// For 1-D histogram data, each point's bin (see [`FitData::bin_edges`]).
    fn bin_edges_nd(&self) -> Option<Vec<(f64, f64)>> {
        None
    }
}

impl<T: FitData + ?Sized> FitDataNd for T {
    fn dim(&self) -> usize {
        1
    }
    fn points_nd(&self) -> Vec<PointNd> {
        self.points()
            .into_iter()
            .map(|p| PointNd::new([p.x, 0.0, 0.0], p.y, p.sigma))
            .collect()
    }
    fn bin_edges_nd(&self) -> Option<Vec<(f64, f64)>> {
        self.bin_edges()
    }
}

/// 2-D data: values with uncertainties at points `(x, y)` — the bins of a 2-D
/// histogram (centers, contents, errors), or measurements on a grid.
///
/// ```
/// # use oxifit::{FitExt, Model, Points2};
/// let x = [0.0, 1.0, 0.0, 1.0];
/// let y = [0.0, 0.0, 1.0, 1.0];
/// let v = [1.0, 3.0, 4.0, 6.0]; // 1 + 2x + 3y
/// let data = Points2::new(&x, &y, &v, &[0.1; 4])?;
/// let plane = Model::new2("plane", &["a", "b", "c"], vec![0.0; 3], |x, y, p| p[0] + p[1] * x + p[2] * y);
/// let fit = data.fit(&plane);
/// assert!(fit.valid && (fit.params[2] - 3.0).abs() < 1e-6);
/// # Ok::<(), oxifit::StatError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Points2 {
    points: Vec<PointNd>,
}

impl Points2 {
    /// Build from parallel `x`, `y`, `value` and `sigma` slices.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if a slice is not as long as `x`.
    pub fn new(x: &[f64], y: &[f64], value: &[f64], sigma: &[f64]) -> Result<Points2, StatError> {
        for n in [y.len(), value.len(), sigma.len()] {
            paired(x.len(), n)?;
        }
        let points = (0..x.len())
            .map(|i| PointNd::new([x[i], y[i], 0.0], value[i], sigma[i]))
            .collect();
        Ok(Points2 { points })
    }

    /// Build directly from points (their first two variables are used).
    #[must_use]
    pub fn from_points(points: Vec<PointNd>) -> Points2 {
        Points2 { points }
    }
}

impl FitDataNd for Points2 {
    fn dim(&self) -> usize {
        2
    }
    fn points_nd(&self) -> Vec<PointNd> {
        self.points.clone()
    }
}

/// 3-D data: values with uncertainties at points `(x, y, z)` — the bins of a
/// 3-D histogram, or measurements on a grid.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Points3 {
    points: Vec<PointNd>,
}

impl Points3 {
    /// Build from parallel `x`, `y`, `z`, `value` and `sigma` slices.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if a slice is not as long as `x`.
    pub fn new(
        x: &[f64],
        y: &[f64],
        z: &[f64],
        value: &[f64],
        sigma: &[f64],
    ) -> Result<Points3, StatError> {
        for n in [y.len(), z.len(), value.len(), sigma.len()] {
            paired(x.len(), n)?;
        }
        let points = (0..x.len())
            .map(|i| PointNd::new([x[i], y[i], z[i]], value[i], sigma[i]))
            .collect();
        Ok(Points3 { points })
    }

    /// Build directly from points.
    #[must_use]
    pub fn from_points(points: Vec<PointNd>) -> Points3 {
        Points3 { points }
    }
}

impl FitDataNd for Points3 {
    fn dim(&self) -> usize {
        3
    }
    fn points_nd(&self) -> Vec<PointNd> {
        self.points.clone()
    }
}

/// Unbinned 2-D data: events at `(x, y)`, optionally weighted — for the
/// unbinned likelihoods with a 2-D model (normalized over the fit region,
/// [`FitOptions::range`](crate::FitOptions::range) and
/// [`range_y`](crate::FitOptions::range_y)).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Events2 {
    points: Vec<PointNd>,
}

impl Events2 {
    /// Unweighted events at `(x[i], y[i])`.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `y` is not as long as `x`.
    pub fn new(x: &[f64], y: &[f64]) -> Result<Events2, StatError> {
        Events2::weighted(x, y, &vec![1.0; x.len()])
    }

    /// Weighted events.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `y` or `w` is not as long as `x`.
    pub fn weighted(x: &[f64], y: &[f64], w: &[f64]) -> Result<Events2, StatError> {
        paired(x.len(), y.len())?;
        paired(x.len(), w.len())?;
        let points = (0..x.len())
            .map(|i| PointNd::new([x[i], y[i], 0.0], w[i], 0.0))
            .collect();
        Ok(Events2 { points })
    }

    /// The number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl FitDataNd for Events2 {
    fn dim(&self) -> usize {
        2
    }
    fn points_nd(&self) -> Vec<PointNd> {
        self.points.clone()
    }
}

/// Unbinned 3-D data: events at `(x, y, z)`, optionally weighted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Events3 {
    points: Vec<PointNd>,
}

impl Events3 {
    /// Unweighted events at `(x[i], y[i], z[i])`.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if `y` or `z` is not as long as `x`.
    pub fn new(x: &[f64], y: &[f64], z: &[f64]) -> Result<Events3, StatError> {
        Events3::weighted(x, y, z, &vec![1.0; x.len()])
    }

    /// Weighted events.
    ///
    /// # Errors
    /// [`StatError::LengthMismatch`] if a slice is not as long as `x`.
    pub fn weighted(x: &[f64], y: &[f64], z: &[f64], w: &[f64]) -> Result<Events3, StatError> {
        for n in [y.len(), z.len(), w.len()] {
            paired(x.len(), n)?;
        }
        let points = (0..x.len())
            .map(|i| PointNd::new([x[i], y[i], z[i]], w[i], 0.0))
            .collect();
        Ok(Events3 { points })
    }

    /// The number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl FitDataNd for Events3 {
    fn dim(&self) -> usize {
        3
    }
    fn points_nd(&self) -> Vec<PointNd> {
        self.points.clone()
    }
}

/// Reject paired inputs whose lengths differ.
fn paired(left: usize, right: usize) -> Result<(), StatError> {
    if left == right {
        Ok(())
    } else {
        Err(StatError::LengthMismatch { left, right })
    }
}
