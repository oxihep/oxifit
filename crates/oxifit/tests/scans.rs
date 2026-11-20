//! Profile likelihoods and 2-D contours.

mod common;

use oxifit::{FitError, FitExt, FitMethod, FitOptions, Model, Points};

fn line() -> (Points, Model) {
    let data = Points::new(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[1.1, 2.9, 5.2, 6.8, 9.1, 11.0],
        &[0.3, 0.2, 0.4, 0.3, 0.5, 0.2],
    )
    .unwrap();
    (
        data,
        Model::polynomial("line", 1).with_params(vec![0.0, 1.0]),
    )
}

#[test]
fn a_linear_fit_contour_is_the_covariance_ellipse() {
    // A linear χ² is exactly quadratic: its Δχ² = δ contour is the ellipse
    // (p − p̂)ᵀ C⁻¹ (p − p̂) = δ, C the covariance.
    let (data, model) = line();
    let opts = FitOptions::new();
    let fit = data.fit_opts(&model, &opts);
    let c = fit.covariance.clone().unwrap();
    let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
    let inv = [
        [c[1][1] / det, -c[0][1] / det],
        [-c[1][0] / det, c[0][0] / det],
    ];
    for delta in [1.0, 2.30] {
        let contour = data.contour(&model, &opts, "p0", "p1", delta, 24).unwrap();
        assert_eq!(contour.points.len(), 24);
        assert_eq!(
            (contour.x_param.as_str(), contour.y_param.as_str()),
            ("p0", "p1")
        );
        let (bx, by) = contour.best;
        for &(x, y) in &contour.points {
            let (dx, dy) = (x - bx, y - by);
            let q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
            assert!(
                (q / delta - 1.0).abs() < 1e-2,
                "δ={delta}: ({x}, {y}) has q = {q}"
            );
        }
        // Its extent along x is √δ·σx (the MINOS points are on the contour).
        let xmax = contour.points.iter().map(|p| p.0).fold(f64::MIN, f64::max);
        assert!(((xmax - bx) / (delta.sqrt() * fit.errors[0]) - 1.0).abs() < 1e-2);
    }
}

#[test]
fn a_profile_of_a_linear_fit_is_a_parabola() {
    let (data, model) = line();
    let fit = data.fit(&model);
    let (b, e) = (fit.params[1], fit.errors[1]);
    let profile = data
        .profile(
            &model,
            &FitOptions::new(),
            "p1",
            (b - 3.0 * e, b + 3.0 * e),
            61,
        )
        .unwrap();
    assert_eq!(profile.values.len(), 61);
    assert!((profile.best - b).abs() < 1e-12 && profile.fval_min == fit.fval);
    for (v, d) in profile.values.iter().zip(&profile.delta) {
        let want = ((v - b) / e).powi(2);
        assert!(
            (d - want).abs() < 1e-3 * (1.0 + want),
            "Δ({v}) = {d}, want {want}"
        );
    }
    let (lo, hi) = profile.interval(1.0).unwrap();
    assert!(((b - lo) / e - 1.0).abs() < 1e-2 && ((hi - b) / e - 1.0).abs() < 1e-2);
    let (lo2, hi2) = profile.interval(4.0).unwrap();
    assert!(
        ((hi2 - lo2) / (4.0 * e) - 1.0).abs() < 1e-2,
        "2σ is twice as wide"
    );
    // Out of range: the 3σ (Δ = 9) crossing is the scan's edge, 4σ is outside.
    assert!(profile.interval(16.0).is_none());
}

#[test]
fn a_profile_interval_matches_minos_on_a_peak() {
    // A likelihood fit of a peak is not parabolic in the width: the profile
    // interval and MINOS both follow the true cost, and agree.
    let data = common::peak_data();
    let model = common::peak_model(3.0).with_params(vec![200.0, 3.0, 0.6, 19.0]);
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let fit = data.fit_opts(&model, &opts.minos(true));
    let (mlo, mhi) = fit.minos.as_ref().unwrap()[2];
    let s = fit.params[2];
    let profile = data
        .profile(&model, &opts, "s", (s + 2.0 * mlo, s + 2.0 * mhi), 41)
        .unwrap();
    let (lo, hi) = profile.interval(1.0).unwrap();
    assert!(
        ((lo - s) / mlo - 1.0).abs() < 0.05,
        "lower: {} vs MINOS {mlo}",
        lo - s
    );
    assert!(
        ((hi - s) / mhi - 1.0).abs() < 0.05,
        "upper: {} vs MINOS {mhi}",
        hi - s
    );
}

#[test]
fn a_peak_contour_lies_at_delta_with_the_other_parameters_refitted() {
    // Mean and width of a peak, the amplitude and background profiled: each
    // contour point, refitted independently with (mean, width) fixed there,
    // sits Δ above the minimum.
    let data = common::peak_data();
    let model = common::peak_model(3.0).with_params(vec![200.0, 3.0, 0.6, 19.0]);
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let best = data.fit_opts(&model, &opts);
    let contour = data.contour(&model, &opts, "mu", "s", 2.30, 12).unwrap();
    for &(mu, s) in &contour.points {
        let mut p = best.params.clone();
        p[1] = mu;
        p[2] = s;
        let fixed = model.clone().with_params(p).fix("mu").fix("s");
        let refit = data.fit_opts(&fixed, &opts.hesse(false));
        let d = refit.fval - best.fval;
        assert!((d - 2.30).abs() < 0.01, "({mu}, {s}): Δ = {d}");
    }
    // It encloses the minimum.
    let (bx, by) = contour.best;
    let xs = contour.points.iter().map(|p| p.0);
    let ys = contour.points.iter().map(|p| p.1);
    assert!(xs.clone().fold(f64::MAX, f64::min) < bx && xs.fold(f64::MIN, f64::max) > bx);
    assert!(ys.clone().fold(f64::MAX, f64::min) < by && ys.fold(f64::MIN, f64::max) > by);
}

#[test]
fn scans_report_what_went_wrong() {
    let (data, model) = line();
    let opts = FitOptions::new();
    assert_eq!(
        data.profile(&model, &opts, "slope", (0.0, 1.0), 5),
        Err(FitError::UnknownParameter("slope".into()))
    );
    let fixed = model.clone().fix("p0");
    assert_eq!(
        data.contour(&fixed, &opts, "p0", "p1", 1.0, 8),
        Err(FitError::FixedParameter("p0".into()))
    );
    // One point cannot determine a line: no valid minimum to scan around.
    let one = Points::new(&[1.0], &[2.0], &[0.1]).unwrap();
    assert_eq!(
        one.profile(&model, &opts, "p1", (0.0, 1.0), 5),
        Err(FitError::FitFailed)
    );
    assert_eq!(
        FitError::FitFailed.to_string(),
        "the fit did not converge to a valid minimum"
    );
}
