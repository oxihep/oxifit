//! The costs, error estimates and failure handling of the fit engine, checked
//! against closed-form answers.

mod common;

use oxifit::{FitExt, FitMethod, FitOptions, Model, Point, Points};

/// Counts with Poisson-like errors `σ = √y`.
fn counts(ys: &[f64]) -> Points {
    let pts = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| Point::new(i as f64, y, y.sqrt()))
        .collect();
    Points::from_points(pts)
}

const YS: [f64; 8] = [12.0, 7.0, 9.0, 15.0, 11.0, 8.0, 10.0, 13.0];

#[test]
fn constant_fit_matches_the_closed_form_of_each_cost() {
    let data = counts(&YS);
    let model = Model::polynomial("c", 0).with_params(vec![5.0]);
    let n = YS.len() as f64;
    let fit = |m| data.fit_opts(&model, &FitOptions::new().method(m));
    // MIGRAD stops once its estimated distance to the minimum (EDM) is below
    // ~1e-4, i.e. it locates the minimum to ~1 % of the parameter error.
    let close = |got: f64, want: f64, err: f64| (got - want).abs() < 0.02 * err;

    // Poisson likelihood: the MLE of a constant is the mean count.
    let mean = YS.iter().sum::<f64>() / n;
    let l = fit(FitMethod::Likelihood);
    assert!(l.valid && close(l.params[0], mean, l.errors[0]), "{l:?}");
    // Its error is √(mean / n) (Fisher information n / c).
    assert!(
        (l.errors[0] / (mean / n).sqrt() - 1.0).abs() < 1e-3,
        "{l:?}"
    );

    // Neyman χ² with σ² = y: the minimum is the harmonic mean.
    let harmonic = n / YS.iter().map(|y| 1.0 / y).sum::<f64>();
    let neyman = fit(FitMethod::Chi2);
    assert!(
        close(neyman.params[0], harmonic, neyman.errors[0]),
        "{neyman:?}"
    );

    // Pearson χ² with σ² = c: d/dc Σ (y − c)²/c = 0 gives c = √(mean of y²).
    let rms = (YS.iter().map(|y| y * y).sum::<f64>() / n).sqrt();
    let pearson = fit(FitMethod::PearsonChi2);
    assert!(
        close(pearson.params[0], rms, pearson.errors[0]),
        "{pearson:?}"
    );
}

#[test]
fn hesse_errors_of_a_linear_fit_are_exact() {
    // For y = a + b·x with errors σᵢ the covariance is (AᵀWA)⁻¹ in closed form.
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.1, 2.9, 5.2, 6.8, 9.1, 11.0];
    let s = [0.3, 0.2, 0.4, 0.3, 0.5, 0.2];
    let data = Points::new(&x, &y, &s).unwrap();
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for (&xi, &si) in x.iter().zip(&s) {
        let w = 1.0 / (si * si);
        s0 += w;
        s1 += w * xi;
        s2 += w * xi * xi;
    }
    let det = s0 * s2 - s1 * s1;
    let (var_a, var_b, cov_ab) = (s2 / det, s0 / det, -s1 / det);

    let model = Model::polynomial("line", 1).with_params(vec![0.0, 1.0]);
    for hesse in [true, false] {
        let fit = data.fit_opts(&model, &FitOptions::new().hesse(hesse));
        assert!(fit.valid, "{fit:?}");
        let cov = fit.covariance.as_ref().unwrap();
        let tol = if hesse { 1e-6 } else { 1e-2 }; // MIGRAD's estimate is approximate
        assert!(
            (cov[0][0] / var_a - 1.0).abs() < tol,
            "hesse={hesse} {cov:?}"
        );
        assert!(
            (cov[1][1] / var_b - 1.0).abs() < tol,
            "hesse={hesse} {cov:?}"
        );
        assert!(
            (cov[0][1] / cov_ab - 1.0).abs() < tol,
            "hesse={hesse} {cov:?}"
        );
    }
}

#[test]
fn minos_brackets_the_minimum_and_matches_a_parabolic_cost() {
    // A linear χ² is exactly parabolic, so MINOS equals the parabolic error.
    let data = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.1, 4.9, 7.0], &[0.2; 4]).unwrap();
    let model = Model::polynomial("line", 1)
        .with_params(vec![0.0, 1.0])
        .fix("p0");
    let fit = data.fit_opts(&model, &FitOptions::new().minos(true));
    let minos = fit.minos.as_ref().expect("MINOS was requested");
    assert_eq!(minos[0], (0.0, 0.0), "a fixed parameter has no MINOS error");
    let (lo, hi) = minos[1];
    assert!(lo < 0.0 && hi > 0.0, "{minos:?}");
    assert!((-lo / fit.errors[1] - 1.0).abs() < 1e-3, "{fit:?}");
    assert!((hi / fit.errors[1] - 1.0).abs() < 1e-3, "{fit:?}");
}

#[test]
fn a_limit_holds_the_parameter_inside_it() {
    // The data want a slope of 2; capping it at 1.5 must hold it there.
    let data = Points::unweighted(&[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 4.0, 6.0]).unwrap();
    let model = Model::polynomial("line", 1)
        .with_params(vec![0.0, 1.0])
        .upper_limit("p1", 1.5);
    let fit = data.fit(&model);
    assert!(fit.params[1] <= 1.5 + 1e-9, "{fit:?}");
    assert!((fit.params[1] - 1.5).abs() < 1e-3, "{fit:?}");
}

#[test]
fn a_panicking_backend_is_reported_as_an_invalid_fit() {
    // Seeded with the mean at −8.64, this exact sample once made the minuit2 0.5
    // numerical gradient panic. The fit must return, not unwind; whatever the
    // backend does, a result that claims validity must be a real minimum.
    let fit = common::peak_data().fit_opts(
        &common::peak_model(-8.64),
        &FitOptions::new().method(FitMethod::Likelihood),
    );
    if fit.valid {
        assert!(fit.chi2.is_finite(), "{fit:?}");
    } else {
        assert!(fit.errors.iter().all(|e| e.is_nan()), "{fit:?}");
    }
}
