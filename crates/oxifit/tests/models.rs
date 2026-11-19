//! Composite (sum) models and Gaussian parameter constraints.

mod common;

use oxifit::{FitExt, FitMethod, FitOptions, Minimizer, Model, Points};

#[test]
fn a_sum_names_its_parameters_after_its_parts() {
    let sig = Model::gaussian("sig").with_params(vec![5.0, 1.0, 0.5]);
    let bkg = Model::polynomial("bkg", 1).with_params(vec![2.0, 0.1]);
    let m = Model::sum("m", &[&sig, &bkg]);
    assert_eq!(
        m.param_names,
        ["sig.constant", "sig.mean", "sig.sigma", "bkg.p0", "bkg.p1"]
    );
    assert_eq!(m.params, [5.0, 1.0, 0.5, 2.0, 0.1]);
    for x in [-1.0, 0.0, 1.0, 2.5] {
        assert!((m.eval(x) - (sig.eval(x) + bkg.eval(x))).abs() < 1e-12);
    }

    // `+` is the same sum, and chaining appends rather than nests.
    let g = Model::gaussian("g");
    let chained = g.clone() + g.clone() + bkg.clone();
    assert_eq!(chained.name, "g+g+bkg");
    assert_eq!(
        chained.param_names,
        [
            "g.constant",
            "g.mean",
            "g.sigma",
            "g_2.constant",
            "g_2.mean",
            "g_2.sigma",
            "bkg.p0",
            "bkg.p1"
        ]
    );
    assert!((chained.eval(0.3) - (2.0 * g.eval(0.3) + bkg.eval(0.3))).abs() < 1e-12);
}

#[test]
fn a_sum_fits_like_the_same_model_written_by_hand() {
    let data = common::peak_data();
    let by_hand = common::peak_model(2.5).with_params(vec![150.0, 2.5, 1.0, 15.0]);
    let sum = (Model::gaussian("peak") + Model::polynomial("flat", 0))
        .with_params(vec![150.0, 2.5, 1.0, 15.0])
        .lower_limit("peak.sigma", 0.05);
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let a = data.fit_opts(&by_hand, &opts);
    let b = data.fit_opts(&sum, &opts);
    assert!(a.valid && b.valid, "{b:?}");
    for i in 0..4 {
        assert!(
            (a.params[i] - b.params[i]).abs() < 0.05 * a.errors[i],
            "p{i}: {b:?}"
        );
        assert!(
            (a.errors[i] / b.errors[i] - 1.0).abs() < 1e-3,
            "p{i}: {b:?}"
        );
    }
}

#[test]
fn a_part_keeps_its_limits_and_fixed_parameters() {
    let sig = Model::gaussian("sig")
        .with_params(vec![150.0, 2.5, 1.0])
        .lower_limit("sigma", 0.05);
    let bkg = Model::polynomial("bkg", 0)
        .with_params(vec![19.0])
        .fix("p0");
    let fit = common::peak_data().fit_opts(
        &Model::sum("m", &[&sig, &bkg]),
        &FitOptions::new().method(FitMethod::Likelihood),
    );
    assert!(fit.valid, "{fit:?}");
    assert_eq!(fit.params[3], 19.0, "the background stayed fixed: {fit:?}");
    assert_eq!(fit.errors[3], 0.0);
    assert_eq!(fit.ndf, 100 - 3);
}

/// Three measurements and a constraint: the combined estimate of a constant
/// is their inverse-variance weighted average, in closed form.
#[test]
fn a_constraint_combines_like_one_more_measurement() {
    let data = Points::new(&[0.0, 1.0, 2.0], &[10.0, 11.0, 9.5], &[1.0, 2.0, 0.5]).unwrap();
    let (mean, sd) = (12.0, 0.8);
    let weights = [1.0, 0.25, 4.0, 1.0 / (sd * sd)];
    let values = [10.0, 11.0, 9.5, mean];
    let wsum: f64 = weights.iter().sum();
    let want = weights.iter().zip(&values).map(|(w, v)| w * v).sum::<f64>() / wsum;
    let want_err = wsum.sqrt().recip();

    let model = Model::polynomial("c", 0)
        .with_params(vec![0.0])
        .constrain("p0", mean, sd);
    for m in [Minimizer::Minuit2, Minimizer::LevenbergMarquardt] {
        let fit = data.fit_opts(&model, &FitOptions::new().minimizer(m).tolerance(1e-4));
        assert!(fit.valid, "{m:?}: {fit:?}");
        assert!(
            (fit.params[0] - want).abs() < 1e-3 * want_err,
            "{m:?}: {fit:?}"
        );
        assert!(
            (fit.errors[0] / want_err - 1.0).abs() < 1e-6,
            "{m:?}: {fit:?}"
        );
        assert_eq!(fit.ndf, 3, "3 points + 1 constraint − 1 parameter");
        // The χ² includes the constraint's term.
        let chi2: f64 = weights
            .iter()
            .zip(&values)
            .map(|(w, v)| w * (v - fit.params[0]).powi(2))
            .sum();
        assert!((fit.chi2 - chi2).abs() < 1e-6, "{m:?}: {fit:?}");
    }
}

#[test]
fn a_constraint_lets_one_point_determine_two_parameters() {
    // One point cannot fix a line on its own; a constraint on the slope can.
    let data = Points::new(&[2.0], &[5.0], &[0.1]).unwrap();
    let line = Model::polynomial("l", 1).with_params(vec![0.0, 0.0]);
    assert!(
        !data.fit(&line).valid,
        "under-determined without the constraint"
    );
    let fit = data.fit(&line.constrain("p1", 2.0, 0.01));
    assert!(fit.valid, "{fit:?}");
    assert!(
        (fit.params[1] - 2.0).abs() < 1e-3 && (fit.params[0] - 1.0).abs() < 0.05,
        "{fit:?}"
    );
    assert_eq!(fit.ndf, 0);
}

#[test]
fn a_removed_or_fixed_constraint_has_no_effect() {
    let data = Points::new(&[0.0, 1.0], &[10.0, 10.0], &[1.0, 1.0]).unwrap();
    let plain = data.fit(&Model::polynomial("c", 0).with_params(vec![1.0]));
    let removed = data.fit(
        &Model::polynomial("c", 0)
            .with_params(vec![1.0])
            .constrain("p0", 20.0, 1.0)
            .constrain("p0", 20.0, 0.0),
    );
    assert_eq!(plain.params, removed.params);
    assert_eq!(plain.ndf, removed.ndf);

    let fixed = data.fit(
        &Model::polynomial("l", 1)
            .with_params(vec![10.0, 0.0])
            .fix("p1")
            .constrain("p1", 5.0, 1.0),
    );
    assert_eq!(fixed.params[1], 0.0);
    assert_eq!(
        fixed.ndf, 1,
        "a fixed parameter's constraint is not a measurement"
    );
    assert!(fixed.chi2 < 1e-6, "{fixed:?}");
}

#[test]
fn a_constraint_carries_into_a_sum() {
    let sig = Model::gaussian("sig")
        .with_params(vec![150.0, 2.5, 1.0])
        .lower_limit("sigma", 0.05)
        .constrain("sigma", 0.6, 0.01);
    let model = sig + Model::polynomial("bkg", 0).with_params(vec![15.0]);
    let fit =
        common::peak_data().fit_opts(&model, &FitOptions::new().method(FitMethod::Likelihood));
    assert!(fit.valid, "{fit:?}");
    assert!(
        (fit.params[2] - 0.6).abs() < 0.03,
        "pulled to the constraint: {fit:?}"
    );
    assert_eq!(fit.ndf, 100 + 1 - 4);
}
