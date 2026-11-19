//! Analytic gradients: the built-in ones are right, fits with and without
//! them agree, and L-BFGS-B uses them.

mod common;

use oxifit::{Events, FitExt, FitMethod, FitOptions, Minimizer, Model, Points, Simultaneous};

#[test]
fn built_in_gradients_match_finite_differences() {
    let xs = [-2.0, -0.3, 0.0, 0.7, 1.9, 4.2];
    for m in [
        Model::gaussian("g").with_params(vec![3.0, 0.5, 1.3]),
        Model::exponential("e").with_params(vec![0.2, -0.4]),
        Model::polynomial("p", 3).with_params(vec![1.0, -2.0, 0.5, 0.1]),
        Model::gaussian("g").with_params(vec![3.0, 0.5, 1.3]) + Model::polynomial("b", 1),
    ] {
        assert!(m.has_gradient(), "{}", m.name);
        assert!(
            m.check_gradient(&xs) < 1e-6,
            "{}: {}",
            m.name,
            m.check_gradient(&xs)
        );
    }
    // A sum has one only when every part does.
    let cb = Model::crystal_ball("cb");
    assert!(!cb.has_gradient() && cb.check_gradient(&xs).is_nan());
    assert!(!(Model::gaussian("g") + cb.clone()).has_gradient());
    // A wrong gradient is caught.
    let wrong = Model::polynomial("p", 1).with_gradient(|x, _p, g| {
        g[0] = 1.0;
        g[1] = 2.0 * x;
    });
    assert!(wrong.check_gradient(&xs) > 0.1);
    // `gradient` is analytic when it can be, numerical otherwise.
    let g = cb.with_params(vec![1.0, 0.0, 1.0, 1.5, 3.0]).gradient(0.5);
    assert_eq!(g.len(), 5);
}

#[test]
fn fits_agree_with_and_without_the_gradient() {
    let data = common::peak_data();
    let model = (Model::gaussian("peak") + Model::polynomial("flat", 0))
        .with_params(vec![190.0, 2.9, 0.65, 18.0])
        .lower_limit("peak.sigma", 0.05);
    for method in [FitMethod::Chi2, FitMethod::Likelihood] {
        let opts = FitOptions::new().method(method);
        let exact = data.fit_opts(&model, &opts);
        let numeric = data.fit_opts(&model, &opts.with_gradient(false));
        assert!(exact.valid && numeric.valid, "{method:?}");
        for i in 0..4 {
            let d = (exact.params[i] - numeric.params[i]).abs();
            assert!(
                d < 0.05 * numeric.errors[i],
                "{method:?} p{i}: {exact:?} vs {numeric:?}"
            );
            let r = exact.errors[i] / numeric.errors[i];
            // HESSE at two minima a few hundredths of an error apart.
            assert!((r - 1.0).abs() < 1e-2, "{method:?} p{i}: error ratio {r}");
        }
    }
}

#[test]
fn a_poor_seed_with_a_gradient_still_converges() {
    // minuit2's analytic-gradient MIGRAD can stall from here; oxifit falls
    // back to finite differences rather than return a worse fit.
    let data = common::peak_data();
    let model = (Model::gaussian("peak") + Model::polynomial("flat", 0))
        .with_params(vec![150.0, 2.5, 1.0, 15.0])
        .lower_limit("peak.sigma", 0.05);
    let fit = data.fit_opts(&model, &FitOptions::new().method(FitMethod::Likelihood));
    assert!(fit.valid && (fit.params[1] - 3.0).abs() < 0.1, "{fit:?}");
}

#[test]
fn unbinned_and_simultaneous_fits_use_the_gradient() {
    let xs: Vec<f64> = (1..=3000)
        .map(|i| {
            let u1 = (f64::from(i) * 0.618_033_988_75).fract().max(1e-12);
            let u2 = (f64::from(i) * 0.754_877_666_25).fract();
            1.0 + 0.5 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect();
    let events = Events::new(&xs);
    let shape = Model::gaussian("g")
        .with_params(vec![1.0, 0.9, 0.6])
        .fix("constant")
        .lower_limit("sigma", 0.01);
    for method in [
        FitMethod::UnbinnedLikelihood,
        FitMethod::ExtendedUnbinnedLikelihood,
    ] {
        let model = if method == FitMethod::UnbinnedLikelihood {
            shape.clone()
        } else {
            Model::gaussian("g")
                .with_params(vec![2000.0, 0.9, 0.6])
                .lower_limit("sigma", 0.01)
        };
        let opts = FitOptions::new().method(method).range(-2.0, 4.0);
        let exact = events.fit_opts(&model, &opts);
        let numeric = events.fit_opts(&model, &opts.with_gradient(false));
        assert!(exact.valid && numeric.valid, "{method:?}");
        for i in 1..3 {
            assert!((exact.params[i] - numeric.params[i]).abs() < 0.05 * numeric.errors[i]);
        }
    }
    // Two channels sharing the mean: the joint gradient.
    let data = common::peak_data();
    let peak = |b: &str| {
        Model::gaussian("peak")
            .with_params(vec![190.0, 2.9, 0.65])
            .lower_limit("sigma", 0.05)
            + Model::polynomial(b, 0).with_params(vec![18.0])
    };
    let lik = FitOptions::new().method(FitMethod::Likelihood);
    let sim = Simultaneous::new()
        .channel("a", &data, &peak("bkg_a"), &lik)
        .channel("b", &data, &peak("bkg_b"), &lik);
    let exact = sim.fit(&FitOptions::new());
    let numeric = sim.fit(&FitOptions::new().with_gradient(false));
    assert!(exact.valid && numeric.valid);
    for i in 0..exact.params.len() {
        assert!((exact.params[i] - numeric.params[i]).abs() < 0.05 * numeric.errors[i]);
    }
}

#[test]
fn lbfgsb_agrees_with_migrad_and_holds_limits() {
    let data = common::peak_data();
    let model = (Model::gaussian("peak") + Model::polynomial("flat", 0))
        .with_params(vec![150.0, 2.5, 1.0, 15.0])
        .lower_limit("peak.sigma", 0.05);
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let migrad = data.fit_opts(&model, &opts);
    for gradient in [true, false] {
        let fit = data.fit_opts(
            &model,
            &opts.minimizer(Minimizer::LBfgsB).with_gradient(gradient),
        );
        assert!(fit.valid, "gradient={gradient}: {fit:?}");
        for i in 0..4 {
            assert!((fit.params[i] - migrad.params[i]).abs() < 0.05 * migrad.errors[i]);
        }
    }
    // A formula model (no analytic gradient) and a limit it must stop at.
    let line = Points::unweighted(&[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 4.0, 6.0]).unwrap();
    let capped = Model::from_formula("l", "[0]+[1]*x")
        .unwrap()
        .with_params(vec![0.0, 1.0])
        .upper_limit("p1", 1.5);
    let fit = line.fit_opts(&capped, &FitOptions::new().minimizer(Minimizer::LBfgsB));
    assert!((fit.params[1] - 1.5).abs() < 1e-3, "{fit:?}");
}

/// MINOS with an analytic gradient: the gradient path builds its covariance
/// from the gradient, and MINOS must not start from MIGRAD's running error
/// estimate (which gave a ±0.0002 interval for an amplitude known to ±4.5).
#[test]
fn minos_with_an_analytic_gradient_matches_the_parabolic_errors() {
    use oxifit::{FitExt, FitMethod, FitOptions, Model, Points};
    let x: Vec<f64> = (0..40).map(|i| 0.25 + 0.5 * f64::from(i)).collect();
    let truth =
        |v: f64| 60.0 * (-0.5 * ((v - 9.0) / 1.2f64).powi(2)).exp() + 30.0 * (-v / 8.0).exp();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (truth(v) + (f64::from(i as u32) * 2.3).sin() * truth(v).sqrt()).round())
        .collect();
    let data = Points::unweighted(&x, &y).unwrap();
    let model = (Model::gaussian("peak") + Model::exponential("bkg"))
        .with_params(vec![50.0, 8.5, 1.5, 3.0, -0.1])
        .limit("peak.sigma", 0.1, 5.0)
        .limit("peak.constant", 0.0, 500.0);
    let fit = data.fit_opts(
        &model,
        &FitOptions::new()
            .method(FitMethod::Likelihood)
            .with_minos(true),
    );
    assert!(fit.valid);
    for (k, &(lo, hi)) in fit.minos.as_ref().unwrap().iter().enumerate() {
        let half = 0.5 * (hi - lo);
        assert!(
            (half / fit.errors[k] - 1.0).abs() < 0.1,
            "p{k}: MINOS ({lo}, {hi}) vs ±{}",
            fit.errors[k]
        );
    }
}
