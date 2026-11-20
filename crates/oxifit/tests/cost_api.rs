//! The public cost API: a data fit as a [`ModelCost`] through a [`Fitter`]
//! is the same fit as `FitExt::fit_opts`; a [`CostSum`] is the simultaneous
//! fit; closures, gradients, settings, profiles and contours work on any cost.

use oxifit::{
    Cost, CostSum, Events, FitExt, FitMethod, FitOptions, Fitter, FnCost, Model, ModelCost, Points,
    Simultaneous,
};

fn peak_data() -> Points {
    let x: Vec<f64> = (0..60).map(|i| 0.1 * f64::from(i) + 0.05).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let truth = 80.0 * (-0.5 * ((x - 3.0) / 0.5f64).powi(2)).exp() + 10.0 - x;
            // A fixed, deterministic scatter around the truth.
            (truth + ((i as f64) * 1.7).sin() * truth.sqrt()).round()
        })
        .collect();
    let e: Vec<f64> = y.iter().map(|v| v.sqrt().max(1.0)).collect();
    Points::new(&x, &y, &e).unwrap()
}

fn peak_model() -> Model {
    (Model::gaussian("g") + Model::polynomial("b", 1))
        .with_params(vec![60.0, 2.8, 0.7, 8.0, -0.5])
        .limit("g.sigma", 0.05, 3.0)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn model_cost_through_fitter_is_the_data_fit() {
    let data = peak_data();
    let model = peak_model().constrain("b.p1", -1.0, 0.5);
    for method in [FitMethod::Chi2, FitMethod::Likelihood] {
        let opts = FitOptions::new().method(method);
        let direct = data.fit_opts(&model, &opts);
        let cost = ModelCost::new(&data, &model, &opts);
        let via = cost.fitter().fit(&opts);
        assert!(direct.valid && via.valid, "{method:?}");
        for k in 0..direct.params.len() {
            assert!(
                close(via.params[k], direct.params[k], 1e-9),
                "{method:?} p{k}"
            );
            assert!(
                close(via.errors[k], direct.errors[k], 1e-9),
                "{method:?} e{k}"
            );
        }
        assert!(close(via.fval, direct.fval, 1e-12));
        assert!(close(via.chi2, direct.chi2, 1e-12));
        assert_eq!(via.ndf, direct.ndf);
        // The cost at a point is the data cost alone (no constraint)…
        let p = &direct.params;
        assert!(close(
            cost.value(p),
            direct.fval - ((p[4] + 1.0) / 0.5).powi(2),
            1e-12
        ));
        // …and the fitter adds the constraint.
        assert!(close(cost.fitter().value(p), direct.fval, 1e-12));
    }
}

#[test]
fn unbinned_model_cost_matches_the_events_fit() {
    let xs: Vec<f64> = (0..2000)
        .map(|i| {
            // Deterministic quantiles of an exponential of slope 0.7 on [0, 5).
            let u = (f64::from(i) + 0.5) / 2000.0;
            -(1.0 - u * (1.0 - (-3.5f64).exp())).ln() / 0.7
        })
        .collect();
    let events = Events::new(&xs);
    let model = Model::new("e", &["n", "l"], vec![1500.0, 0.5], |x, p| {
        p[0] * p[1] * (-p[1] * x).exp()
    })
    .lower_limit("l", 0.01);
    let opts = FitOptions::new()
        .method(FitMethod::ExtendedUnbinnedLikelihood)
        .range(0.0, 5.0);
    let direct = events.fit_opts(&model, &opts);
    let via = ModelCost::new(&events, &model, &opts).fitter().fit(&opts);
    assert!(direct.valid && via.valid);
    for k in 0..2 {
        assert!(close(via.params[k], direct.params[k], 1e-9));
        assert!(close(via.errors[k], direct.errors[k], 1e-9));
    }
}

#[test]
fn cost_sum_is_the_simultaneous_fit() {
    let a = Points::new(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.1, 4.9, 7.2], &[0.2; 4]).unwrap();
    let b = Points::new(&[0.0, 1.0, 2.0, 3.0], &[5.0, 7.0, 9.1, 10.8], &[0.3; 4]).unwrap();
    let line = |c: &'static str| {
        Model::new("line", &[c, "slope"], vec![0.0, 1.0], |x, p| {
            p[0] + p[1] * x
        })
    };
    let opts = FitOptions::new();
    let sim = Simultaneous::new()
        .channel("A", &a, &line("c_a"), &opts)
        .channel("B", &b, &line("c_b"), &opts)
        .fit(&opts);
    let sum = CostSum::new()
        .with(ModelCost::new(&a, &line("c_a"), &opts))
        .with(ModelCost::new(&b, &line("c_b"), &opts));
    assert_eq!(sum.param_names(), ["c_a", "slope", "c_b"]);
    assert_eq!(sum.n_measurements(), Some(8));
    assert!(sum.is_gof() && !sum.has_gradient()); // closure models
    let fit = sum.fitter(vec![0.0, 1.0, 0.0]).fit(&opts);
    assert!(sim.valid && fit.valid);
    for k in 0..3 {
        assert!(close(fit.params[k], sim.params[k], 1e-9));
        assert!(close(fit.errors[k], sim.errors[k], 1e-9));
    }
    assert!(close(fit.chi2, sim.chi2, 1e-12));
    assert_eq!(fit.ndf, sim.ndf);
}

/// A correlated 2-D Gaussian `−2 ln L`: means (1, −2), σ (0.5, 2), ρ = 0.6.
fn gauss2() -> FnCost {
    let (m, s, rho) = ([1.0, -2.0], [0.5, 2.0], 0.6);
    let q = move |p: &[f64]| {
        let (u, v) = ((p[0] - m[0]) / s[0], (p[1] - m[1]) / s[1]);
        (u * u - 2.0 * rho * u * v + v * v) / (1.0 - rho * rho)
    };
    FnCost::new(&["a", "b"], q).with_gradient(move |p, g| {
        let (u, v) = ((p[0] - m[0]) / s[0], (p[1] - m[1]) / s[1]);
        let k = 2.0 / (1.0 - rho * rho);
        g[0] = k * (u - rho * v) / s[0];
        g[1] = k * (v - rho * u) / s[1];
    })
}

#[test]
fn closure_cost_gives_the_gaussian_errors_and_correlation() {
    let cost = gauss2();
    for gradient in [true, false] {
        let fit = Fitter::new(&cost, vec![0.0, 0.0]).fit(&FitOptions::new().gradient(gradient));
        assert!(fit.valid);
        assert!(close(fit.params[0], 1.0, 1e-5) && close(fit.params[1], -2.0, 1e-5));
        assert!(close(fit.errors[0], 0.5, 1e-4) && close(fit.errors[1], 2.0, 1e-4));
        let c = fit.covariance.as_ref().unwrap();
        assert!(close(c[0][1] / (c[0][0] * c[1][1]).sqrt(), 0.6, 1e-4));
        // A cost that does not count measurements: no ndf, no χ².
        assert_eq!(fit.ndf, 0);
        assert!(fit.chi2.is_nan());
    }
}

#[test]
fn every_minimizer_runs_on_a_cost() {
    use oxifit::{CmaEs, DifferentialEvolution, Minimizer, MultiStart};
    let cost = gauss2();
    // Steps of the errors' scale: SIMPLEX, alone, needs them.
    let fitter = Fitter::new(&cost, vec![0.3, 0.3])
        .limit("a", -5.0, 5.0)
        .limit("b", -10.0, 10.0)
        .step("a", 0.5)
        .step("b", 2.0);
    for m in [
        Minimizer::Minuit2,
        Minimizer::Minuit2Combined,
        Minimizer::Simplex,
        Minimizer::LBfgsB,
        Minimizer::LevenbergMarquardt,
        Minimizer::MultiStart(MultiStart::default()),
        Minimizer::DifferentialEvolution(DifferentialEvolution::default()),
        Minimizer::CmaEs(CmaEs::default()),
    ] {
        let fit = fitter.fit(&FitOptions::new().minimizer(m));
        assert!(fit.valid, "{m:?}: {fit:?}");
        // Within a hundredth of a standard deviation (MIGRAD's EDM goal);
        // SIMPLEX alone stops coarser, as in ROOT, so a tenth for it.
        let tol = if matches!(m, Minimizer::Simplex) {
            0.1
        } else {
            0.01
        };
        let ok = (fit.params[0] - 1.0).abs() < tol * 0.5 && (fit.params[1] + 2.0).abs() < tol * 2.0;
        assert!(ok, "{m:?}: {:?}", fit.params);
    }
}

#[test]
fn settings_fix_release_constrain_and_bad_names() {
    let cost = gauss2();
    // Fixed b = 0: the conditional minimum a = 1 + ρ·(σa/σb)·(0 − (−2)) = 1.3.
    let fixed = Fitter::new(&cost, vec![0.0, 0.0]).fix("b");
    assert_eq!(fixed.is_fixed("b"), Some(true));
    let fit = fixed.fit(&FitOptions::new());
    assert!(fit.valid && close(fit.params[0], 1.3, 1e-5) && fit.params[1] == 0.0);
    let fit = fixed.clone().release("b").fit(&FitOptions::new());
    assert!(close(fit.params[1], -2.0, 1e-4));
    // A constraint b = 0 ± 2 (as precise as the cost's): the minimum moves halfway.
    let fit = Fitter::new(&cost, vec![0.0, 0.0])
        .constrain("b", 0.0, 2.0)
        .fit(&FitOptions::new());
    assert!(close(fit.params[1], -1.0, 1e-4));
    // A setter's typo, an unknown `set`, or the wrong number of seeds: invalid.
    assert!(
        !Fitter::new(&cost, vec![0.0, 0.0])
            .fix("c")
            .fit(&FitOptions::new())
            .valid
    );
    assert!(
        !Fitter::new(&cost, vec![0.0, 0.0])
            .set("c", 1.0)
            .fit(&FitOptions::new())
            .valid
    );
    assert!(!Fitter::new(&cost, vec![0.0]).fit(&FitOptions::new()).valid);
    let set = Fitter::new(&cost, vec![0.0, 0.0]).set("b", 5.0).fix("b");
    assert_eq!(set.fit(&FitOptions::new()).params[1], 5.0);
}

#[test]
fn profile_and_contour_of_a_cost() {
    let cost = gauss2();
    let fitter = Fitter::new(&cost, vec![0.0, 0.0]);
    let opts = FitOptions::new();
    let prof = fitter.profile("a", (0.0, 2.0), 41, &opts).unwrap();
    let (lo, hi) = prof.interval(1.0).unwrap();
    assert!(close(lo, 0.5, 2e-3) && close(hi, 1.5, 2e-3), "{lo} {hi}");
    let contour = fitter.contour(("a", "b"), 1.0, 16, &opts).unwrap();
    // On the Δ = 1 contour, the quadratic form is 1 at every point.
    for &(a, b) in &contour.points {
        let v = cost.value(&[a, b]);
        assert!(close(v, 1.0, 1e-3), "{v}");
    }
    assert!(matches!(
        fitter.profile("zz", (0.0, 1.0), 5, &opts),
        Err(oxifit::FitError::UnknownParameter(_))
    ));
    assert!(matches!(
        fitter.clone().fix("a").profile("a", (0.0, 1.0), 5, &opts),
        Err(oxifit::FitError::FixedParameter(_))
    ));
}

#[test]
fn a_seed_on_a_limit_still_moves() {
    // Minuit's limit transform is flat at the limit; a parameter seeded
    // exactly there must still be fitted (a signal strength seeded at 0).
    let cost = FnCost::new(&["mu"], |p| ((p[0] - 2.0) / 0.5).powi(2));
    for fitter in [
        Fitter::new(&cost, vec![0.0]).lower_limit("mu", 0.0),
        Fitter::new(&cost, vec![0.0]).limit("mu", 0.0, 10.0),
        Fitter::new(&cost, vec![10.0]).limit("mu", 0.0, 10.0),
        Fitter::new(&cost, vec![5.0]).upper_limit("mu", 5.0),
    ] {
        let fit = fitter.fit(&FitOptions::new());
        assert!(fit.valid && (fit.params[0] - 2.0).abs() < 1e-3, "{fit:?}");
    }
    // A fixed parameter on its limit stays there.
    let fit = Fitter::new(&cost, vec![0.0])
        .lower_limit("mu", 0.0)
        .fix("mu")
        .fit(&FitOptions::new());
    assert_eq!(fit.params[0], 0.0);
}
