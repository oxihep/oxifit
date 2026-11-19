//! The minimizer backends — local Minuit2 variants and the global searches —
//! and the Minuit2 options.

mod common;

use oxifit::{
    CmaEs, DifferentialEvolution, FitExt, FitMethod, FitOptions, FitResult, Loss, Minimizer, Model,
    MultiStart, Points,
};

fn line() -> (Points, Model) {
    let data = Points::new(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[1.1, 2.9, 5.2, 6.8, 9.1, 11.0],
        &[0.3, 0.2, 0.4, 0.3, 0.5, 0.2],
    )
    .unwrap();
    let model = Model::polynomial("line", 1)
        .with_params(vec![0.0, 1.0])
        .limit("p0", -10.0, 10.0)
        .limit("p1", -10.0, 10.0);
    (data, model)
}

fn global() -> [Minimizer; 3] {
    [
        Minimizer::MultiStart(MultiStart::new(8)),
        Minimizer::DifferentialEvolution(DifferentialEvolution::new()),
        Minimizer::CmaEs(CmaEs::new()),
    ]
}

#[test]
fn every_minuit_backend_agrees_on_a_line() {
    let (data, model) = line();
    let reference = data.fit(&model);
    assert!(reference.valid, "{reference:?}");
    let mut backends = vec![
        Minimizer::Minuit2Combined,
        Minimizer::Simplex,
        Minimizer::LevenbergMarquardt,
        Minimizer::LBfgsB,
    ];
    backends.extend(global());
    for m in backends {
        let fit = data.fit_opts(&model, &FitOptions::new().minimizer(m));
        assert!(fit.valid, "{m:?}: {fit:?}");
        for i in 0..2 {
            // Same minimum (to MIGRAD's precision) and the same HESSE errors.
            let pull = (fit.params[i] - reference.params[i]) / reference.errors[i];
            assert!(pull.abs() < 0.05, "{m:?} p{i}: {fit:?}");
            assert!(
                (fit.errors[i] / reference.errors[i] - 1.0).abs() < 1e-3,
                "{m:?}: {fit:?}"
            );
        }
        assert!(fit.nfcn > 0 && fit.edm.is_finite(), "{m:?}: {fit:?}");
    }
}

#[test]
fn a_global_search_finds_the_peak_a_poor_seed_misses() {
    let data = common::peak_data();
    let model = common::peak_model(-8.0).limit("mu", -10.0, 10.0);
    let lik = FitOptions::new().method(FitMethod::Likelihood);

    // MIGRAD from the far seed settles on the flat background instead.
    let local = data.fit_opts(&model, &lik);
    assert!(
        !local.valid || (local.params[1] - 3.0).abs() > 1.0,
        "the local fit was expected to miss: {local:?}"
    );

    for m in global() {
        let fit = data.fit_opts(&model, &lik.minimizer(m));
        assert!(fit.valid, "{m:?}: {fit:?}");
        assert!((fit.params[1] - 3.0).abs() < 0.2, "{m:?}: {fit:?}");
        assert!(fit.params[2] > 0.3 && fit.params[2] < 1.0, "{m:?}: {fit:?}");
        assert!(fit.chi2 < local.chi2 || !local.valid, "{m:?}: {fit:?}");
        assert!(fit.nfcn > local.nfcn, "a global search costs more: {fit:?}");
    }
}

#[test]
fn a_global_search_is_reproducible_and_keeps_fixed_parameters() {
    let data = common::peak_data();
    let model = common::peak_model(-8.0)
        .limit("mu", -10.0, 10.0)
        .with_params(vec![100.0, -8.0, 1.0, 19.0])
        .fix("c");
    for m in global() {
        let opts = FitOptions::new().method(FitMethod::Likelihood).minimizer(m);
        let a = data.fit_opts(&model, &opts);
        let b = data.fit_opts(&model, &opts);
        assert_eq!(a, b, "{m:?}: the same seed must give the same fit");
        assert_eq!(a.params[3], 19.0, "{m:?}: a fixed parameter moved: {a:?}");
        assert_eq!(a.errors[3], 0.0, "{m:?}: {a:?}");
    }
    // A different generator seed explores differently but lands in the same minimum.
    let other = Minimizer::DifferentialEvolution(DifferentialEvolution::new().seed(99));
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let c = data.fit_opts(&model, &opts.minimizer(other));
    assert!((c.params[1] - 3.0).abs() < 0.2, "{c:?}");
}

#[test]
fn a_global_search_stays_inside_the_limits() {
    // The data want a slope of 2; the box (and so the answer) stops at 1.5.
    let data = Points::unweighted(&[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 4.0, 6.0]).unwrap();
    let model = Model::polynomial("line", 1)
        .with_params(vec![0.0, 1.0])
        .limit("p0", -5.0, 5.0)
        .limit("p1", -1.5, 1.5);
    for m in global() {
        let fit = data.fit_opts(&model, &FitOptions::new().minimizer(m));
        assert!(fit.params[1] <= 1.5 + 1e-9, "{m:?}: {fit:?}");
        assert!((fit.params[1] - 1.5).abs() < 1e-3, "{m:?}: {fit:?}");
    }
}

#[test]
fn a_tighter_tolerance_locates_the_minimum_more_precisely() {
    // A constant fitted to counts with σ² = y: the χ² minimum is the harmonic mean.
    let ys = [12.0, 7.0, 9.0, 15.0, 11.0, 8.0, 10.0, 13.0];
    let x: Vec<f64> = (0..ys.len()).map(|i| i as f64).collect();
    let s: Vec<f64> = ys.iter().map(|y: &f64| y.sqrt()).collect();
    let data = Points::new(&x, &ys, &s).unwrap();
    let harmonic = ys.len() as f64 / ys.iter().map(|y| 1.0 / y).sum::<f64>();
    let model = Model::polynomial("c", 0).with_params(vec![5.0]);
    let fit = data.fit_opts(&model, &FitOptions::new().tolerance(1e-6).strategy(2));
    assert!(fit.valid && fit.edm < 1e-8, "{fit:?}");
    assert!(
        (fit.params[0] - harmonic).abs() < 1e-4 * fit.errors[0],
        "{fit:?}"
    );
}

#[test]
fn a_call_limit_too_small_to_converge_is_invalid() {
    let data = common::peak_data();
    let model = common::peak_model(2.0);
    let fit = data.fit_opts(&model, &FitOptions::new().max_fcn(5));
    assert!(!fit.valid, "{fit:?}");
}

#[test]
fn a_robust_loss_has_no_chi2_but_keeps_its_cost() {
    let (data, _) = line();
    // Seeded near the line: far out, a Cauchy cost is almost flat.
    let model = Model::polynomial("line", 1).with_params(vec![1.0, 2.0]);
    let fit = data.fit_opts(&model, &FitOptions::new().loss(Loss::Cauchy));
    assert!(fit.valid, "{fit:?}");
    assert!(fit.chi2.is_nan() && fit.p_value().is_nan(), "{fit:?}");
    assert!(fit.fval.is_finite() && fit.fval >= 0.0, "{fit:?}");
    // Without one, the two coincide.
    let plain = data.fit(&model);
    assert_eq!(plain.chi2, plain.fval);
    assert!(plain.p_value() > 0.0);
}

#[test]
fn a_result_can_be_built_by_hand() {
    let r = FitResult::new(vec![1.0, 2.0], vec![0.1, 0.0], 3.5, 4, true);
    assert_eq!(r.fval, 3.5);
    assert!(r.minos.is_none() && r.covariance.is_none() && r.edm.is_nan());
    assert_eq!(r.nfcn, 0);
    assert!((r.chi2_per_ndf() - 0.875).abs() < 1e-12);
}

#[test]
fn levenberg_marquardt_minimizes_every_cost() {
    // LM works on residuals whose squares must sum to the cost: it must land
    // on the same minimum as MIGRAD for every method, robust loss included.
    let data = common::peak_data();
    let model = common::peak_model(3.2).with_params(vec![180.0, 3.2, 0.7, 18.0]);
    for method in [
        FitMethod::Chi2,
        FitMethod::PearsonChi2,
        FitMethod::Likelihood,
    ] {
        for loss in [Loss::Linear, Loss::SoftL1] {
            let opts = FitOptions::new().method(method).loss(loss);
            let migrad = data.fit_opts(&model, &opts);
            let lm = data.fit_opts(&model, &opts.minimizer(Minimizer::LevenbergMarquardt));
            assert!(migrad.valid && lm.valid, "{method:?}/{loss:?}: {lm:?}");
            assert!(
                (lm.fval - migrad.fval).abs() < 1e-3,
                "{method:?}/{loss:?}: LM {} vs MIGRAD {}",
                lm.fval,
                migrad.fval
            );
            for i in 0..4 {
                let pull = (lm.params[i] - migrad.params[i]) / migrad.errors[i];
                assert!(pull.abs() < 0.05, "{method:?}/{loss:?} p{i}: {lm:?}");
            }
        }
    }
}

#[cfg(feature = "argmin")]
#[test]
fn nelder_mead_holds_one_and_two_sided_limits_exactly() {
    // The data want a slope of 2 and an intercept of 0.
    let data = Points::unweighted(&[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 4.0, 6.0]).unwrap();
    let nm = FitOptions::new().minimizer(Minimizer::NelderMead);
    for (model, slope) in [
        (
            Model::polynomial("l", 1)
                .with_params(vec![0.0, 1.0])
                .limit("p1", -1.0, 1.5),
            1.5,
        ),
        (
            Model::polynomial("l", 1)
                .with_params(vec![0.0, 1.0])
                .upper_limit("p1", 1.2),
            1.2,
        ),
        (
            Model::polynomial("l", 1)
                .with_params(vec![0.0, 3.0])
                .lower_limit("p1", 2.5),
            2.5,
        ),
    ] {
        let fit = data.fit_opts(&model, &nm);
        assert!((fit.params[1] - slope).abs() < 1e-4, "{fit:?}");
        // With the slope held at the limit, the best intercept is
        // mean(y) − slope·mean(x).
        let intercept = 3.0 - slope * 1.5;
        assert!((fit.params[0] - intercept).abs() < 1e-3, "{fit:?}");
    }
    // An interior minimum is unaffected by a far limit.
    let wide = Model::polynomial("l", 1)
        .with_params(vec![0.5, 1.0])
        .limit("p1", -10.0, 10.0);
    let fit = data.fit_opts(&wide, &nm);
    assert!(fit.valid && (fit.params[1] - 2.0).abs() < 1e-5, "{fit:?}");
}

#[test]
fn the_default_rescues_a_far_seed_that_migrad_alone_misses() {
    use oxifit::Events;
    // 50 000 Gaussian events; seeded far off, MIGRAD alone stalls (minuit2
    // 0.5), the default (MIGRAD with a SIMPLEX fallback) converges.
    let xs: Vec<f64> = (1..=50_000)
        .map(|i| {
            let u1 = (f64::from(i) * 0.618_033_988_75).fract().max(1e-12);
            let u2 = (f64::from(i) * 0.754_877_666_25).fract();
            2.5 + 0.7 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect();
    let events = Events::new(&xs);
    let shape = Model::new("g", &["mu", "s"], vec![2.0, 1.5], |x, p| {
        (-0.5 * ((x - p[0]) / p[1]).powi(2)).exp()
    })
    .lower_limit("s", 0.01);
    let opts = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(0.0, 5.0);
    assert_eq!(FitOptions::new().minimizer, Minimizer::Minuit2Combined);
    let migrad = events.fit_opts(&shape, &opts.minimizer(Minimizer::Minuit2));
    let default = events.fit_opts(&shape, &opts);
    assert!(
        !migrad.valid,
        "MIGRAD alone was expected to stall: {migrad:?}"
    );
    assert!(
        default.valid && (default.params[0] - 2.5).abs() < 0.01,
        "{default:?}"
    );

    // Where MIGRAD succeeds, the default is MIGRAD: the same result, the same cost.
    let (data, model) = line();
    let a = data.fit_opts(&model, &FitOptions::new().minimizer(Minimizer::Minuit2));
    let b = data.fit(&model);
    assert_eq!(a, b);
}
