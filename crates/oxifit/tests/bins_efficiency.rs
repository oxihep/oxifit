//! Bin-integrated binned fits and binomial efficiency fits, against ROOT 6.40
//! (`tests/data/root_bins_efficiency.py`): `TH1::Fit` with and without option
//! `"I"` on the comparison's peak histogram rebinned to 20 bins of width 0.5
//! (where the peak, σ ≈ 0.6, curves within a bin), and `TEfficiency::Fit`.

use oxifit::{BinomialData, FitExt, FitMethod, FitOptions, Model, Points};

const COARSE: [f64; 20] = [
    3700.0, 2939.0, 2502.0, 2019.0, 1676.0, 1389.0, 1064.0, 1005.0, 1289.0, 1867.0, 2116.0, 1498.0,
    698.0, 332.0, 225.0, 198.0, 150.0, 134.0, 101.0, 98.0,
];

fn hist() -> Points {
    let x: Vec<f64> = (0..20).map(|i| 0.25 + 0.5 * f64::from(i)).collect();
    let e: Vec<f64> = COARSE.iter().map(|c| c.sqrt()).collect();
    Points::new(&x, &COARSE, &e).unwrap()
}

fn peak() -> Model {
    Model::new(
        "peak",
        &["A", "mu", "sigma", "c0", "c1"],
        vec![1500.0, 5.0, 0.8, 8.1, -0.35],
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + (p[3] + p[4] * x).exp(),
    )
}

/// (method, integrate, ROOT params, ROOT errors)
const ROOT: [(FitMethod, bool, [f64; 5], [f64; 5]); 4] = [
    (
        FitMethod::Chi2,
        false,
        [1630.726842, 5.184833, 0.613032, 8.304262, -0.396241],
        [35.498302, 0.013229, 0.012535, 0.010904, 0.004322],
    ),
    (
        FitMethod::Chi2,
        true,
        [1678.382516, 5.184816, 0.595553, 8.30261, -0.396215],
        [37.539363, 0.013228, 0.01291, 0.010879, 0.004322],
    ),
    (
        FitMethod::Likelihood,
        false,
        [1629.831687, 5.184792, 0.612675, 8.30402, -0.395767],
        [35.663229, 0.013148, 0.012475, 0.010834, 0.00425],
    ),
    (
        FitMethod::Likelihood,
        true,
        [1677.606882, 5.184807, 0.59512, 8.302361, -0.39573],
        [37.721957, 0.013147, 0.012847, 0.010809, 0.00425],
    ),
];

#[test]
fn bin_integrated_fits_match_root_option_i() {
    let data = hist();
    for (method, integrate, params, errors) in ROOT {
        for model in [peak(), {
            // The same function from built-in shapes: analytic gradients.
            (Model::gaussian("g") + Model::exponential("b"))
                .with_params(vec![1500.0, 5.0, 0.8, 8.1, -0.35])
        }] {
            let fit = data.fit_opts(
                &model,
                &FitOptions::new().method(method).bin_integration(integrate),
            );
            assert!(fit.valid, "{method:?} {integrate}");
            for k in 0..5 {
                let pull = (fit.params[k] - params[k]) / errors[k];
                assert!(
                    pull.abs() < 0.01,
                    "{method:?} I={integrate} p{k}: {} vs {}",
                    fit.params[k],
                    params[k]
                );
                let rel = fit.errors[k] / errors[k] - 1.0;
                assert!(
                    rel.abs() < 0.01,
                    "{method:?} I={integrate} e{k}: {} vs {}",
                    fit.errors[k],
                    errors[k]
                );
            }
        }
    }
}

#[test]
fn explicit_bin_edges_are_used() {
    // The same 20 bins given as edges: the same fit as inferring them.
    let edges: Vec<f64> = (0..=20).map(|i| 0.5 * f64::from(i)).collect();
    let with_edges = hist().with_bin_edges(&edges).unwrap();
    let opts = FitOptions::new()
        .method(FitMethod::Likelihood)
        .bin_integration(true);
    let a = hist().fit_opts(&peak(), &opts);
    let b = with_edges.fit_opts(&peak(), &opts);
    for k in 0..5 {
        assert!((a.params[k] - b.params[k]).abs() < 1e-6 * a.errors[k]);
    }
    // Wrong bins (twice as wide) change the fit: the edges are what is used.
    let wide: Vec<f64> = (0..=20).map(f64::from).collect();
    let c = Points::new(
        &(0..20)
            .map(|i| 0.25 + 0.5 * f64::from(i))
            .collect::<Vec<_>>(),
        &COARSE,
        &[1.0; 20],
    )
    .unwrap()
    .with_bin_edges(&wide)
    .unwrap()
    .fit_opts(&peak(), &opts);
    assert!((c.params[2] - a.params[2]).abs() > 5.0 * a.errors[2]);
    assert!(hist().with_bin_edges(&[0.0, 1.0]).is_err());
}

const X: [f64; 30] = [
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
    26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
];
const PASSED: [f64; 30] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 6.0, 5.0, 13.0, 20.0, 33.0, 26.0, 37.0, 44.0,
    33.0, 37.0, 47.0, 29.0, 39.0, 42.0, 30.0, 33.0, 44.0, 47.0, 35.0, 38.0, 48.0,
];
const TOTAL: [f64; 30] = [
    30.0, 37.0, 44.0, 51.0, 35.0, 42.0, 49.0, 33.0, 40.0, 47.0, 31.0, 38.0, 45.0, 52.0, 36.0, 43.0,
    50.0, 34.0, 41.0, 48.0, 32.0, 39.0, 46.0, 30.0, 37.0, 44.0, 51.0, 35.0, 42.0, 49.0,
];

fn turn_on() -> Model {
    Model::new(
        "turn-on",
        &["threshold", "width", "plateau"],
        vec![20.0, 2.0, 0.9],
        |x, p| p[2] * 0.5 * (1.0 + oxistats::erf((x - p[0]) / (2f64.sqrt() * p[1]))),
    )
    .limit("plateau", 0.0, 1.0)
}

#[test]
fn binomial_efficiency_fit_matches_tefficiency() {
    let data = BinomialData::new(&X, &PASSED, &TOTAL).unwrap();
    let fit = data.fit(&turn_on(), &FitOptions::new());
    assert!(fit.valid, "{fit:?}");
    let params = [22.046703, 2.564744, 0.952759];
    let errors = [0.213002, 0.207715, 0.009601];
    for k in 0..3 {
        assert!(
            ((fit.params[k] - params[k]) / errors[k]).abs() < 0.01,
            "p{k}: {fit:?}"
        );
        assert!(
            (fit.errors[k] / errors[k] - 1.0).abs() < 0.01,
            "e{k}: {} vs {}",
            fit.errors[k],
            errors[k]
        );
    }
    // A goodness-of-fit deviance with its degrees of freedom.
    assert!(fit.chi2.is_finite() && fit.ndf == 27);
}

#[test]
fn unbinned_efficiency_agrees_with_the_binned_one() {
    // Expand the bins into single pass/fail events at the bin centers: the
    // binomial likelihood is then the same up to a constant, so the fit is too.
    let (mut x, mut ok) = (Vec::new(), Vec::new());
    for i in 0..30 {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "whole, non-negative counts"
        )]
        let (total, passed) = (TOTAL[i] as usize, PASSED[i] as usize);
        for j in 0..total {
            x.push(X[i]);
            ok.push(j < passed);
        }
    }
    let events = BinomialData::events(&x, &ok).unwrap();
    let binned = BinomialData::new(&X, &PASSED, &TOTAL)
        .unwrap()
        .fit(&turn_on(), &FitOptions::new());
    let unbinned = events.fit(&turn_on(), &FitOptions::new());
    for k in 0..3 {
        assert!((unbinned.params[k] - binned.params[k]).abs() < 0.01 * binned.errors[k]);
        assert!((unbinned.errors[k] / binned.errors[k] - 1.0).abs() < 0.01);
    }
    assert!(unbinned.chi2.is_nan()); // no saturated model for single events
}

#[test]
fn efficiency_cost_makes_asimov_and_toys() {
    use oxifit::{BinomialCost, Cost, Fitter};
    let data = BinomialData::new(&X, &PASSED, &TOTAL).unwrap();
    let cost = BinomialCost::new(&data, &turn_on(), &FitOptions::new());
    let truth = [22.0, 2.5, 0.95];
    let asimov = cost.asimov(&truth).unwrap();
    let fit = Fitter::new(&*asimov, vec![20.0, 2.0, 0.9])
        .limit("plateau", 0.0, 1.0)
        .fit(&FitOptions::new());
    for k in 0..3 {
        assert!(
            (fit.params[k] - truth[k]).abs() < 0.01 * fit.errors[k],
            "{fit:?}"
        );
    }
    let (a, b) = (cost.toy(&truth, 5).unwrap(), cost.toy(&truth, 5).unwrap());
    assert_eq!(a.value(&truth), b.value(&truth));
    assert!(BinomialData::new(&X, &PASSED[..3], &TOTAL).is_err());
}
