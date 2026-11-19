//! Unbinned and extended unbinned likelihood fits, and the SumW2 correction
//! for weighted events — checked against their closed-form answers.

use oxifit::{Events, FitExt, FitMethod, FitOptions, Minimizer, Model};

/// `n` draws from N(mean, sigma) (SplitMix64 + Box–Muller, reproducible).
fn gaussian_sample(n: usize, mean: f64, sigma: f64, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut uniform = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (1.0 - uniform(), uniform());
            mean + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

/// The sample mean and the ML (1/n) standard deviation.
fn moments(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (
        m,
        (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt(),
    )
}

/// A Gaussian *shape* (no amplitude): what the plain unbinned likelihood fits.
fn shape() -> Model {
    Model::new("g", &["mu", "s"], vec![0.5, 1.0], |x, p| {
        (-0.5 * ((x - p[0]) / p[1]).powi(2)).exp()
    })
    .lower_limit("s", 0.01)
}

/// A Gaussian *density* with a yield: what the extended likelihood fits.
fn density() -> Model {
    Model::new("n·g", &["n", "mu", "s"], vec![1000.0, 0.5, 1.0], |x, p| {
        p[0] / (p[2] * (std::f64::consts::TAU).sqrt()) * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp()
    })
    .lower_limit("s", 0.01)
}

/// A wide range: truncating N(1, 0.5) at ±12σ changes nothing measurable.
fn opts(method: FitMethod) -> FitOptions {
    FitOptions::new()
        .method(method)
        .range(-5.0, 7.0)
        .tolerance(1e-3)
}

#[test]
fn unbinned_gaussian_fit_is_the_sample_mean_and_ml_width() {
    let x = gaussian_sample(2000, 1.0, 0.5, 1);
    let (m, sd) = moments(&x);
    let n = x.len() as f64;
    let fit = Events::new(&x).fit_opts(&shape(), &opts(FitMethod::UnbinnedLikelihood));
    assert!(fit.valid, "{fit:?}");
    assert!(
        (fit.params[0] - m).abs() < 0.01 * fit.errors[0],
        "{fit:?} vs {m}"
    );
    assert!(
        (fit.params[1] - sd).abs() < 0.01 * fit.errors[1],
        "{fit:?} vs {sd}"
    );
    // Fisher information: σ/√n for the mean, σ/√(2n) for the width.
    assert!(
        (fit.errors[0] / (sd / n.sqrt()) - 1.0).abs() < 0.01,
        "{fit:?}"
    );
    assert!(
        (fit.errors[1] / (sd / (2.0 * n).sqrt()) - 1.0).abs() < 0.01,
        "{fit:?}"
    );
    // No goodness of fit for an unbinned likelihood.
    assert!(fit.chi2.is_nan() && fit.p_value().is_nan() && fit.fval.is_finite());
    assert_eq!(fit.ndf, 2000 - 2);
}

#[test]
fn extended_fit_yield_is_the_event_count() {
    let x = gaussian_sample(1500, 1.0, 0.5, 2);
    let (m, sd) = moments(&x);
    let fit = Events::new(&x).fit_opts(&density(), &opts(FitMethod::ExtendedUnbinnedLikelihood));
    assert!(fit.valid, "{fit:?}");
    // ∂/∂n: Σ 1/n − ∫g = 0, so n̂ = N (the shape integrates to 1), ± √N.
    assert!(
        (fit.params[0] - 1500.0).abs() < 0.01 * fit.errors[0],
        "{fit:?}"
    );
    assert!(
        (fit.errors[0] / 1500f64.sqrt() - 1.0).abs() < 0.01,
        "{fit:?}"
    );
    assert!((fit.params[1] - m).abs() < 0.01 * fit.errors[1], "{fit:?}");
    assert!((fit.params[2] - sd).abs() < 0.01 * fit.errors[2], "{fit:?}");
}

#[test]
fn events_outside_the_range_are_ignored() {
    let x = gaussian_sample(1000, 1.0, 0.5, 3);
    let mut with_outliers = x.clone();
    with_outliers.extend([-40.0, 25.0, 31.0]);
    let o = opts(FitMethod::UnbinnedLikelihood);
    let a = Events::new(&x).fit_opts(&shape(), &o);
    let b = Events::new(&with_outliers).fit_opts(&shape(), &o);
    assert_eq!(a.params, b.params);
    assert_eq!(a.ndf, b.ndf);
}

#[test]
fn sumw2_gives_weighted_events_the_errors_their_weights_carry() {
    // Every weight 2: the same estimates; plain errors pretend there are 2N
    // events (√2 too small), the SumW2 sandwich restores the N-event errors.
    let x = gaussian_sample(1200, 1.0, 0.5, 4);
    let w = vec![2.0; x.len()];
    for (method, model) in [
        (FitMethod::UnbinnedLikelihood, shape()),
        (FitMethod::ExtendedUnbinnedLikelihood, density()),
    ] {
        let o = opts(method);
        let plain = Events::new(&x).fit_opts(&model, &o);
        let weighted = Events::weighted(&x, &w).unwrap();
        let naive = weighted.fit_opts(&model, &o);
        let corrected = weighted.fit_opts(&model, &o.with_sumw2(true));
        assert!(plain.valid && naive.valid && corrected.valid, "{method:?}");
        let shift = usize::from(method == FitMethod::ExtendedUnbinnedLikelihood);
        for i in shift..model.params.len() {
            assert!((naive.params[i] - plain.params[i]).abs() < 0.02 * plain.errors[i]);
            let r = naive.errors[i] / plain.errors[i];
            assert!(
                (r * 2f64.sqrt() - 1.0).abs() < 0.02,
                "{method:?} p{i}: naive ×{r}"
            );
            let r = corrected.errors[i] / plain.errors[i];
            assert!((r - 1.0).abs() < 0.02, "{method:?} p{i}: corrected ×{r}");
        }
        if shift == 1 {
            // The weighted yield is Σw = 2N, its corrected error √Σw² = 2√N.
            assert!(
                (corrected.params[0] / 2400.0 - 1.0).abs() < 1e-3,
                "{corrected:?}"
            );
            let want = (4.0 * 1200f64).sqrt();
            assert!(
                (corrected.errors[0] / want - 1.0).abs() < 0.02,
                "{corrected:?}"
            );
        }
    }
}

#[test]
fn least_squares_backends_fall_back_to_migrad_on_an_unbinned_fit() {
    let x = gaussian_sample(800, 1.0, 0.5, 5);
    let o = opts(FitMethod::UnbinnedLikelihood);
    let migrad = Events::new(&x).fit_opts(&shape(), &o);
    let lm = Events::new(&x).fit_opts(&shape(), &o.minimizer(Minimizer::LevenbergMarquardt));
    assert!(lm.valid, "{lm:?}");
    assert!((lm.params[0] - migrad.params[0]).abs() < 0.02 * migrad.errors[0]);
}

#[test]
fn weighted_events_need_matching_lengths() {
    assert!(Events::weighted(&[1.0, 2.0], &[1.0]).is_err());
    let e = Events::weighted(&[1.0, 2.0], &[0.5, 1.5]).unwrap();
    assert_eq!((e.len(), e.weights()), (2, Some(&[0.5, 1.5][..])));
}
