//! Convolutions, checked against the exponentially modified Gaussian.

use oxifit::{FitExt, FitMethod, FitOptions, Model, Points, Toys};

/// A decay `n/τ·exp(−t/τ)` for `t ≥ 0`, with its gradient.
fn decay() -> Model {
    Model::new("decay", &["n", "tau"], vec![1.0, 1.5], |t, p| {
        if t < 0.0 {
            0.0
        } else {
            p[0] / p[1] * (-t / p[1]).exp()
        }
    })
    .with_gradient(|t, p, g| {
        if t >= 0.0 {
            let e = (-t / p[1]).exp();
            g[0] = e / p[1];
            g[1] = p[0] * e * (t - p[1]) / (p[1] * p[1] * p[1]);
        }
    })
}

/// The decay convolved with N(μ, σ): the exponentially modified Gaussian,
/// `n/(2τ)·exp(σ²/(2τ²) − (x − μ)/τ)·erfc((σ²/τ − (x − μ))/(σ√2))`.
fn ex_gaussian(x: f64, n: f64, tau: f64, mu: f64, s: f64) -> f64 {
    let z = (s * s / tau - (x - mu)) / (s * std::f64::consts::SQRT_2);
    n / (2.0 * tau) * (s * s / (2.0 * tau * tau) - (x - mu) / tau).exp() * oxistats::erfc(z)
}

#[test]
fn a_decay_smeared_by_a_gaussian_is_the_ex_gaussian() {
    let (n, tau, mu, s) = (100.0, 1.5, 0.05, 0.12);
    let res = Model::normal("res").with_params(vec![mu, s]);
    let m = Model::convolve(
        "measured",
        &decay().with_params(vec![n, tau]),
        &res,
        (-1.5, 1.5),
    );
    assert_eq!(
        m.param_names(),
        ["decay.n", "decay.tau", "res.mean", "res.sigma"]
    );
    for x in [-0.4, -0.1, 0.0, 0.03, 0.2, 1.0, 3.0, 6.0] {
        let want = ex_gaussian(x, n, tau, mu, s);
        let got = m.eval(x);
        assert!(
            (got - want).abs() < 1e-8 * want.max(1e-6) + 1e-12,
            "x={x}: {got} vs {want}"
        );
    }
    // Its gradient (from both parts') is right.
    assert!(m.has_gradient());
    assert!(m.check_gradient(&[-0.2, 0.0, 0.4, 2.0]) < 1e-6);
    // Without a gradient on a part, there is none.
    let plain = Model::new("d", &["tau"], vec![1.5], |t, p| {
        if t < 0.0 { 0.0 } else { (-t / p[0]).exp() }
    });
    assert!(!Model::convolve("c", &plain, &res, (-1.0, 1.0)).has_gradient());
    // A convolution is not a sum: `+` makes a new sum with it as one part.
    let total = m.clone() + Model::polynomial("bkg", 0);
    assert_eq!(total.param_names().len(), 5);
    assert_eq!(total.param_names()[0], "measured.decay.n");
}

#[test]
fn a_smeared_lifetime_fit_recovers_the_lifetime() {
    // Toys from the smeared decay, fitted with it: τ and σ come back.
    let truth = Model::convolve(
        "measured",
        &decay().with_params(vec![400.0, 1.5]),
        &Model::normal("res").with_params(vec![0.0, 0.15]),
        (-1.5, 1.5),
    )
    .fix("res.mean");
    let xs: Vec<f64> = (0..60).map(|i| -0.5 + 0.1 * (f64::from(i) + 0.5)).collect();
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let study = Toys::binned(&truth, &xs).seed(2).run(&truth, &opts, 40);
    assert!(study.valid_fraction() > 0.95, "{}", study.valid_fraction());
    for name in ["decay.tau", "res.sigma"] {
        let p = study.pull_summary(name).unwrap();
        assert!(p.mean.abs() < 4.0 * p.mean_err, "{name}: {p:?}");
        assert!((p.width - 1.0).abs() < 4.0 * p.width_err, "{name}: {p:?}");
    }
    // One fit, with and without the gradient, lands on the same point.
    let data: Points = Points::from_points(
        xs.iter()
            .map(|&x| {
                let y = truth.eval(x).round();
                oxifit::Point::new(x, y, y.sqrt())
            })
            .collect(),
    );
    let a = data.fit_opts(&truth, &opts);
    let b = data.fit_opts(&truth, &opts.gradient(false));
    assert!(a.valid && b.valid);
    assert!((a.params[1] - b.params[1]).abs() < 0.05 * b.errors[1]);
}
