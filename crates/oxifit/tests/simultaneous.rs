//! Simultaneous fits of several datasets with shared parameters.

mod common;

use oxifit::{Events, FitExt, FitMethod, FitOptions, Minimizer, Model, Points, Simultaneous};

fn line(intercept: &'static str) -> Model {
    Model::new("line", &[intercept, "b"], vec![0.0, 1.0], |x, p| {
        p[0] + p[1] * x
    })
}

/// Solve the 3×3 system `m·θ = v` (Cramer's rule — fine for a test).
fn solve3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut mk = m;
        for r in 0..3 {
            mk[r][k] = v[r];
        }
        *o = det(mk) / d;
    }
    out
}

#[test]
fn two_lines_with_a_shared_slope_match_the_joint_least_squares() {
    let (xa, ya, sa) = (
        [0.0, 1.0, 2.0, 3.0],
        [1.1, 2.9, 5.2, 6.8],
        [0.3, 0.2, 0.4, 0.3],
    );
    let (xb, yb, sb) = ([0.5, 1.5, 2.5], [6.1, 8.0, 9.9], [0.2, 0.25, 0.2]);
    let a = Points::new(&xa, &ya, &sa).unwrap();
    let b = Points::new(&xb, &yb, &sb).unwrap();
    let sim = Simultaneous::new()
        .channel("A", &a, &line("c_a"), &FitOptions::new())
        .channel("B", &b, &line("c_b"), &FitOptions::new());
    assert_eq!(sim.param_names(), ["c_a", "b", "c_b"]);
    let fit = sim.fit(&FitOptions::new().tolerance(1e-4));
    assert!(fit.valid, "{fit:?}");

    // θ = (c_a, b, c_b): rows (1, x, 0) for A and (0, x, 1) for B.
    let mut m = [[0.0; 3]; 3];
    let mut v = [0.0; 3];
    let rows = xa
        .iter()
        .zip(&ya)
        .zip(&sa)
        .map(|((&x, &y), &s)| ([1.0, x, 0.0], y, s))
        .chain(
            xb.iter()
                .zip(&yb)
                .zip(&sb)
                .map(|((&x, &y), &s)| ([0.0, x, 1.0], y, s)),
        );
    for (r, y, s) in rows {
        let w = 1.0 / (s * s);
        for i in 0..3 {
            v[i] += w * r[i] * y;
            for j in 0..3 {
                m[i][j] += w * r[i] * r[j];
            }
        }
    }
    let theta = solve3(m, v);
    for k in 0..3 {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        let var = solve3(m, e)[k]; // (AᵀWA)⁻¹ diagonal
        assert!(
            (fit.params[k] - theta[k]).abs() < 1e-3 * var.sqrt(),
            "θ{k}: {fit:?}"
        );
        assert!(
            (fit.errors[k] / var.sqrt() - 1.0).abs() < 1e-5,
            "σ{k}: {fit:?}"
        );
    }
    assert_eq!(fit.ndf, 7 - 3);
    // The cost splits over the channels.
    let parts = sim.channel_costs(&fit);
    assert_eq!(parts.iter().map(|p| p.2).collect::<Vec<_>>(), [4, 3]);
    let sum: f64 = parts.iter().map(|p| p.1).sum();
    assert!((sum - fit.chi2).abs() < 1e-9 && fit.chi2 == fit.fval);
    // Levenberg–Marquardt sees the joint residuals.
    let lm = sim.fit(&FitOptions::new().minimizer(Minimizer::LevenbergMarquardt));
    assert!(
        (lm.params[1] - theta[1]).abs() < 0.02 * fit.errors[1],
        "{lm:?}"
    );
}

#[test]
fn sharing_a_parameter_combines_the_channels_precision() {
    // Two independent peak samples measure the same mean: the joint error is
    // close to the inverse-variance combination of the separate ones.
    let data = common::peak_data();
    let model = |tag: &str| {
        let p = |n: &str| format!("{tag}.{n}");
        let names = [p("A"), "mu".to_string(), p("s"), p("c")];
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Model::new(tag, &refs, vec![200.0, 3.0, 0.6, 19.0], |x, q| {
            q[0] * (-0.5 * ((x - q[1]) / q[2]).powi(2)).exp() + q[3]
        })
        .lower_limit(&p("s"), 0.05)
    };
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let one = data.fit_opts(&model("x"), &opts);
    let sim = Simultaneous::new()
        .channel("first", &data, &model("x"), &opts)
        .channel("second", &data, &model("y"), &opts);
    let fit = sim.fit(&opts);
    assert!(fit.valid, "{fit:?}");
    let mu = sim.param_names().iter().position(|n| n == "mu").unwrap();
    // The same sample twice: half the variance, the same value.
    assert!((fit.params[mu] - one.params[1]).abs() < 0.02 * one.errors[1]);
    assert!(
        (fit.errors[mu] * 2f64.sqrt() / one.errors[1] - 1.0).abs() < 0.02,
        "{fit:?}"
    );
}

#[test]
fn binned_and_unbinned_channels_fit_together() {
    // A histogram and a list of events from the same peak share mean and width.
    let hist = common::peak_data();
    let binned = Model::new(
        "h",
        &["A", "mu", "s", "c"],
        vec![200.0, 3.0, 0.6, 19.0],
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + p[3],
    )
    .lower_limit("s", 0.05);
    let shape = Model::new("e", &["mu", "s"], vec![3.0, 0.6], |x, p| {
        (-0.5 * ((x - p[0]) / p[1]).powi(2)).exp()
    });
    let events = Events::new(&[2.4, 3.1, 2.9, 3.6, 3.0, 2.2, 3.3, 2.8, 3.9, 3.2]);
    let sim = Simultaneous::new()
        .channel(
            "hist",
            &hist,
            &binned,
            &FitOptions::new().method(FitMethod::Likelihood),
        )
        .channel(
            "events",
            &events,
            &shape,
            &FitOptions::new()
                .method(FitMethod::UnbinnedLikelihood)
                .range(0.0, 6.0),
        );
    assert_eq!(sim.param_names(), ["A", "mu", "s", "c"]);
    let fit = sim.fit(&FitOptions::new());
    assert!(fit.valid, "{fit:?}");
    assert!((fit.params[1] - 3.0).abs() < 0.2, "{fit:?}");
    assert!(fit.chi2.is_nan(), "an unbinned channel: no χ²");
    assert_eq!(fit.ndf, 100 + 10 - 4);
    // LM has no residuals for the unbinned channel and falls back to MIGRAD.
    let lm = sim.fit(&FitOptions::new().minimizer(Minimizer::LevenbergMarquardt));
    assert!((lm.params[1] - fit.params[1]).abs() < 0.02 * fit.errors[1]);
}

#[test]
fn a_shared_parameter_takes_its_settings_from_the_first_channel() {
    let a = Points::new(&[0.0, 1.0, 2.0], &[0.0, 2.0, 4.0], &[0.1; 3]).unwrap();
    let b = Points::new(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0], &[0.1; 3]).unwrap();
    // The first channel caps the slope at 1.5 and constrains it; the second's
    // own constraint on it is ignored (one constraint, counted once).
    let first = line("c_a").upper_limit("b", 1.5).constrain("b", 1.4, 0.1);
    let second = line("c_b").constrain("b", 9.0, 0.1);
    let sim = Simultaneous::new()
        .channel("A", &a, &first, &FitOptions::new())
        .channel("B", &b, &second, &FitOptions::new());
    let fit = sim.fit(&FitOptions::new());
    assert!(fit.params[1] <= 1.5 + 1e-9, "{fit:?}");
    assert_eq!(fit.ndf, 6 + 1 - 3);
    // Too few points for the parameters: an invalid fit, not a panic.
    let lone = Points::new(&[1.0], &[2.0], &[0.1]).unwrap();
    let thin = Simultaneous::new().channel("A", &lone, &line("c"), &FitOptions::new());
    assert!(!thin.fit(&FitOptions::new()).valid);
}
