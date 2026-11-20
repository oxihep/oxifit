//! The No-U-Turn Sampler against posteriors known exactly — a correlated
//! Gaussian, a 20-dimensional one, a banana and a bounded Poisson signal —
//! and its gradient paths (analytic and finite-difference) against each other.

use oxifit::{Fitter, FnCost, Nuts, Sampler};

/// A correlated 2-D Gaussian `−2 ln L`: means (1, −2), σ (0.5, 2), ρ = 0.6.
fn gauss2() -> FnCost {
    FnCost::new(&["a", "b"], |p| {
        let (u, v) = ((p[0] - 1.0) / 0.5, (p[1] + 2.0) / 2.0);
        (u * u - 1.2 * u * v + v * v) / (1.0 - 0.36)
    })
}

/// The Monte Carlo error of a mean: σ·√(τ/N).
fn mc_error(chain: &oxifit::Chain, name: &str) -> f64 {
    let n = chain.values(name).unwrap().len() as f64;
    let tau = chain.autocorrelation_time(name).unwrap();
    chain.std(name).unwrap() * (tau / n).sqrt()
}

#[test]
fn gaussian_moments_correlation_and_diagnostics() {
    let cost = gauss2();
    let chain = Nuts::new(Fitter::new(&cost, vec![0.0, 0.0]))
        .seed(3)
        .run(1000, 2000)
        .unwrap();
    assert_eq!(chain.divergences(), 0);
    // At or a little above the 0.8 target: the averaged warm-up step size is
    // slightly conservative (as in Stan).
    let acc = chain.acceptance();
    assert!((0.75..0.97).contains(&acc), "{acc}");
    for (name, mean, sd) in [("a", 1.0, 0.5), ("b", -2.0, 2.0)] {
        let err = mc_error(&chain, name);
        let m = chain.mean(name).unwrap();
        assert!((m - mean).abs() < 5.0 * err, "{name} mean {m} ± {err}");
        assert!(
            (chain.std(name).unwrap() / sd - 1.0).abs() < 0.04,
            "{name} std"
        );
        assert!(chain.r_hat(name).unwrap() < 1.01, "{name} R̂");
        // Nearly independent draws, where the ensemble needs tens of steps.
        assert!(chain.autocorrelation_time(name).unwrap() < 5.0, "{name} τ");
    }
    let c = chain.covariance();
    let rho = c[0][1] / (c[0][0] * c[1][1]).sqrt();
    assert!((rho - 0.6).abs() < 0.03, "rho {rho}");
}

#[test]
fn twenty_dimensions() {
    // Independent N(k, (1 + k/4)²) for k = 0..20: widths over a factor 6,
    // learned by the metric.
    let names: Vec<String> = (0..20).map(|k| format!("x{k}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let cost = FnCost::new(&refs, |p| {
        p.iter()
            .enumerate()
            .map(|(k, x)| ((x - k as f64) / (1.0 + k as f64 / 4.0)).powi(2))
            .sum()
    })
    .with_gradient(|p, g| {
        for (k, (x, gk)) in p.iter().zip(g.iter_mut()).enumerate() {
            let s = 1.0 + k as f64 / 4.0;
            *gk = 2.0 * (x - k as f64) / (s * s);
        }
    });
    let chain = Nuts::new(Fitter::new(&cost, vec![0.0; 20]))
        .seed(1)
        .run(1000, 1000)
        .unwrap();
    assert_eq!(chain.divergences(), 0);
    for (k, name) in names.iter().enumerate() {
        let s = 1.0 + k as f64 / 4.0;
        let err = mc_error(&chain, name);
        assert!(
            (chain.mean(name).unwrap() - k as f64).abs() < 5.0 * err,
            "{name} mean"
        );
        assert!(
            (chain.std(name).unwrap() / s - 1.0).abs() < 0.08,
            "{name} std"
        );
        assert!(chain.r_hat(name).unwrap() < 1.02, "{name} R̂");
    }
}

#[test]
fn banana_quantiles_match_a_grid_integration() {
    let cost = FnCost::new(&["x", "y"], |p| {
        p[0] * p[0] + ((p[1] - p[0] * p[0]) / 0.5).powi(2)
    });
    let chain = Nuts::new(Fitter::new(&cost, vec![0.0, 0.5]))
        .seed(11)
        .run(1000, 4000)
        .unwrap();
    let (nx, ny) = (1200, 3000);
    let (x0, x1, y0, y1) = (-6.0, 6.0, -3.0, 30.0);
    let (hx, hy) = ((x1 - x0) / f64::from(nx), (y1 - y0) / f64::from(ny));
    let marginal: Vec<f64> = (0..ny)
        .map(|j| {
            let y = y0 + hy * (f64::from(j) + 0.5);
            (0..nx)
                .map(|i| {
                    let x = x0 + hx * (f64::from(i) + 0.5);
                    (-0.5 * x * x - 0.5 * ((y - x * x) / 0.5).powi(2)).exp()
                })
                .sum::<f64>()
        })
        .collect();
    let total: f64 = marginal.iter().sum();
    let quantile = |q: f64| -> (f64, f64) {
        let mut acc = 0.0;
        for (j, m) in marginal.iter().enumerate() {
            acc += m / total;
            if acc >= q {
                return (y0 + hy * (j as f64 + 1.0), m / total / hy);
            }
        }
        (y1, 0.0)
    };
    let tau = chain.autocorrelation_time("y").unwrap();
    let n = chain.values("y").unwrap().len() as f64;
    for q in [0.16, 0.5, 0.84, 0.95] {
        let (exact, density) = quantile(q);
        let got = chain.quantile("y", q).unwrap();
        let tol = 5.0 * (q * (1.0 - q) * tau / n).sqrt() / density + hy;
        assert!(
            (got - exact).abs() < tol,
            "q {q}: {got} vs {exact} (± {tol})"
        );
    }
}

#[test]
fn bounded_poisson_signal_upper_limit() {
    // n = 7 on b = 3.2, s ≥ 0: the posterior piles up against the limit,
    // sampled through s = e^u with its Jacobian.
    let (n, b) = (7.0_f64, 3.2);
    let cost = FnCost::new(&["s"], move |p| 2.0 * ((p[0] + b) - n * (p[0] + b).ln()));
    let chain = Nuts::new(Fitter::new(&cost, vec![3.0]).lower_limit("s", 0.0))
        .seed(5)
        .run(1000, 4000)
        .unwrap();
    let h = 1e-4;
    let dens: Vec<f64> = (0..400_000)
        .map(|k| {
            let s = h * (f64::from(k) + 0.5);
            ((s + b).ln() * n - (s + b)).exp()
        })
        .collect();
    let total: f64 = dens.iter().sum();
    let mut acc = 0.0;
    let mut exact = f64::NAN;
    for (k, d) in dens.iter().enumerate() {
        acc += d / total;
        if acc >= 0.95 {
            exact = h * (k as f64 + 1.0);
            break;
        }
    }
    let got = chain.quantile("s", 0.95).unwrap();
    assert!((got - exact).abs() < 0.03 * exact, "{got} vs {exact}");
    assert!(chain.values("s").unwrap().iter().all(|&s| s >= 0.0));
    // A two-sided limit (logit coordinates) as well.
    let chain = Nuts::new(Fitter::new(&cost, vec![3.0]).limit("s", 0.0, 40.0))
        .seed(6)
        .run(1000, 4000)
        .unwrap();
    let got = chain.quantile("s", 0.95).unwrap();
    assert!((got - exact).abs() < 0.03 * exact, "{got} vs {exact}");
}

#[test]
fn analytic_and_numerical_gradients_agree_and_runs_reproduce() {
    let plain = gauss2();
    let with_grad = gauss2().with_gradient(|p, g| {
        let (u, v) = ((p[0] - 1.0) / 0.5, (p[1] + 2.0) / 2.0);
        g[0] = (2.0 * u - 1.2 * v) / 0.64 / 0.5;
        g[1] = (2.0 * v - 1.2 * u) / 0.64 / 2.0;
    });
    let run = |c: &FnCost, seed| {
        Nuts::new(Fitter::new(c, vec![0.0, 0.0]).fix("b").release("b"))
            .seed(seed)
            .chains(2)
            .run(300, 600)
            .unwrap()
    };
    let (a, b) = (run(&plain, 4), run(&with_grad, 4));
    for name in ["a", "b"] {
        let tol = 5.0 * (mc_error(&a, name).powi(2) + mc_error(&b, name).powi(2)).sqrt();
        assert!((a.mean(name).unwrap() - b.mean(name).unwrap()).abs() < tol);
    }
    assert_eq!(run(&with_grad, 4), b);
    assert_ne!(run(&with_grad, 5), b);
    // A fixed parameter stays put; the ensemble sampler reports no divergences.
    let fixed = Nuts::new(Fitter::new(&plain, vec![0.0, 3.0]).fix("b"))
        .run(200, 400)
        .unwrap();
    assert!(fixed.values("b").unwrap().iter().all(|&v| v == 3.0));
    assert!((fixed.mean("a").unwrap() - 1.75).abs() < 0.1);
    let ens = Sampler::new(Fitter::new(&plain, vec![0.0, 0.0]))
        .run(50, 200)
        .unwrap();
    assert_eq!(ens.divergences(), 0);
}
