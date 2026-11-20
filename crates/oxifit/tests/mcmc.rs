//! The ensemble sampler against posteriors known exactly: a correlated
//! Gaussian, a banana (quantiles from a grid integration), and a Poisson
//! signal with a flat prior (a truncated Gamma) — within their Monte Carlo
//! errors from the measured autocorrelation time.

use oxifit::{Fitter, FnCost, Sampler};

/// A correlated 2-D Gaussian `−2 ln L`: means (1, −2), σ (0.5, 2), ρ = 0.6.
fn gauss2() -> FnCost {
    FnCost::new(&["a", "b"], |p| {
        let (u, v) = ((p[0] - 1.0) / 0.5, (p[1] + 2.0) / 2.0);
        (u * u - 1.2 * u * v + v * v) / (1.0 - 0.36)
    })
}

/// The Monte Carlo error of a mean from the chain: σ·√(τ/N).
fn mc_error(chain: &oxifit::Chain, name: &str) -> f64 {
    let n = chain.values(name).unwrap().len() as f64;
    let tau = chain.autocorrelation_time(name).unwrap();
    chain.std(name).unwrap() * (tau / n).sqrt()
}

#[test]
fn gaussian_posterior_moments_and_correlation() {
    let cost = gauss2();
    let chain = Sampler::new(Fitter::new(&cost, vec![0.0, 0.0]))
        .seed(7)
        .run(500, 4000)
        .unwrap();
    assert!(
        chain.acceptance() > 0.3 && chain.acceptance() < 0.9,
        "{}",
        chain.acceptance()
    );
    for (name, mean, sd) in [("a", 1.0, 0.5), ("b", -2.0, 2.0)] {
        let err = mc_error(&chain, name);
        let m = chain.mean(name).unwrap();
        assert!((m - mean).abs() < 5.0 * err, "{name} mean {m} ± {err}");
        assert!(
            (chain.std(name).unwrap() / sd - 1.0).abs() < 0.03,
            "{name} std"
        );
    }
    let c = chain.covariance();
    let rho = c[0][1] / (c[0][0] * c[1][1]).sqrt();
    assert!((rho - 0.6).abs() < 0.02, "rho {rho}");
    // 68.3 % central interval of a: 1 ± 0.5.
    let (lo, hi) = chain.interval("a", 0.6827).unwrap();
    assert!(
        (lo - 0.5).abs() < 0.03 && (hi - 1.5).abs() < 0.03,
        "[{lo}, {hi}]"
    );
    let tau = chain.autocorrelation_time("a").unwrap();
    assert!(tau > 1.0 && tau < 100.0, "tau {tau}");
}

#[test]
fn banana_quantiles_match_a_grid_integration() {
    // p(x, y) ∝ exp(−x²/2 − (y − x²)²/(2·0.5²)): curved and non-Gaussian.
    let cost = FnCost::new(&["x", "y"], |p| {
        p[0] * p[0] + ((p[1] - p[0] * p[0]) / 0.5).powi(2)
    });
    let chain = Sampler::new(Fitter::new(&cost, vec![0.0, 0.5]))
        .walkers(64)
        .seed(11)
        .run(1000, 6000)
        .unwrap();
    // The exact marginal CDF of y on a fine grid.
    let (nx, ny) = (1200, 3000);
    let (x0, x1, y0, y1) = (-6.0, 6.0, -3.0, 30.0);
    let (hx, hy) = ((x1 - x0) / f64::from(nx), (y1 - y0) / ny as f64);
    let mut marginal = vec![0.0; ny];
    for (j, m) in marginal.iter_mut().enumerate() {
        let y = y0 + hy * (j as f64 + 0.5);
        *m = (0..nx)
            .map(|i| {
                let x = x0 + hx * (f64::from(i) + 0.5);
                (-0.5 * x * x - 0.5 * ((y - x * x) / 0.5).powi(2)).exp()
            })
            .sum::<f64>();
    }
    let total: f64 = marginal.iter().sum();
    // The exact quantile and the marginal density there.
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
        // A quantile's Monte Carlo error: √(q(1−q)·τ/N) / p(y_q); five of
        // them, and the grid's bin.
        let tol = 5.0 * (q * (1.0 - q) * tau / n).sqrt() / density + hy;
        assert!(
            (got - exact).abs() < tol,
            "q {q}: {got} vs {exact} (± {tol})"
        );
    }
}

#[test]
fn poisson_signal_posterior_upper_limit() {
    // n = 7 observed on b = 3.2, flat prior on s ≥ 0: p(s) ∝ (s + b)⁷e^{−(s+b)}.
    let (n, b) = (7.0_f64, 3.2);
    let cost = FnCost::new(&["s"], move |p| 2.0 * ((p[0] + b) - n * (p[0] + b).ln()));
    let chain = Sampler::new(Fitter::new(&cost, vec![3.0]).lower_limit("s", 0.0))
        .seed(5)
        .run(300, 5000)
        .unwrap();
    // The exact 95 % upper limit by integrating the posterior on a grid.
    let h = 1e-4;
    let dens: Vec<f64> = (0..400_000)
        .map(|k| {
            let s = h * (f64::from(k) + 0.5);
            ((s + b).ln() * n - (s + b)).exp()
        })
        .collect();
    let total: f64 = dens.iter().sum();
    let mut acc = 0.0;
    let mut exact = 0.0;
    for (k, d) in dens.iter().enumerate() {
        acc += d / total;
        if acc >= 0.95 {
            exact = h * (k as f64 + 1.0);
            break;
        }
    }
    let got = chain.quantile("s", 0.95).unwrap();
    assert!(
        (got - exact).abs() < 0.04 * exact,
        "95 % limit {got} vs {exact}"
    );
    assert!(chain.values("s").unwrap().iter().all(|&s| s >= 0.0));
}

#[test]
fn reproducible_and_fixed_parameters_stay_fixed() {
    let cost = gauss2();
    let run = |seed| {
        Sampler::new(Fitter::new(&cost, vec![0.0, 3.0]).fix("b"))
            .seed(seed)
            .run(50, 200)
            .unwrap()
    };
    let (a, b) = (run(1), run(1));
    assert_eq!(a, b);
    assert_ne!(a, run(2));
    assert!(a.values("b").unwrap().iter().all(|&v| v == 3.0));
    // Conditional on b = 3: a ~ N(1 + 0.6·0.25·5, 0.5·√(1 − 0.36)) = N(1.75, 0.4).
    assert!((a.mean("a").unwrap() - 1.75).abs() < 0.1);
    assert!(a.values("zz").is_none());
}
