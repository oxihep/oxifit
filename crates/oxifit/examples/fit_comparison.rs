//! The oxifit side of the fitter comparison in `comparison/` (ROOT, iminuit,
//! scipy, zfit): the same datasets, models, starting values and costs, each fit
//! timed over repeats. Prints one JSON object per fit.
//!
//! ```sh
//! python comparison/make_data.py comparison/data
//! cargo run --release -p oxifit --example fit_comparison -- comparison/data
//! cargo run --release -p oxifit --features rayon --example fit_comparison -- comparison/data
//! ```

use std::time::Instant;

use oxifit::{Events, FitExt, FitMethod, FitOptions, FitResult, Model, Points};

/// Starting values, the same for every tool (see `comparison/README.md`).
const BINNED_SEED: [f64; 5] = [300.0, 5.0, 0.8, 6.5, -0.35];
const UNBINNED_SEED: [f64; 5] = [8_000.0, 5.0, 0.8, 80_000.0, 0.3];
const LO: f64 = 0.0;
const HI: f64 = 10.0;

fn read_columns(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// Fit `repeats` times (each from the seed) and report the median time.
fn timed(repeats: usize, mut fit: impl FnMut() -> FitResult) -> (FitResult, f64) {
    let mut times = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let t = Instant::now();
        let r = fit();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        last = Some(r);
    }
    times.sort_by(f64::total_cmp);
    (last.unwrap(), times[times.len() / 2])
}

fn emit(tool: &str, problem: &str, r: &FitResult, ms: f64, repeats: usize) {
    let list = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.10e}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "{{\"tool\": \"{tool}\", \"problem\": \"{problem}\", \"params\": [{}], \"errors\": [{}], \
         \"fval\": {:.10e}, \"nfcn\": {}, \"valid\": {}, \"time_ms\": {ms:.4}, \"repeats\": {repeats}}}",
        list(&r.params),
        list(&r.errors),
        r.fval,
        r.nfcn,
        r.valid
    );
}

/// The built-in parametrization `(A, μ, σ, c0, c1)` as the comparison's
/// `(Nₛ, μ, σ, N_b, λ)`, errors propagated through the covariance.
fn to_yields(r: &FitResult) -> FitResult {
    let map = |p: &[f64]| -> Vec<f64> {
        let lambda = -p[4];
        vec![
            p[0] * p[2] * std::f64::consts::TAU.sqrt(),
            p[1],
            p[2],
            p[3].exp() * (1.0 - (-lambda * HI).exp()) / lambda,
            lambda,
        ]
    };
    let mut out = r.clone();
    out.params = map(&r.params);
    if let Some(cov) = &r.covariance {
        // A numerical Jacobian of the map (every parameter is free here).
        let jac: Vec<Vec<f64>> = (0..5)
            .map(|k| {
                let h = 1e-6 * r.params[k].abs().max(1e-3);
                let (mut up, mut down) = (r.params.clone(), r.params.clone());
                up[k] += h;
                down[k] -= h;
                let (a, b) = (map(&up), map(&down));
                (0..5).map(|i| (a[i] - b[i]) / (2.0 * h)).collect()
            })
            .collect();
        out.errors = (0..5)
            .map(|i| {
                let mut v = 0.0;
                for a in 0..5 {
                    for b in 0..5 {
                        v += jac[a][i] * cov[a][b] * jac[b][i];
                    }
                }
                v.sqrt()
            })
            .collect();
    }
    out
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "comparison/data".into());
    let tag = if cfg!(feature = "rayon") {
        "+rayon"
    } else {
        ""
    };

    // --- Binned: a Gaussian peak on exp(c0 + c1·x), at the bin centers. ---
    let rows = read_columns(&format!("{dir}/peak_hist.txt"));
    let x: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    let y: Vec<f64> = rows.iter().map(|r| r[1]).collect();
    let err: Vec<f64> = y.iter().map(|v| v.sqrt()).collect();
    let hist = Points::new(&x, &y, &err).unwrap();
    // As a user would first write it: a closure (finite-difference gradients).
    let closure = Model::new(
        "peak",
        &["A", "mu", "sigma", "c0", "c1"],
        BINNED_SEED.to_vec(),
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + (p[3] + p[4] * x).exp(),
    );
    // The same function from built-in shapes, which carry analytic gradients.
    let builtin =
        (Model::gaussian("sig") + Model::exponential("bkg")).with_params(BINNED_SEED.to_vec());
    for (problem, method) in [
        ("binned_chi2", FitMethod::Chi2),
        ("binned_likelihood", FitMethod::Likelihood),
    ] {
        let opts = FitOptions::new().method(method);
        for (variant, model) in [("closure", &closure), ("builtin", &builtin)] {
            let (r, ms) = timed(200, || hist.fit_opts(model, &opts));
            emit(&format!("oxifit-{variant}{tag}"), problem, &r, ms, 200);
        }
    }

    // --- Unbinned extended: Ns·N(μ, σ) + Nb·λe^{-λx}/(1 − e^{-10λ}) on [0, 10]. ---
    let density = |x: f64, p: &[f64]| {
        let g = (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() / (p[2] * std::f64::consts::TAU.sqrt());
        let e = p[4] * (-p[4] * x).exp() / (1.0 - (-p[4] * HI).exp());
        p[0] * g + p[3] * e
    };
    let names = ["Ns", "mu", "sigma", "Nb", "lambda"];
    for (file, scale, repeats) in [("events_100k", 1.0, 10), ("events_1M", 10.0, 3)] {
        let xs: Vec<f64> = read_columns(&format!("{dir}/{file}.txt"))
            .into_iter()
            .map(|r| r[0])
            .collect();
        let events = Events::new(&xs);
        let mut seed = UNBINNED_SEED.to_vec();
        seed[0] *= scale;
        seed[3] *= scale;
        let plain = Model::new("mix", &names, seed.clone(), density)
            .lower_limit("sigma", 0.01)
            .lower_limit("lambda", 0.01);
        let with_grad = plain.clone().with_gradient(|x, p, g| {
            let s = p[2];
            let u = (x - p[1]) / s;
            let n = (-0.5 * u * u).exp() / (s * std::f64::consts::TAU.sqrt());
            let l = p[4];
            let norm = 1.0 - (-l * HI).exp();
            let e = l * (-l * x).exp() / norm;
            g[0] = n;
            g[1] = p[0] * n * u / s;
            g[2] = p[0] * n * (u * u - 1.0) / s;
            g[3] = e;
            // d/dλ [λ e^{-λx} / (1 − e^{-λH})]
            let dnorm = HI * (-l * HI).exp();
            g[4] = p[3] * ((-l * x).exp() * (1.0 - l * x) / norm - e * dnorm / norm);
        });
        // The same density from built-in shapes — a Gaussian of amplitude
        // A = Ns/(σ√2π) and exp(c0 + c1·x) — which carry analytic gradients
        // and integrals: no numerical normalization at all.
        let amplitude = seed[0] / (seed[2] * std::f64::consts::TAU.sqrt());
        let c0 = (seed[3] * seed[4] / (1.0 - (-seed[4] * HI).exp())).ln();
        let builtin = (Model::gaussian("sig") + Model::exponential("bkg"))
            .with_params(vec![amplitude, seed[1], seed[2], c0, -seed[4]])
            .lower_limit("sig.sigma", 0.01);
        let opts = FitOptions::new()
            .method(FitMethod::ExtendedUnbinnedLikelihood)
            .range(LO, HI);
        for (variant, model) in [
            ("closure", &plain),
            ("gradient", &with_grad),
            ("builtin", &builtin),
        ] {
            let (mut r, ms) = timed(repeats, || events.fit_opts(model, &opts));
            if variant == "builtin" {
                r = to_yields(&r);
            }
            emit(
                &format!("oxifit-{variant}{tag}"),
                &format!("unbinned_{file}"),
                &r,
                ms,
                repeats,
            );
        }
    }
}
