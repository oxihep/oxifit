//! Bayesian and frequentist answers to the same questions, side by side.
//!
//! 1. **A peak's parameters**: the posterior of a Gaussian peak on a
//!    background (flat priors) from the ensemble sampler and from NUTS,
//!    against the maximum likelihood fit with MINOS errors — near-identical
//!    for well-measured parameters, as they should be. NUTS's draws are far
//!    less correlated (a smaller autocorrelation time τ).
//! 2. **A small signal**: 7 events observed on 3.2 expected, the signal's
//!    95 % credible upper limit (flat prior on s ≥ 0) against the CLs limit —
//!    the two classic answers to "how much signal is excluded?".
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p oxifit --example bayesian_fit
//! ```

use oxifit::{
    FitMethod, FitOptions, Fitter, FnCost, HypoTest, Model, ModelCost, Nuts, Points, Sampler,
};

fn main() {
    let start = std::time::Instant::now();

    // ---- 1. A peak --------------------------------------------------------
    let x: Vec<f64> = (0..40).map(|i| 0.25 + 0.5 * f64::from(i)).collect();
    let truth =
        |v: f64| 60.0 * (-0.5 * ((v - 9.0) / 1.2f64).powi(2)).exp() + 30.0 * (-v / 8.0).exp();
    // Deterministic "data": the truth with a fixed, Poisson-sized scatter.
    let counts: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            (truth(v) + ((i as f64) * 2.3).sin() * truth(v).sqrt())
                .round()
                .max(0.0)
        })
        .collect();
    let data = Points::unweighted(&x, &counts).expect("lengths");
    let model = (Model::gaussian("peak") + Model::exponential("bkg"))
        .with_params(vec![50.0, 8.5, 1.5, 3.0, -0.1])
        .limit("peak.sigma", 0.1, 5.0)
        .limit("peak.constant", 0.0, 500.0);
    let cost = ModelCost::new(
        &data,
        &model,
        &FitOptions::new().method(FitMethod::Likelihood),
    );
    let fitter = cost.fitter();
    let fit = fitter.fit(&FitOptions::new().minos(true));
    let chain = Sampler::new(fitter.clone())
        .seed(1)
        .run(500, 3000)
        .expect("sampling");
    let nuts = Nuts::new(fitter.clone())
        .seed(1)
        .run(1000, 2000)
        .expect("sampling");
    println!(
        "1. Gaussian peak on a falling background (binned likelihood)\n   \
         ensemble: {} walkers × 3000 steps, acceptance {:.2}; \
         NUTS: 4 chains × 2000 draws, acceptance {:.2}, {} divergences",
        chain.samples().len() / 3000,
        chain.acceptance(),
        nuts.acceptance(),
        nuts.divergences()
    );
    println!(
        "   parameter        ML fit (MINOS 68 %)             ensemble median [68 %]       τ    \
         NUTS median [68 %]           τ    R̂"
    );
    let minos = fit.minos.clone().unwrap_or_default();
    for (k, name) in model.param_names().iter().enumerate() {
        let (lo, hi) = minos.get(k).copied().unwrap_or((f64::NAN, f64::NAN));
        let (a, b) = chain.interval(name, 0.6827).expect("a parameter");
        let med = chain.quantile(name, 0.5).expect("a parameter");
        let (na, nb) = nuts.interval(name, 0.6827).expect("a parameter");
        let nmed = nuts.quantile(name, 0.5).expect("a parameter");
        println!(
            "   {name:<15} {:>8.3} [{:>8.3}, {:>8.3}]   {med:>8.3} [{a:>8.3}, {b:>8.3}] {:>5.0}   \
             {nmed:>8.3} [{na:>8.3}, {nb:>8.3}] {:>4.1} {:.3}",
            fit.params[k],
            fit.params[k] + lo,
            fit.params[k] + hi,
            chain.autocorrelation_time(name).unwrap_or(f64::NAN),
            nuts.autocorrelation_time(name).unwrap_or(f64::NAN),
            nuts.r_hat(name).unwrap_or(f64::NAN)
        );
    }

    // ---- 2. A small signal ------------------------------------------------
    let (n, b) = (7.0_f64, 3.2);
    let counting =
        Model::new("count", &["s"], vec![1.0], move |_, p| p[0] + b).limit("s", 0.0, 50.0);
    let single = Points::unweighted(&[0.0], &[n]).expect("lengths");
    let cost = ModelCost::new(
        &single,
        &counting,
        &FitOptions::new().method(FitMethod::Likelihood),
    );
    let limit = HypoTest::new(cost.fitter(), "s")
        .expect("s is free")
        .upper_limit(0.95, (0.0, 30.0))
        .expect("limit");
    let toys = HypoTest::new(cost.fitter(), "s")
        .expect("s is free")
        .cls_toys(limit.observed, 4000, 3)
        .expect("toys");
    let posterior = FnCost::new(&["s"], move |p| 2.0 * ((p[0] + b) - n * (p[0] + b).ln()));
    let chain = Sampler::new(Fitter::new(&posterior, vec![3.0]).limit("s", 0.0, 50.0))
        .seed(2)
        .run(300, 10_000)
        .expect("sampling");
    println!("\n2. {n} events observed on b = {b}: 95 % upper limits on the signal s");
    println!(
        "   CLs (asymptotic)             s < {:.2}   (CLs from 4000 toys there: {:.3} ± {:.3})",
        limit.observed, toys.result.cls, toys.cls_error
    );
    println!(
        "   Bayesian, flat prior s ≥ 0   s < {:.2}   (posterior mean {:.2}, median {:.2})",
        chain.quantile("s", 0.95).expect("s"),
        chain.mean("s").expect("s"),
        chain.quantile("s", 0.5).expect("s")
    );
    println!(
        "   (with so few events the asymptotic CLs is approximate; the toys show by how much)"
    );
    println!("\n(total {:.1?})", start.elapsed());
}
