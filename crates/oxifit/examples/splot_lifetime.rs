//! A B⁰ lifetime from sWeights: separate signal from background in the
//! invariant mass, then measure the signal's decay-time distribution alone,
//! with no model of the background's decay times.
//!
//! 1. An extended unbinned fit of the mass (a Gaussian peak on a linear
//!    background) gives the signal and background yields.
//! 2. sWeights (sPlot) from that fit weight each event by how signal-like its
//!    mass is; their sums are the yields.
//! 3. A weighted unbinned fit of the decay time with the signal's model
//!    alone gives the lifetime — and three error estimates: the naive one
//!    (as if the Σw were unweighted events), RooFit's SumW2, and the
//!    asymptotically correct one (Langenbruch, EPJC 82 (2022) 393).
//! 4. Toys of the whole procedure show which errors are right: the pulls'
//!    width.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p oxifit --example splot_lifetime
//! ```

use oxifit::{Events, FitExt, FitMethod, FitOptions, FitResult, Model, SWeights};

const M_LO: f64 = 5.18;
const M_HI: f64 = 5.38;
const M_B: f64 = 5.2797;
const RESOLUTION: f64 = 0.018;
/// Decay times (ps) in [0, 12): the B⁰ lifetime and the background's.
const T_HI: f64 = 12.0;
const TAU_B: f64 = 1.519;
const TAU_BKG: f64 = 0.6;
const N_SIG: usize = 1_500;
const N_BKG: usize = 4_000;

/// A small reproducible generator.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn decay(&mut self, tau: f64) -> f64 {
        loop {
            let t = -tau * self.uniform().ln();
            if t < T_HI {
                return t;
            }
        }
    }
    /// A falling linear background in mass, by accept–reject.
    fn background_mass(&mut self) -> f64 {
        loop {
            let m = M_LO + (M_HI - M_LO) * self.uniform();
            if self.uniform() < 1.0 - 1.5 * (m - M_LO) {
                return m;
            }
        }
    }
}

/// A sample: `(mass, decay time)` per event.
fn sample(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed);
    let (mut m, mut t) = (Vec::new(), Vec::new());
    while m.len() < N_SIG {
        let x = M_B + RESOLUTION * rng.gauss();
        if (M_LO..M_HI).contains(&x) {
            m.push(x);
            t.push(rng.decay(TAU_B));
        }
    }
    for _ in 0..N_BKG {
        m.push(rng.background_mass());
        t.push(rng.decay(TAU_BKG));
    }
    (m, t)
}

fn gauss(m: f64, mean: f64, sigma: f64) -> f64 {
    (-0.5 * ((m - mean) / sigma).powi(2)).exp() / (sigma * std::f64::consts::TAU.sqrt())
}

/// The linear background density on the mass range, slope `s` (unit area).
fn linear(m: f64, s: f64) -> f64 {
    let w = M_HI - M_LO;
    (1.0 + s * (m - 0.5 * (M_LO + M_HI))) / w
}

/// Step 1: the extended mass fit — yields, mean, resolution, slope.
fn mass_fit(masses: &[f64]) -> FitResult {
    let model = Model::new(
        "mass",
        &["n_sig", "n_bkg", "mean", "sigma", "slope"],
        vec![1000.0, 4000.0, 5.28, 0.02, -3.0],
        |m, p| p[0] * gauss(m, p[2], p[3]) + p[1] * linear(m, p[4]),
    )
    .limit("sigma", 0.005, 0.05)
    .limit("slope", -9.9, 9.9);
    Events::new(masses).fit_opts(
        &model,
        &FitOptions::new()
            .method(FitMethod::ExtendedUnbinnedLikelihood)
            .range(M_LO, M_HI),
    )
}

/// Step 2: sWeights from the mass fit's shapes and yields.
fn sweights(masses: &[f64], fit: &FitResult) -> SWeights {
    let (mean, sigma, slope) = (fit.params[2], fit.params[3], fit.params[4]);
    let signal = Model::new("sig", &[], vec![], move |m, _| gauss(m, mean, sigma));
    let background = Model::new("bkg", &[], vec![], move |m, _| linear(m, slope));
    SWeights::compute(
        &Events::new(masses),
        &[&signal, &background],
        &fit.params[..2],
        &FitOptions::new().range(M_LO, M_HI),
    )
    .expect("sWeights")
}

/// Step 3: the weighted lifetime fit, `exp(−t/τ)` normalized on [0, 12).
fn lifetime_fit(times: &[f64], weights: &[f64], opts: FitOptions) -> FitResult {
    let model =
        Model::new("decay", &["tau"], vec![1.3], |t, p| (-t / p[0]).exp()).limit("tau", 0.2, 6.0);
    Events::weighted(times, weights)
        .expect("same lengths")
        .fit_opts(
            &model,
            &opts.method(FitMethod::UnbinnedLikelihood).range(0.0, T_HI),
        )
}

fn main() {
    let start = std::time::Instant::now();
    let (m, t) = sample(2026);
    let fit = mass_fit(&m);
    println!("1. Mass fit ({} candidates)", m.len());
    println!(
        "   signal {:.0} ± {:.0} (truth {N_SIG}), background {:.0} ± {:.0} (truth {N_BKG}), \
         mean {:.4} GeV, resolution {:.4} GeV",
        fit.params[0], fit.errors[0], fit.params[1], fit.errors[1], fit.params[2], fit.params[3]
    );

    let sw = sweights(&m, &fit);
    let w = sw.weights(0);
    let negative = w.iter().filter(|&&x| x < 0.0).count();
    println!("\n2. sWeights");
    println!(
        "   signal weights sum to {:.1} (the yield), range [{:.2}, {:.2}], {negative} negative",
        w.iter().sum::<f64>(),
        w.iter().copied().fold(f64::INFINITY, f64::min),
        w.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\n3. Signal lifetime from the sWeighted decay times (truth {TAU_B} ps)");
    let variants = [
        ("naive (Σw events)", FitOptions::new()),
        ("SumW2", FitOptions::new().sumw2(true)),
        ("asymptotic", FitOptions::new().asymptotic_errors(true)),
    ];
    for (label, opts) in variants {
        let f = lifetime_fit(&t, w, opts);
        println!(
            "   τ = {:.4} ± {:.4} ps   [{label}]",
            f.params[0], f.errors[0]
        );
    }
    // For scale: the fit to the true signal events alone.
    let truth_only = lifetime_fit(&t[..N_SIG], &vec![1.0; N_SIG], FitOptions::new());
    println!(
        "   τ = {:.4} ± {:.4} ps   [the {N_SIG} true signal events, no background]",
        truth_only.params[0], truth_only.errors[0]
    );

    // ---- 4. Coverage -------------------------------------------------------
    let ntoys = 300;
    let mut pulls = [Vec::new(), Vec::new(), Vec::new()];
    for toy in 0..ntoys {
        let (m, t) = sample(10_000 + toy);
        let fit = mass_fit(&m);
        if !fit.valid {
            continue;
        }
        let sw = sweights(&m, &fit);
        for (k, (_, opts)) in variants.iter().enumerate() {
            let f = lifetime_fit(&t, sw.weights(0), *opts);
            if f.valid && f.errors[0] > 0.0 {
                pulls[k].push((f.params[0] - TAU_B) / f.errors[0]);
            }
        }
    }
    println!("\n4. {ntoys} toys of the whole procedure: pulls (τ̂ − τ)/σ̂");
    println!("   errors              pull mean      pull width");
    for ((label, _), p) in variants.iter().zip(&pulls) {
        let n = p.len() as f64;
        let mean = p.iter().sum::<f64>() / n;
        let width = (p.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        println!(
            "   {label:<18}  {mean:+.3} ± {:.3}   {width:.3} ± {:.3}",
            width / n.sqrt(),
            width / (2.0 * (n - 1.0)).sqrt()
        );
    }
    println!("   (a width of 1 means the errors are right)");
    println!("\n(total {:.1?})", start.elapsed());
}
