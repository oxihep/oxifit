//! A search for a narrow resonance on a smoothly falling background — the
//! shape of a diphoton or dimuon bump hunt — from the background model to
//! the exclusion limits:
//!
//! 1. **Choose the background function.** Fit exponentials of polynomials of
//!    rising order to the spectrum and keep the lowest order the next one
//!    does not improve significantly (a likelihood-ratio test).
//! 2. **Check its bias** ("spurious signal"): fit signal + background to a
//!    signal-free, high-statistics background shape and see how much signal
//!    the function invents, relative to the statistical error.
//! 3. **Scan the local significance** p₀ across the mass range.
//! 4. **Fit the signal** at the most significant mass, with the mass free,
//!    and profile the signal strength.
//! 5. **Correct for the look-elsewhere effect**: Gross–Vitells, with the
//!    upcrossings counted on background-only pseudo-experiments.
//! 6. **Set CLs upper limits** on the signal strength at each mass, observed
//!    and expected (the Brazil band), and cross-check one with toys.
//!
//! The data are generated here (a fixed seed), so the truth is known. Run
//! with:
//!
//! ```sh
//! cargo run --release -p oxifit --example resonance_search
//! ```

use oxifit::{
    Cost, FitExt, FitMethod, FitOptions, FitResult, Fitter, HypoTest, Model, ModelCost, Points,
    global_significance, upcrossings,
};

/// The mass window, its binning, and the signal resolution (GeV).
const LO: f64 = 100.0;
const HI: f64 = 160.0;
const BINS: usize = 120;
const WIDTH: f64 = (HI - LO) / BINS as f64;
const RESOLUTION: f64 = 1.7;
/// The signal yield `mu = 1` stands for (a "reference cross-section").
const REF_SIGNAL: f64 = 200.0;

/// The truth the data are generated from.
const TRUE_MASS: f64 = 125.0;
const TRUE_SIGNAL: f64 = 260.0;
const TRUE_BACKGROUND: f64 = 40_000.0;
/// The background shape: `exp(a·x + b·x²)` with `x = (m − 100)/60`.
const TRUE_SHAPE: (f64, f64) = (-3.5, 0.9);

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
    /// A draw from `exp(a·x + b·x²)` on `[0, 1)` (`a < 0 < b`): an
    /// exponential by inversion, accepted with probability `exp(b·(x² − 1))`.
    fn background(&mut self, (a, b): (f64, f64)) -> f64 {
        loop {
            let x = (1.0 - self.uniform() * (1.0 - a.exp())).ln() / a;
            if self.uniform() < (b * (x * x - 1.0)).exp() {
                return x;
            }
        }
    }
}

/// `x` for a mass.
fn scaled(m: f64) -> f64 {
    (m - LO) / (HI - LO)
}

/// The bin centers.
fn centers() -> Vec<f64> {
    (0..BINS).map(|i| LO + WIDTH * (i as f64 + 0.5)).collect()
}

/// The background model: counts per bin `exp(c₀ + c₁x + … + c_k x^k)`.
fn background(order: usize) -> Model {
    let names: Vec<String> = (0..=order).map(|k| format!("c{k}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut seed = vec![0.0; order + 1];
    seed[0] = 6.0;
    if order >= 1 {
        seed[1] = -3.0;
    }
    Model::new("background", &refs, seed, poly_exp)
}

/// `exp(c₀ + c₁x + …)` at mass `m`.
fn poly_exp(m: f64, c: &[f64]) -> f64 {
    let x = scaled(m);
    c.iter().rev().fold(0.0, |acc, &ck| acc * x + ck).exp()
}

/// The fitter with the mass fixed at `m`.
fn at_mass<'c>(f: &Fitter<'c>, m: f64) -> Fitter<'c> {
    f.clone().set("mass", m).fix("mass")
}

/// Signal + background: `mu·REF_SIGNAL` events in a Gaussian peak of the
/// detector resolution at `mass`, on the background of `order`.
fn signal_plus_background(order: usize) -> Model {
    let bkg = background(order);
    let mut names = vec!["mu".to_string(), "mass".to_string()];
    names.extend(bkg.param_names.iter().cloned());
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut seed = vec![1.0, TRUE_MASS];
    seed.extend(bkg.params.iter());
    Model::new("s+b", &refs, seed, move |m, p| {
        let peak = (-0.5 * ((m - p[1]) / RESOLUTION).powi(2)).exp()
            / (RESOLUTION * std::f64::consts::TAU.sqrt());
        p[0] * REF_SIGNAL * WIDTH * peak + poly_exp(m, &p[2..])
    })
    .limit("mu", 0.0, 20.0)
    .limit("mass", LO + 5.0, HI - 5.0)
}

fn likelihood() -> FitOptions {
    FitOptions::new().method(FitMethod::Likelihood)
}

fn main() {
    let start = std::time::Instant::now();
    let xs = centers();

    // ---- The data -------------------------------------------------------
    let mut rng = Rng(20_261_119);
    let mut counts = vec![0.0; BINS];
    let mut fill = |m: f64| {
        if (LO..HI).contains(&m) {
            counts[((m - LO) / WIDTH) as usize] += 1.0;
        }
    };
    for _ in 0..TRUE_BACKGROUND as usize {
        let x = rng.background(TRUE_SHAPE);
        fill(LO + x * (HI - LO));
    }
    for _ in 0..TRUE_SIGNAL as usize {
        let m = TRUE_MASS + RESOLUTION * rng.gauss();
        fill(m);
    }
    let data = Points::unweighted(&xs, &counts).expect("same lengths");
    println!(
        "Generated {} background + {} signal events (m = {TRUE_MASS} GeV, σ = {RESOLUTION} GeV) \
         in {BINS} bins over [{LO}, {HI}] GeV.\n",
        TRUE_BACKGROUND, TRUE_SIGNAL
    );

    // ---- 1. The background function -------------------------------------
    // Fit the whole spectrum with the background alone (a small peak barely
    // moves it) and test each extra order by the likelihood ratio,
    // Δ(−2 ln L) ~ χ²₁ if the extra term is not needed.
    println!("1. Background function: exp(polynomial of order k), binned likelihood");
    println!("   k   −2lnL (deviance)  ndf   Δ vs k−1   p(Δ)");
    let mut chosen = None;
    let mut previous: Option<FitResult> = None;
    for order in 1..=4 {
        let fit = data.fit_opts(&background(order), &likelihood());
        let (delta, p) = match &previous {
            Some(prev) => {
                let d = prev.fval - fit.fval;
                (d, oxistats::chi_square_prob(d.max(0.0), 1))
            }
            None => (f64::NAN, f64::NAN),
        };
        println!(
            "   {order}   {:>10.2}       {:>3}   {:>8.2}   {:>6.4}",
            fit.fval, fit.ndf, delta, p
        );
        // The order before the first one that does not improve the fit.
        if chosen.is_none() && p > 0.05 {
            chosen = Some(order - 1);
        }
        previous = Some(fit);
    }
    let order = chosen.unwrap_or(4);
    println!("   → order {order}: the next order does not improve the fit (p > 0.05).\n");

    // ---- 2. Spurious signal ---------------------------------------------
    // A signal-free spectrum with ten times the data's statistics, exactly
    // the true background shape (an Asimov dataset): any signal a function
    // fits there is a bias of the function, to compare with the signal's
    // statistical error on the data.
    println!("2. Spurious signal: S+B fitted to a signal-free, 10× background (Asimov)");
    let truth_norm = {
        let (a, b) = TRUE_SHAPE;
        let n = 10_000;
        let integral: f64 = (0..n)
            .map(|i| {
                let x = (i as f64 + 0.5) / n as f64;
                (a * x + b * x * x).exp() / n as f64
            })
            .sum();
        10.0 * TRUE_BACKGROUND / integral * WIDTH / (HI - LO)
    };
    let asimov_counts: Vec<f64> = xs
        .iter()
        .map(|&m| {
            let x = scaled(m);
            truth_norm * (TRUE_SHAPE.0 * x + TRUE_SHAPE.1 * x * x).exp()
        })
        .collect();
    let asimov = Points::unweighted(&xs, &asimov_counts).expect("same lengths");
    println!("   k   max |spurious μ| / σ(μ) over 110–150 GeV");
    for k in 1..=3 {
        let model = signal_plus_background(k).limit("mu", -20.0, 20.0);
        let worst = (110..=150)
            .step_by(5)
            .map(|m| {
                let fixed = model.clone().fix("mass");
                let mut fixed = fixed;
                fixed.params[1] = f64::from(m);
                let spur = asimov.fit_opts(&fixed, &likelihood());
                // σ(μ) on the data's statistics: √10 larger than on the 10× sample.
                (spur.params[0] / (spur.errors[0] * 10f64.sqrt())).abs()
            })
            .fold(0.0, f64::max);
        let verdict = if worst < 0.2 { "ok (< 0.2)" } else { "biased" };
        println!("   {k}   {worst:>6.3}   {verdict}");
    }
    println!();

    // ---- 3. Local significance scan -------------------------------------
    // Signal + background with the chosen background, seeded from the
    // background-only fit; at each hypothesised mass (held fixed), the
    // discovery test of μ = 0.
    let model = signal_plus_background(order);
    let bkg_fit = data.fit_opts(&background(order), &likelihood());
    let mut seeds = vec![0.5, TRUE_MASS];
    seeds.extend(&bkg_fit.params);
    let cost = ModelCost::new(&data, &model, &likelihood());
    let fitter = cost.fitter().with_params(seeds);
    let masses: Vec<f64> = (0..=80).map(|k| 110.0 + 0.5 * f64::from(k)).collect();
    let scan_q0 = |f: &Fitter<'_>| -> Vec<f64> {
        masses
            .iter()
            .map(|&m| {
                HypoTest::new(at_mass(f, m), "mu")
                    .ok()
                    .and_then(|t| t.discovery(1.0).ok())
                    .map_or(f64::NAN, |d| d.q0)
            })
            .collect()
    };
    let q0 = scan_q0(&fitter);
    let (best, q_max) = q0
        .iter()
        .enumerate()
        .filter(|(_, q)| q.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, &q)| (masses[i], q))
        .expect("a scan point");
    println!("3. Local significance, μ = 0 tested at each mass (asymptotic q₀)");
    println!("   mass (GeV)   local Z");
    for (m, q) in masses.iter().zip(&q0).step_by(8) {
        println!("   {m:>8.1}     {:>5.2}", q.max(0.0).sqrt());
    }
    let expected = HypoTest::new(at_mass(&fitter, best), "mu")
        .expect("POI")
        .discovery(TRUE_SIGNAL / REF_SIGNAL)
        .expect("test");
    println!(
        "   largest: Z_local = {:.2} at {best:.1} GeV (median expected for the true signal: {:.2})\n",
        q_max.sqrt(),
        expected.z_expected
    );

    // ---- 4. The signal fit ----------------------------------------------
    // Mass free, from the scan's best point; MINOS and the profile give the
    // signal strength's (asymmetric) interval.
    let fitter = fitter.set("mass", best);
    let fit = fitter.fit(&FitOptions::new().with_minos(true));
    let (lo, hi) = fit.minos.as_ref().map_or((f64::NAN, f64::NAN), |m| m[0]);
    println!("4. Signal + background fit (mass free)");
    println!(
        "   μ    = {:.3} {:+.3}/{:+.3} (MINOS)   → {:.0} ± {:.0} signal events (truth {:.0})",
        fit.params[0],
        hi,
        lo,
        fit.params[0] * REF_SIGNAL,
        fit.errors[0] * REF_SIGNAL,
        TRUE_SIGNAL
    );
    println!(
        "   mass = {:.2} ± {:.2} GeV (truth {TRUE_MASS})",
        fit.params[1], fit.errors[1]
    );
    let profile = fitter
        .profile(
            "mu",
            (0.0, 3.0 * fit.params[0].max(0.5)),
            61,
            &FitOptions::new(),
        )
        .expect("profile");
    if let Some((a, b)) = profile.interval(1.0) {
        println!("   profile likelihood 68 % interval on μ: [{a:.3}, {b:.3}]\n");
    }

    // ---- 5. Look-elsewhere effect ----------------------------------------
    // Gross–Vitells: count the upcrossings of a low level c₀ in q₀ scans of
    // background-only pseudo-experiments, then extrapolate to q_max.
    let c0 = 0.5;
    let bg_only = fitter
        .clone()
        .set("mu", 0.0)
        .fix("mu")
        .fit(&FitOptions::new());
    let ntoys = 20;
    let mut crossings = 0;
    for t in 0..ntoys {
        let toy = cost.toy(&bg_only.params, 1000 + t).expect("binned toys");
        let toy_fitter = Fitter::new(&*toy, fit.params.clone())
            .limit("mu", 0.0, 20.0)
            .limit("mass", LO + 5.0, HI - 5.0);
        crossings += upcrossings(&scan_q0(&toy_fitter), c0);
    }
    let mean_n = crossings as f64 / ntoys as f64;
    let (p_global, z_global) = global_significance(q_max, c0, mean_n);
    println!("5. Look-elsewhere effect (Gross–Vitells, {ntoys} background-only toy scans)");
    println!("   ⟨N(c₀ = {c0})⟩ = {mean_n:.2} upcrossings");
    println!(
        "   p_local = {:.2e} (Z = {:.2})  →  p_global = {p_global:.2e} (Z = {z_global:.2})\n",
        oxistats::pvalue_from_significance(q_max.sqrt()),
        q_max.sqrt()
    );

    // ---- 6. Upper limits -------------------------------------------------
    println!("6. 95 % CL upper limits on μ (CLs, q̃_μ, asymptotic)");
    println!("   mass    observed   expected   −2σ    −1σ    +1σ    +2σ");
    for m in (110..=150).step_by(4).map(f64::from) {
        let test = HypoTest::new(at_mass(&fitter, m), "mu").expect("POI");
        match test.upper_limit(0.95, (0.0, 20.0)) {
            Ok(l) => println!(
                "   {m:>5.1}   {:>7.3}    {:>7.3}   {:>5.3}  {:>5.3}  {:>5.3}  {:>5.3}",
                l.observed,
                l.expected[2],
                l.expected[0],
                l.expected[1],
                l.expected[3],
                l.expected[4]
            ),
            Err(e) => println!("   {m:>5.1}   {e}"),
        }
    }
    // A toy-based cross-check at one mass away from the peak.
    let m = 140.0;
    let test = HypoTest::new(at_mass(&fitter, m), "mu").expect("POI");
    let limit = test.upper_limit(0.95, (0.0, 20.0)).expect("limit");
    let asym = test.cls(limit.observed).expect("cls");
    let toys = test.cls_toys(limit.observed, 3000, 7).expect("toys");
    println!(
        "   cross-check at {m} GeV, μ = {:.3}: CLs asymptotic {:.4}, from 3000+3000 toys {:.4} ± {:.4}",
        limit.observed, asym.cls, toys.result.cls, toys.cls_error
    );
    println!("\n(total {:.1?})", start.elapsed());
}
