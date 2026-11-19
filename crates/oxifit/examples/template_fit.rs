//! A template fit with systematic uncertainties — the HistFactory recipe of
//! a search: simulated templates in a signal region (SR) and a control
//! region (CR), fitted to data together with their uncertainties.
//!
//! - **Samples**: a signal (strength `mu`), top-quark pairs (normalization
//!   `mu_ttbar` measured in the CR) and W+jets (normalized from simulation).
//! - **Uncertainties**: W+jets cross-section (±10 %, `normsys`), jet energy
//!   scale (shape, `histosys`, on every background), the simulation's finite
//!   statistics per bin (`staterror`), and the luminosity (2 %).
//!
//! The example then goes through what a search reports: pre- and post-fit
//! yields, the signal strength with MINOS, nuisance-parameter pulls and
//! constraints, the impact ranking on `mu`, the observed and expected
//! significance, and the 95 % CL upper limit with its expected band, checked
//! with toys. The
//! data are generated (fixed seed) with `mu = 0.8`. Run with:
//!
//! ```sh
//! cargo run --release -p oxifit --example template_fit
//! ```

use oxifit::{Channel, FitOptions, HypoTest, Sample, TemplateModel};

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
    /// A Poisson draw (inversion; the means here are below a few hundred).
    fn poisson(&mut self, mean: f64) -> f64 {
        if mean > 200.0 {
            // A normal approximation is ample this far out.
            let (u1, u2) = (self.uniform(), self.uniform());
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            return (mean + mean.sqrt() * g).round().max(0.0);
        }
        let (mut k, mut p, limit) = (0.0, 1.0, (-mean).exp());
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }
}

/// Expected counts per bin of a discriminant in [0, 1): `norm` events with a
/// density `∝ exp(slope·x)`.
fn shape(bins: usize, norm: f64, slope: f64) -> Vec<f64> {
    let total = if slope == 0.0 {
        1.0
    } else {
        (slope.exp() - 1.0) / slope
    };
    (0..bins)
        .map(|b| {
            let (a, c) = (b as f64 / bins as f64, (b + 1) as f64 / bins as f64);
            let part = if slope == 0.0 {
                c - a
            } else {
                ((slope * c).exp() - (slope * a).exp()) / slope
            };
            norm * part / total
        })
        .collect()
}

fn main() {
    let start = std::time::Instant::now();
    let (sr_bins, cr_bins) = (8, 6);

    // ---- Templates ("from simulation") -------------------------------------
    let signal = shape(sr_bins, 30.0, 3.0);
    let ttbar_sr = shape(sr_bins, 160.0, -1.5);
    let wjets_sr = shape(sr_bins, 120.0, -2.5);
    let ttbar_cr = shape(cr_bins, 900.0, -0.8);
    let wjets_cr = shape(cr_bins, 150.0, -1.2);
    // Jet energy scale: ±1σ moves the backgrounds toward higher (lower)
    // discriminant values by a few per cent.
    let jes = |nominal: &[f64], s: f64| -> Vec<f64> {
        let n = nominal.len() as f64;
        nominal
            .iter()
            .enumerate()
            .map(|(b, &v)| v * (1.0 + s * 0.06 * (b as f64 / (n - 1.0) - 0.5)))
            .collect()
    };
    // Simulation statistics: each template from a sample ~4× the data.
    let mc_err =
        |nominal: &[f64]| -> Vec<f64> { nominal.iter().map(|v| (v / 4.0).sqrt()).collect() };

    // ---- The data: mu = 0.8, top normalization 1.15, W+jets +0.5σ ----------
    let mut rng = Rng(42);
    let truth = |sig: &[f64], tt: &[f64], wj: &[f64], mu: f64| -> Vec<f64> {
        (0..tt.len())
            .map(|b| mu * sig.get(b).copied().unwrap_or(0.0) + 1.15 * tt[b] + 1.05 * wj[b])
            .collect()
    };
    let sr_data: Vec<f64> = truth(&signal, &ttbar_sr, &wjets_sr, 0.8)
        .into_iter()
        .map(|m| rng.poisson(m))
        .collect();
    let cr_data: Vec<f64> = truth(&[], &ttbar_cr, &wjets_cr, 0.0)
        .into_iter()
        .map(|m| rng.poisson(m))
        .collect();

    // ---- The model ---------------------------------------------------------
    let background = |name: &str, nominal: &[f64], norm: Option<&str>, xsec: bool| {
        let mut s = Sample::new(name, nominal)
            .histo_sys("jes", &jes(nominal, -1.0), &jes(nominal, 1.0))
            .lumi("lumi");
        if let Some(n) = norm {
            s = s.norm_factor(n);
        }
        if xsec {
            s = s.norm_sys("wjets_xsec", 0.9, 1.1);
        }
        s
    };
    let sr_stat = mc_err(
        &ttbar_sr
            .iter()
            .zip(&wjets_sr)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>(),
    );
    let cr_stat = mc_err(
        &ttbar_cr
            .iter()
            .zip(&wjets_cr)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>(),
    );
    let model = TemplateModel::new(vec![
        Channel::new("SR", &sr_data)
            .sample(
                Sample::new("signal", &signal)
                    .norm_factor("mu")
                    .lumi("lumi"),
            )
            .sample(
                background("ttbar", &ttbar_sr, Some("mu_ttbar"), false)
                    .stat_error("mc_SR", &sr_stat),
            )
            .sample(background("wjets", &wjets_sr, None, true)),
        Channel::new("CR", &cr_data)
            .sample(
                background("ttbar", &ttbar_cr, Some("mu_ttbar"), false)
                    .stat_error("mc_CR", &cr_stat),
            )
            .sample(background("wjets", &wjets_cr, None, true)),
    ])
    .and_then(|m| m.lumi_uncertainty(0.02))
    .expect("a valid model");
    println!(
        "Model: {} channels, {} parameters ({})\n",
        model.channels().len(),
        model.param_names().len(),
        model.param_names()[..6].join(", ") + ", …"
    );

    // ---- Pre-fit yields ----------------------------------------------------
    let prefit = TemplateModel::expected(&model, &model.inits());
    println!("Pre-fit yields (summed over bins)");
    println!("   channel   signal   ttbar   wjets    total   observed");
    for (c, ch) in model.channels().iter().enumerate() {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        let samples: Vec<f64> = prefit[c].iter().map(|s| sum(s)).collect();
        let (sig, tt, wj) = if samples.len() == 3 {
            (samples[0], samples[1], samples[2])
        } else {
            (0.0, samples[0], samples[1])
        };
        let obs = if c == 0 { sum(&sr_data) } else { sum(&cr_data) };
        println!(
            "   {:<8} {sig:>7.1} {tt:>7.1} {wj:>7.1} {:>8.1} {obs:>9.0}",
            ch.name(),
            sig + tt + wj
        );
    }

    // ---- The fit -----------------------------------------------------------
    let fitter = model.fitter();
    let fit = fitter.fit(&FitOptions::new().with_minos(true));
    let names = model.param_names();
    let idx = |n: &str| names.iter().position(|m| m == n).expect("a parameter");
    let (i_mu, i_tt) = (idx("mu"), idx("mu_ttbar"));
    let (lo, hi) = fit.minos.as_ref().map_or((f64::NAN, f64::NAN), |m| {
        // MINOS covers the free parameters in order.
        let free: Vec<usize> = (0..names.len()).filter(|&i| fit.errors[i] > 0.0).collect();
        let k = free.iter().position(|&i| i == i_mu).unwrap_or(0);
        m[k]
    });
    println!(
        "\nFit: mu = {:.3} {hi:+.3}/{lo:+.3} (truth 0.8), mu_ttbar = {:.3} ± {:.3} (truth 1.15), \
         χ²/ndf = {:.1}/{}",
        fit.params[i_mu], fit.params[i_tt], fit.errors[i_tt], fit.chi2, fit.ndf
    );

    // ---- Post-fit yields ---------------------------------------------------
    println!("\nPost-fit yields in the SR, per bin (± from the fit covariance)");
    println!("   bin   signal   total ± error      observed");
    let yields = model.yields(&fit);
    let sr = &yields[0];
    for b in 0..sr_bins {
        println!(
            "   {b:>3}  {:>7.1}  {:>7.1} ± {:<6.1}  {:>8.0}",
            sr.samples[0].1[b], sr.total[b], sr.total_error[b], sr.observed[b]
        );
    }

    // ---- Pulls and impacts -------------------------------------------------
    println!("\nNuisance parameters: pull (θ̂ − θ₀)/σ₀ and constraint σ̂/σ₀");
    for (name, pull, constraint) in model.pulls(&fit) {
        if !name.starts_with("mc_") || pull.abs() > 0.5 {
            println!("   {name:<12} {pull:>+6.2}   {constraint:>5.2}");
        }
    }
    println!("   (MC-statistics γ with |pull| < 0.5 not shown)");
    println!("\nImpact on mu (refit with each parameter at θ̂ ± σ̂, and ± σ₀)");
    println!("   rank  parameter      post-fit up/down    pre-fit up/down");
    for (rank, imp) in model
        .impacts(&fitter, &fit, "mu")
        .iter()
        .take(6)
        .enumerate()
    {
        println!(
            "   {:>4}  {:<12}  {:>+7.3} {:>+7.3}    {:>+7.3} {:>+7.3}",
            rank + 1,
            imp.name,
            imp.postfit.0,
            imp.postfit.1,
            imp.prefit.0,
            imp.prefit.1
        );
    }

    // ---- Significance and limit -------------------------------------------
    let test = HypoTest::new(fitter, "mu").expect("mu is free");
    let d = test.discovery(1.0).expect("discovery test");
    println!(
        "\nSignificance of the signal: observed Z = {:.2} (p₀ = {:.2e}), expected for mu = 1: {:.2}",
        d.z, d.p0, d.z_expected
    );
    let limit = test.upper_limit(0.95, (0.0, 10.0)).expect("limit");
    println!(
        "95 % CL upper limit on mu: observed {:.3}, expected {:.3} [−1σ {:.3}, +1σ {:.3}; −2σ {:.3}, +2σ {:.3}]",
        limit.observed,
        limit.expected[2],
        limit.expected[1],
        limit.expected[3],
        limit.expected[0],
        limit.expected[4]
    );
    let toys = test
        .cls_toys(limit.observed, 2000, 1)
        .expect("toy-based CLs");
    println!(
        "Cross-check at mu = {:.3}: CLs asymptotic 0.050, from 2000+2000 toys {:.3} ± {:.3}",
        limit.observed, toys.result.cls, toys.cls_error
    );
    println!("\n(total {:.1?})", start.elapsed());
}
