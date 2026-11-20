//! A trigger efficiency by tag and probe, the way a collider experiment
//! measures one from data: Z → μμ events where one muon (the tag) fired the
//! trigger, and the other (the probe) did or did not.
//!
//! 1. **Per probe-pT bin, a simultaneous fit** of the dimuon mass for passing
//!    and failing probes: one Z yield `N` and one efficiency `ε` shared by the
//!    two categories (`ε·N` passing, `(1 − ε)·N` failing), each with its own
//!    background — so background is subtracted by the fit, not by hand.
//! 2. **The turn-on**: an error-function step fitted to the efficiencies.
//! 3. **Binomial or χ²**: on a background-free sample (matched to the true
//!    muons, as in simulation), the binomial likelihood
//!    ([`Efficiency`](oxifit::Efficiency)) against a χ² with binomial errors —
//!    which, at efficiencies near 0 and 1, biases the curve; toys show by how
//!    much.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p oxifit --example tag_and_probe
//! ```

use oxifit::{Efficiency, FitExt, FitMethod, FitOptions, Model, Points, Simultaneous};

const M_LO: f64 = 70.0;
const M_HI: f64 = 110.0;
const BINS: usize = 40;
const Z_MASS: f64 = 91.19;
const Z_WIDTH: f64 = 3.0;
/// The true trigger turn-on: threshold 24 GeV, resolution 2 GeV, plateau 0.94.
const TRUE: [f64; 3] = [24.0, 2.0, 0.94];

fn efficiency(pt: f64, p: &[f64]) -> f64 {
    p[2] * 0.5 * (1.0 + oxistats::erf((pt - p[0]) / (2f64.sqrt() * p[1])))
}

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
}

/// Probe-pT bins (GeV).
const PT_EDGES: [f64; 10] = [14.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 32.0, 40.0, 60.0];

/// The mass histograms of passing and failing probes in one pT bin:
/// `n_z` Z events and `n_bkg` background pairs (a falling continuum, whose
/// probes pass less often).
fn generate(
    rng: &mut Rng,
    pt_lo: f64,
    pt_hi: f64,
    n_z: usize,
    n_bkg: usize,
) -> ([f64; BINS], [f64; BINS]) {
    let (mut pass, mut fail) = ([0.0; BINS], [0.0; BINS]);
    let width = (M_HI - M_LO) / BINS as f64;
    let mut fill = |m: f64, passed: bool| {
        if (M_LO..M_HI).contains(&m) {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "inside the histogram range: a bin index"
            )]
            let b = ((m - M_LO) / width) as usize;
            if passed {
                pass[b] += 1.0;
            } else {
                fail[b] += 1.0;
            }
        }
    };
    for _ in 0..n_z {
        let pt = pt_lo + (pt_hi - pt_lo) * rng.uniform();
        let m = Z_MASS + Z_WIDTH * rng.gauss();
        let passed = rng.uniform() < efficiency(pt, &TRUE);
        fill(m, passed);
    }
    for _ in 0..n_bkg {
        let m = M_LO - 25.0 * (1.0 - rng.uniform() * (1.0 - (-(M_HI - M_LO) / 25.0f64).exp())).ln();
        let pt = pt_lo + (pt_hi - pt_lo) * rng.uniform();
        let passed = rng.uniform() < 0.5 * efficiency(pt, &TRUE);
        fill(m, passed);
    }
    (pass, fail)
}

fn main() {
    let start = std::time::Instant::now();
    let mut rng = Rng(7);
    let width = (M_HI - M_LO) / BINS as f64;
    let centers: Vec<f64> = (0..BINS).map(|b| M_LO + width * (b as f64 + 0.5)).collect();
    let gauss = move |m: f64, mean: f64, sigma: f64| {
        width * (-0.5 * ((m - mean) / sigma).powi(2)).exp() / (sigma * std::f64::consts::TAU.sqrt())
    };

    // ---- 1. Simultaneous pass/fail fits per pT bin -----------------------
    println!("1. Tag and probe: simultaneous pass/fail fits of the Z peak per probe-pT bin");
    println!("   pT bin (GeV)   N(Z)            ε (fit, MINOS)            ε (truth, bin average)");
    let mut eff_x = Vec::new();
    let mut eff_y = Vec::new();
    let mut eff_e = Vec::new();
    for w in PT_EDGES.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (pass, fail) = generate(&mut rng, lo, hi, 3000, 2500);
        let pass_data = Points::unweighted(&centers, &pass).expect("lengths");
        let fail_data = Points::unweighted(&centers, &fail).expect("lengths");
        let pass_model = Model::new(
            "pass",
            &["N", "eff", "mass", "sigma", "bp0", "bp1"],
            vec![3000.0, 0.5, 91.0, 3.0, 3.0, -0.04],
            move |m, p| p[1] * p[0] * gauss(m, p[2], p[3]) + (p[4] + p[5] * (m - M_LO)).exp(),
        )
        .limit("eff", 0.0, 1.0)
        .limit("sigma", 1.0, 6.0);
        let fail_model = Model::new(
            "fail",
            &["N", "eff", "mass", "sigma", "bf0", "bf1"],
            vec![3000.0, 0.5, 91.0, 3.0, 3.0, -0.04],
            move |m, p| {
                (1.0 - p[1]) * p[0] * gauss(m, p[2], p[3]) + (p[4] + p[5] * (m - M_LO)).exp()
            },
        );
        let opts = FitOptions::new().method(FitMethod::Likelihood);
        let fit = Simultaneous::new()
            .channel("pass", &pass_data, &pass_model, &opts)
            .channel("fail", &fail_data, &fail_model, &opts)
            .fit(&opts.with_minos(true));
        // The truth averaged over the bin (uniform pT within it).
        let truth: f64 = (0..200)
            .map(|k| efficiency(lo + (hi - lo) * (f64::from(k) + 0.5) / 200.0, &TRUE))
            .sum::<f64>()
            / 200.0;
        println!(
            "   [{lo:>4.0}, {hi:>4.0})    {:>6.0} ± {:<4.0}   {:.4} {:+.4}/{:+.4}   {truth:.4}",
            fit.params[0],
            fit.errors[0],
            fit.params[1],
            fit.minos.as_ref().map_or(f64::NAN, |m| m[1].1),
            fit.minos.as_ref().map_or(f64::NAN, |m| m[1].0)
        );
        // MINOS: an efficiency near 0 or 1 has a one-sided interval, and its
        // parabolic error can vanish; take the larger side.
        let (m_lo, m_hi) = fit.minos.as_ref().map_or((0.0, 0.0), |m| m[1]);
        eff_x.push(0.5 * (lo + hi));
        eff_y.push(fit.params[1]);
        eff_e.push(fit.errors[1].max(m_lo.abs()).max(m_hi.abs()));
    }

    // ---- 2. The turn-on --------------------------------------------------
    let turn_on = Model::new(
        "turn-on",
        &["threshold", "resolution", "plateau"],
        vec![22.0, 3.0, 0.9],
        efficiency,
    )
    .limit("plateau", 0.0, 1.0)
    .lower_limit("resolution", 0.1);
    let edges: Vec<f64> = PT_EDGES.to_vec();
    let points = Points::new(&eff_x, &eff_y, &eff_e)
        .expect("lengths")
        .with_bin_edges(&edges)
        .expect("edges");
    // Each point is a bin average: compare it with the curve averaged over
    // the bin, not its value at the center.
    let fit = points.fit_opts(&turn_on, &FitOptions::new().with_bin_integration(true));
    println!("\n2. Turn-on fitted to the tag-and-probe efficiencies (χ², bin-averaged)");
    let names = ["threshold (GeV)", "resolution (GeV)", "plateau"];
    for k in 0..3 {
        println!(
            "   {:<17} {:.3} ± {:.3}   (truth {})",
            names[k], fit.params[k], fit.errors[k], TRUE[k]
        );
    }
    println!("   χ²/ndf = {:.1}/{}", fit.chi2, fit.ndf);

    // ---- 3. Binomial against χ² on a matched, background-free sample ------
    // Few probes per pT bin, efficiencies near 0 and 1: where binomial errors
    // vanish and the χ² misbehaves.
    let ntoys = 400;
    let x: Vec<f64> = (0..40).map(|i| 15.0 + 0.5 * f64::from(i) + 0.25).collect();
    let mut pulls_binomial = vec![Vec::new(); 3];
    let mut pulls_chi2 = vec![Vec::new(); 3];
    let (mut limited_binomial, mut limited_chi2) = (0, 0);
    for _ in 0..ntoys {
        let total = vec![25.0; x.len()];
        let passed: Vec<f64> = x
            .iter()
            .map(|&pt| {
                (0..25)
                    .filter(|_| rng.uniform() < efficiency(pt, &TRUE))
                    .count() as f64
            })
            .collect();
        let binomial = Efficiency::new(&x, &passed, &total)
            .expect("lengths")
            .fit(&turn_on, &FitOptions::new());
        // χ² on k/n with the binomial error √(ε(1−ε)/n), which is 0 at
        // k = 0 or n: those points are dropped, as a χ² must.
        let (cx, cy, ce): (Vec<f64>, Vec<f64>, Vec<f64>) = passed
            .iter()
            .zip(&x)
            .filter(|(k, _)| **k > 0.0 && **k < 25.0)
            .map(|(k, &pt)| {
                let e = k / 25.0;
                (pt, e, (e * (1.0 - e) / 25.0).sqrt())
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut a, mut b, mut c), (p, e, s)| {
                    a.push(p);
                    b.push(e);
                    c.push(s);
                    (a, b, c)
                },
            );
        let chi2 = Points::new(&cx, &cy, &ce).expect("lengths").fit(&turn_on);
        // A plateau fitted at its limit of 1 has no parabolic error: such
        // toys are counted, not pulled.
        let at_limit = |f: &oxifit::FitResult| f.params[2] > 1.0 - 1e-4;
        for (fit, pulls, limited) in [
            (&binomial, &mut pulls_binomial, &mut limited_binomial),
            (&chi2, &mut pulls_chi2, &mut limited_chi2),
        ] {
            if !fit.valid {
                continue;
            }
            if at_limit(fit) {
                *limited += 1;
                continue;
            }
            for k in 0..3 {
                pulls[k].push((fit.params[k] - TRUE[k]) / fit.errors[k]);
            }
        }
    }
    let summary = |p: &[f64]| {
        let n = p.len() as f64;
        let mean = p.iter().sum::<f64>() / n;
        let width = (p.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        (mean, width)
    };
    println!(
        "\n3. {ntoys} toys of a matched sample (25 probes per 0.5-GeV bin): pulls (fit − truth)/error"
    );
    println!("   parameter          binomial mean / width      χ² mean / width");
    for k in 0..3 {
        let (bm, bw) = summary(&pulls_binomial[k]);
        let (cm, cw) = summary(&pulls_chi2[k]);
        println!(
            "   {:<17}  {bm:>+6.3} / {bw:.3}            {cm:>+6.3} / {cw:.3}",
            names[k]
        );
    }
    println!(
        "   (plateau at its limit of 1, not pulled: {limited_binomial} binomial, {limited_chi2} χ² toys)"
    );
    println!("   The χ² drops the empty and full bins and weights each bin by an error");
    println!("   estimated from its own count, which pulls the curve and misjudges its errors.");
    println!("\n(total {:.1?})", start.elapsed());
}
