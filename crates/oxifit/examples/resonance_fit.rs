//! Fit a resonance on top of a falling background: a Z → μμ peak.
//!
//! Generates a toy dimuon mass spectrum (a Gaussian peak at the Z mass over
//! an exponential Drell–Yan-like background) and bins it. The fit follows
//! the usual recipe: fit the background alone in a signal-free sideband,
//! then use it and the peak bin to seed the full fit with ROOT's composite
//! formula `gaus(0)+expo(3)`, done twice — with a Neyman χ² and with a binned
//! Poisson likelihood. MINOS gives asymmetric errors. Run with:
//!
//! ```sh
//! cargo run -p oxifit --example resonance_fit
//! ```

use oxifit::{FitExt, FitMethod, FitOptions, Model, Points};

/// A small reproducible generator: uniform, Gaussian and exponential draws.
struct Toy(u64);

impl Toy {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self, mean: f64, sigma: f64) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        mean + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// `exp(-x / scale)` truncated to `[lo, hi)`, by inversion.
    fn falling(&mut self, scale: f64, lo: f64, hi: f64) -> f64 {
        let (a, b) = ((-lo / scale).exp(), (-hi / scale).exp());
        -scale * (a - self.uniform() * (a - b)).ln()
    }
}

fn main() {
    // Truth: 6000 Z events (mass 91.19, resolution 2.4 GeV) on 4000
    // background events falling with a 25 GeV scale, in [70, 110) GeV.
    let (lo, hi, nbins) = (70.0, 110.0, 40);
    let (m_true, sigma_true, n_sig, n_bkg) = (91.19, 2.4, 6000, 4000);
    let mut toy = Toy(2024);
    let mut counts = vec![0.0; nbins];
    let width = (hi - lo) / nbins as f64;
    let mut fill = |m: f64| {
        if (lo..hi).contains(&m) {
            counts[((m - lo) / width) as usize] += 1.0;
        }
    };
    for _ in 0..n_sig {
        fill(toy.gauss(m_true, sigma_true));
    }
    for _ in 0..n_bkg {
        fill(toy.falling(25.0, lo, hi));
    }

    // Bin centres, counts and √N errors: what a histogram hands a fitter.
    let x: Vec<f64> = (0..nbins).map(|i| lo + (i as f64 + 0.5) * width).collect();
    let err: Vec<f64> = counts.iter().map(|&n: &f64| n.sqrt()).collect();
    let data = Points::new(&x, &counts, &err).expect("as many errors as points");

    // Step 1: the background alone, in the low sideband (70–80 GeV), where
    // the Z contributes almost nothing.
    let sideband = data.fit_opts(
        &Model::exponential("background").with_params(vec![5.0, 0.0]),
        &FitOptions::new()
            .method(FitMethod::Likelihood)
            .range(lo, 80.0),
    );
    let (b0, b1) = (sideband.params[0], sideband.params[1]);
    println!(
        "sideband background: exp({b0:.2} {b1:+.4}·m), slope scale {:.1} GeV\n",
        -1.0 / b1
    );

    // Step 2: seed the peak from the tallest bin, above that background.
    let peak = (0..nbins)
        .max_by(|&a, &b| counts[a].total_cmp(&counts[b]))
        .expect("bins");
    let height = counts[peak] - (b0 + b1 * x[peak]).exp();

    // gaus(0) is [0]·exp(−½((x−[1])/[2])²), expo(3) is exp([3] + [4]·x).
    let model = Model::from_formula("Z peak + background", "gaus(0)+expo(3)")
        .expect("a ROOT formula")
        .with_params(vec![height, x[peak], 2.0, b0, b1])
        .lower_limit("p2", 0.1);

    for (label, method) in [
        ("Neyman χ²", FitMethod::Chi2),
        ("Poisson likelihood", FitMethod::Likelihood),
    ] {
        let opts = FitOptions::new().method(method).with_minos(true);
        let r = data.fit_opts(&model, &opts);
        // The Gaussian's area in events: amplitude × σ√(2π) / bin width.
        let yield_ = r.params[0] * r.params[2] * (2.0 * std::f64::consts::PI).sqrt() / width;
        let yield_err = yield_
            * ((r.errors[0] / r.params[0]).powi(2) + (r.errors[2] / r.params[2]).powi(2)).sqrt();
        println!(
            "{label} fit ({}):",
            if r.valid {
                "converged"
            } else {
                "NOT converged"
            }
        );
        print!("  mass       {:7.3} ± {:.3} GeV", r.params[1], r.errors[1]);
        if let Some(m) = &r.minos {
            print!("   (MINOS {:+.3} / {:+.3})", m[1].0, m[1].1);
        }
        println!("   truth {m_true}");
        println!(
            "  resolution {:7.3} ± {:.3} GeV   truth {sigma_true}",
            r.params[2], r.errors[2]
        );
        println!("  signal     {yield_:7.0} ± {yield_err:.0} events   truth {n_sig}");
        println!(
            "  χ²/ndf     {:.1}/{} = {:.2}, p-value {:.2}\n",
            r.chi2,
            r.ndf,
            r.chi2_per_ndf(),
            r.p_value()
        );
    }
}
