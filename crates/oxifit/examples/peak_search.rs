//! Find a narrow peak whose position is not known in advance: a bump hunt.
//!
//! Generates a toy diphoton-like mass spectrum: a narrow Gaussian signal at
//! 125 GeV over a steeply falling background in [100, 180) GeV. The fit model,
//! ROOT's `gaus(0)+expo(3)`, is seeded the way a search would be — the peak in
//! the middle of the window, a guessed width — without looking at the data.
//!
//! A local fit (MIGRAD) only finds the minimum nearest its seed, and from here
//! that is not the peak. The global searches (multistart, differential
//! evolution, CMA-ES) look over the whole mass window
//! (the `limit` on `mean` is the search range) before MIGRAD polishes the
//! answer, so they find it, at the price of many more cost evaluations.
//! Run with:
//!
//! ```sh
//! cargo run -p oxifit --example peak_search
//! ```

use oxifit::{
    CmaEs, DifferentialEvolution, FitExt, FitMethod, FitOptions, Minimizer, Model, MultiStart,
    Points,
};

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
    // Truth: 1200 signal events (mass 125, resolution 1.7 GeV) on 30 000
    // background events falling with a 22 GeV scale, in [100, 180) GeV.
    let (lo, hi, nbins) = (100.0, 180.0, 80);
    let (m_true, sigma_true, n_sig, n_bkg) = (125.0, 1.7, 1200, 30_000);
    let mut toy = Toy(7);
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
        fill(toy.falling(22.0, lo, hi));
    }
    let x: Vec<f64> = (0..nbins).map(|i| lo + (i as f64 + 0.5) * width).collect();
    let err: Vec<f64> = counts.iter().map(|&n: &f64| n.sqrt()).collect();
    let data = Points::new(&x, &counts, &err).expect("equal lengths");

    // gaus(0)+expo(3): [0] peak height, [1] mean, [2] sigma, [3]+[4]·m the
    // log-background. Seeded blind: the peak mid-window with a guessed width,
    // the background from the first and last bins.
    let slope = (counts[nbins - 1] / counts[0]).ln() / (x[nbins - 1] - x[0]);
    let model = Model::from_formula("bump", "gaus(0)+expo(3)")
        .expect("valid formula")
        .with_params(vec![50.0, 140.0, 3.0, counts[0].ln() - slope * x[0], slope])
        .limit("p0", 0.0, 1000.0)
        .limit("p1", lo, hi) // the search range for the peak
        .limit("p2", 0.5, 10.0)
        .limit("p3", -20.0, 40.0)
        .limit("p4", -1.0, 0.0);

    println!("truth: mass {m_true:.2} GeV, resolution {sigma_true:.2} GeV\n");
    println!(
        "{:<24} {:>8} {:>16} {:>14} {:>9} {:>6}",
        "minimizer", "valid", "mass [GeV]", "σ [GeV]", "−2lnL", "calls"
    );
    let lik = FitOptions::new().method(FitMethod::Likelihood);
    for (name, minimizer) in [
        ("MIGRAD (local)", Minimizer::Minuit2),
        (
            "multistart, 20 starts",
            Minimizer::MultiStart(MultiStart::new(20)),
        ),
        (
            "differential evolution",
            Minimizer::DifferentialEvolution(DifferentialEvolution::new()),
        ),
        ("CMA-ES", Minimizer::CmaEs(CmaEs::new())),
    ] {
        let fit = data.fit_opts(&model, &lik.minimizer(minimizer));
        println!(
            "{:<24} {:>8} {:>8.2} ± {:<5.2} {:>6.2} ± {:<5.2} {:>9.1} {:>6}",
            name,
            fit.valid,
            fit.params[1],
            fit.errors[1],
            fit.params[2],
            fit.errors[2],
            fit.chi2,
            fit.nfcn
        );
    }
}
