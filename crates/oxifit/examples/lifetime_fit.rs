//! Measure a lifetime: fit the decay-time distribution of B⁰ mesons.
//!
//! Proper decay times follow `exp(−t/τ)`. Reconstruction usually removes
//! the shortest times (a displaced-vertex cut), so the toy keeps only
//! `t > 0.3 ps`, which an exponential fit is insensitive to: the slope does
//! not depend on where the distribution starts. The fit uses ROOT's `expo`
//! (`exp(p0 + p1·t)`) and a binned Poisson likelihood; `τ = −1/p1`. Run with:
//!
//! ```sh
//! cargo run -p oxifit --example lifetime_fit
//! ```

use oxifit::{FitExt, FitMethod, FitOptions, Model, Points};

struct Toy(u64);

impl Toy {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn main() {
    let tau_true = 1.519; // ps, the PDG B⁰ lifetime
    let (t_min, t_max, nbins) = (0.3, 10.0, 50);
    let mut toy = Toy(31);

    // 20 000 decays, with the short-time cut applied.
    let width = (t_max - t_min) / nbins as f64;
    let mut counts = vec![0.0; nbins];
    let mut kept = 0;
    for _ in 0..20_000 {
        let t = -tau_true * toy.uniform().ln();
        if (t_min..t_max).contains(&t) {
            counts[((t - t_min) / width) as usize] += 1.0;
            kept += 1;
        }
    }
    let x: Vec<f64> = (0..nbins)
        .map(|i| t_min + (i as f64 + 0.5) * width)
        .collect();
    let err: Vec<f64> = counts.iter().map(|&n: &f64| n.sqrt()).collect();
    let data = Points::new(&x, &counts, &err).expect("as many errors as points");

    let model = Model::exponential("decay time").with_params(vec![8.0, -0.5]);
    for (label, method) in [
        ("Neyman χ²", FitMethod::Chi2),
        ("Poisson likelihood", FitMethod::Likelihood),
    ] {
        let r = data.fit_opts(&model, &FitOptions::new().method(method));
        let slope = r.params[1];
        let tau = -1.0 / slope;
        let tau_err = r.errors[1] / (slope * slope);
        println!(
            "{label:<19} τ = {tau:.3} ± {tau_err:.3} ps   (truth {tau_true}, pull {:+.2})",
            (tau - tau_true) / tau_err
        );
    }
    println!(
        "\n{kept} decays kept after the t > {t_min} ps cut; the statistical error is about \
         τ/√N = {:.3} ps.",
        tau_true / (kept as f64).sqrt()
    );
    println!("Neyman χ² drops the empty tail bins and biases τ low; the likelihood does not.");
}
