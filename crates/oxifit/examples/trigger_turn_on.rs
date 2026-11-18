//! Fit a trigger efficiency turn-on curve.
//!
//! A single-muon trigger with a 25 GeV threshold fires with an efficiency
//! that rises from 0 to a plateau as the offline muon pT crosses the
//! threshold, smeared by the trigger's coarser pT resolution. The standard
//! parametrisation is an error-function step
//!
//! ```text
//! ε(pT) = plateau · ½ [1 + erf((pT − threshold) / (√2 · resolution))]
//! ```
//!
//! fitted here to binned efficiencies with binomial errors, using a custom
//! `Model` whose function calls oxistats' `erf`. Run with:
//!
//! ```sh
//! cargo run -p oxifit --example trigger_turn_on
//! ```

use oxifit::{FitExt, Model, Points};
use oxistats::erf;

fn turn_on(pt: f64, p: &[f64]) -> f64 {
    p[0] * 0.5 * (1.0 + erf((pt - p[1]) / (std::f64::consts::SQRT_2 * p[2])))
}

/// A small reproducible uniform generator.
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
    let truth = [0.96, 25.0, 3.0]; // plateau, threshold [GeV], resolution [GeV]
    let mut toy = Toy(7);

    // Tag-and-probe style: in each offline-pT bin, count probe muons and how
    // many fired the trigger.
    let (lo, hi, nbins, per_bin) = (10.0, 60.0, 25, 400);
    let width = (hi - lo) / nbins as f64;
    let (mut x, mut eff, mut err) = (Vec::new(), Vec::new(), Vec::new());
    println!(
        "{:>8} {:>7} {:>7} {:>10}",
        "pT [GeV]", "probes", "passed", "efficiency"
    );
    for i in 0..nbins {
        let centre = lo + (i as f64 + 0.5) * width;
        let passed = (0..per_bin)
            .filter(|_| {
                let pt = centre + (toy.uniform() - 0.5) * width;
                toy.uniform() < turn_on(pt, &truth)
            })
            .count();
        let e = passed as f64 / per_bin as f64;
        // Binomial error, kept away from zero at 0 % and 100 %.
        let sigma = (e * (1.0 - e) / per_bin as f64)
            .sqrt()
            .max(0.5 / per_bin as f64);
        if i % 3 == 0 {
            println!("{centre:>8.1} {per_bin:>7} {passed:>7} {e:>10.3}");
        }
        x.push(centre);
        eff.push(e);
        err.push(sigma);
    }

    let data = Points::new(&x, &eff, &err).expect("as many errors as points");
    let model = Model::new(
        "turn-on",
        &["plateau", "threshold", "resolution"],
        // Start inside the plateau's [0, 1] range: on a limit, Minuit's
        // error estimate for that parameter breaks down.
        vec![0.9, 20.0, 5.0],
        turn_on,
    )
    .limit("plateau", 0.0, 1.0)
    .lower_limit("resolution", 0.1);
    let r = data.fit(&model);

    println!(
        "\nfit ({}), χ²/ndf = {:.1}/{}",
        if r.valid {
            "converged"
        } else {
            "NOT converged"
        },
        r.chi2,
        r.ndf
    );
    for (k, name) in ["plateau", "threshold", "resolution"].iter().enumerate() {
        println!(
            "  {name:<10} {:7.3} ± {:.3}   truth {}",
            r.params[k], r.errors[k], truth[k]
        );
    }
    // Where the trigger is fully efficient: 99 % of the plateau, the usual
    // offline pT cut so the analysis avoids the turn-on region.
    let (thr, res) = (r.params[1], r.params[2]);
    let pt99 = thr + 2.326 * res; // erf argument for 99 % of the plateau
    println!("\n99 % of the plateau is reached at pT ≈ {pt99:.1} GeV: a safe offline cut.");
}
