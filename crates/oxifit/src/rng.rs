//! The crate's random numbers: a seeded SplitMix64 generator and the
//! distributions the global searches and toy studies draw from.

/// A small deterministic generator (SplitMix64): global searches and toy
/// studies must be reproducible from their seed, and need nothing stronger.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// The generator for stream `index` of a study seeded with `seed`: each
    /// toy gets its own, so a study does not depend on the thread count.
    pub(crate) fn stream(seed: u64, index: u64) -> Rng {
        let mut mix = Rng(seed ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03));
        Rng(mix.next_u64())
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`n > 0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        // n is a population or dimension count: the modulo bias is negligible.
        usize::try_from(self.next_u64() % n as u64).expect("below n, a usize")
    }

    /// A standard normal draw (Box–Muller).
    pub(crate) fn gauss(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // in (0, 1]: ln is finite
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A binomial draw of `n` trials of probability `p`: counted for up to a
    /// few hundred trials, else from the normal approximation (rounded and
    /// clamped), which is then far below any statistical effect.
    pub(crate) fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n == 0 || p.is_nan() || p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return n;
        }
        if n <= 500 {
            return (0..n).filter(|_| self.uniform() < p).count() as u64;
        }
        let (mean, sd) = (n as f64 * p, (n as f64 * p * (1.0 - p)).sqrt());
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=n"
        )]
        let k = (mean + sd * self.gauss()).round().clamp(0.0, n as f64) as u64;
        k
    }

    /// A Poisson draw of mean `lambda` (`0` for a non-positive or `NaN` mean):
    /// Knuth's product of uniforms below 10, Hörmann's transformed rejection
    /// (PTRS, 1993) above — exact at any mean, in O(1) for large ones.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "PTRS accepts only whole k ≥ 0"
    )]
    pub(crate) fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda.is_nan() || lambda <= 0.0 {
            return 0;
        }
        if lambda < 10.0 {
            let limit = (-lambda).exp();
            let (mut k, mut p) = (0, 1.0);
            loop {
                p *= self.uniform();
                if p <= limit {
                    return k;
                }
                k += 1;
            }
        }
        let slam = lambda.sqrt();
        let loglam = lambda.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.024_83 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.uniform() - 0.5;
            let v = self.uniform();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln()
                <= -lambda + k * loglam - oxistats::gammaln(k + 1.0)
            {
                return k as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn poisson_draws_have_the_right_mean_and_variance() {
        let mut rng = Rng(5);
        for lambda in [0.7, 3.0, 9.9, 10.0, 47.5, 1.0e4] {
            let n = 40_000_u32;
            let draws: Vec<f64> = (0..n).map(|_| rng.poisson(lambda) as f64).collect();
            let mean = draws.iter().sum::<f64>() / f64::from(n);
            let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / f64::from(n - 1);
            // Within 5 standard errors of the mean and of the variance.
            let se_mean = (lambda / f64::from(n)).sqrt();
            let se_var = lambda * (2.0 / f64::from(n)).sqrt() + (lambda / f64::from(n)).sqrt();
            assert!(
                (mean - lambda).abs() < 5.0 * se_mean,
                "λ={lambda}: mean {mean}"
            );
            assert!((var - lambda).abs() < 5.0 * se_var, "λ={lambda}: var {var}");
        }
        assert_eq!(rng.poisson(0.0), 0);
        assert_eq!(rng.poisson(f64::NAN), 0);
    }
}
