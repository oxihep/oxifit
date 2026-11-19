//! Shared fixtures for the integration tests.
// Each test binary uses only some of these.
#![allow(dead_code)]

use oxifit::{Model, Point, Points};

/// A Gaussian peak on a flat background, 100 bins on [−10, 10], with the peak
/// at x ≈ 3. Seeded with the mean at −8.64, this exact sample once made the
/// minuit2 0.5 numerical gradient panic (`f64::clamp` with min > max).
pub const PEAK: [f64; 100] = [
    7., 19., 21., 23., 18., 12., 14., 17., 16., 19., 14., 26., 16., 34., 17., 18., 15., 25., 19.,
    23., 27., 22., 13., 26., 13., 22., 27., 16., 21., 19., 15., 28., 20., 19., 7., 16., 24., 13.,
    22., 20., 21., 24., 27., 25., 19., 22., 18., 16., 24., 18., 18., 18., 24., 17., 17., 19., 29.,
    24., 45., 52., 86., 112., 163., 205., 236., 222., 185., 165., 131., 87., 62., 40., 18., 30.,
    25., 17., 14., 21., 20., 13., 20., 24., 16., 17., 18., 14., 18., 20., 29., 17., 25., 21., 29.,
    18., 14., 17., 20., 21., 21., 15.,
];

/// [`PEAK`] as points with Poisson errors `σ = √y`.
pub fn peak_data() -> Points {
    let pts = PEAK
        .iter()
        .enumerate()
        .map(|(i, &y)| Point::new(-10.0 + 20.0 * (i as f64 + 0.5) / 100.0, y, y.sqrt()))
        .collect();
    Points::from_points(pts)
}

/// A Gaussian on a constant, seeded with the mean at `mu` (the truth is ≈ 3).
pub fn peak_model(mu: f64) -> Model {
    Model::new(
        "peak",
        &["A", "mu", "s", "c"],
        vec![100.0, mu, 1.0, 10.0],
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + p[3],
    )
    .lower_limit("s", 0.05)
}
