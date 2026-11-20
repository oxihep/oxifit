//! Toy studies: the fits' pulls are standard normal when the fit is unbiased
//! and its errors are right.

use oxifit::{FitMethod, FitOptions, Model, ToyStudy, Toys};

/// The pulls of every parameter are within 4 standard errors of mean 0 and
/// width 1.
fn assert_unit_pulls(study: &ToyStudy) {
    for name in &study.param_names {
        let p = study.pull_summary(name).unwrap();
        assert!(p.mean.abs() < 4.0 * p.mean_err, "{name}: pull mean {p:?}");
        assert!(
            (p.width - 1.0).abs() < 4.0 * p.width_err,
            "{name}: pull width {p:?}"
        );
    }
}

fn peak() -> Model {
    Model::new(
        "peak",
        &["A", "mu", "s", "c"],
        vec![120.0, 3.0, 0.8, 10.0],
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + p[3],
    )
    .lower_limit("s", 0.05)
}

#[test]
fn binned_likelihood_toys_have_unit_pulls() {
    let xs: Vec<f64> = (0..60)
        .map(|i| -10.0 + 20.0 * (f64::from(i) + 0.5) / 60.0)
        .collect();
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let study = Toys::binned(&peak(), &xs).seed(7).run(&peak(), &opts, 400);
    assert_eq!(study.fits.len(), 400);
    assert!(study.valid_fraction() > 0.98, "{}", study.valid_fraction());
    assert_eq!(study.truth, peak().params());
    assert_unit_pulls(&study);
}

/// A Gaussian density with a yield, for the extended likelihood.
fn density() -> Model {
    Model::new("n·g", &["n", "mu", "s"], vec![400.0, 1.0, 0.5], |x, p| {
        p[0] / (p[2] * std::f64::consts::TAU.sqrt()) * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp()
    })
    .lower_limit("s", 0.01)
}

#[test]
fn extended_unbinned_toys_have_unit_pulls_and_a_poisson_yield() {
    let opts = FitOptions::new()
        .method(FitMethod::ExtendedUnbinnedLikelihood)
        .range(-2.0, 4.0);
    let study = Toys::unbinned(&density(), (-2.0, 4.0), 0)
        .extended(true)
        .seed(3)
        .run(&density(), &opts, 300);
    assert!(study.valid_fraction() > 0.98);
    assert_unit_pulls(&study);
    // The fitted yield is the event count: a Poisson of mean ∫f ≈ 400.
    let n = study.values("n").unwrap();
    let mean = n.iter().sum::<f64>() / n.len() as f64;
    assert!(
        (mean - 400.0).abs() < 4.0 * (400.0 / n.len() as f64).sqrt(),
        "{mean}"
    );
}

#[test]
fn unbinned_shape_toys_have_unit_pulls() {
    let shape = Model::new("g", &["mu", "s"], vec![1.0, 0.5], |x, p| {
        (-0.5 * ((x - p[0]) / p[1]).powi(2)).exp()
    })
    .lower_limit("s", 0.01);
    let opts = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(-2.0, 4.0);
    let study = Toys::unbinned(&shape, (-2.0, 4.0), 250)
        .seed(11)
        .run(&shape, &opts, 300);
    assert_unit_pulls(&study);
}

#[test]
fn a_study_is_reproducible_from_its_seed() {
    let xs: Vec<f64> = (0..20).map(f64::from).collect();
    let flat = Model::polynomial("flat", 0).with_params(vec![30.0]);
    let opts = FitOptions::new().method(FitMethod::Likelihood);
    let a = Toys::binned(&flat, &xs).seed(9).run(&flat, &opts, 20);
    let b = Toys::binned(&flat, &xs).seed(9).run(&flat, &opts, 20);
    let c = Toys::binned(&flat, &xs).seed(10).run(&flat, &opts, 20);
    assert_eq!(a, b);
    assert_ne!(a.values("p0"), c.values("p0"));
    assert!(a.pulls("nope").is_none() && a.pull_summary("nope").is_none());
}
