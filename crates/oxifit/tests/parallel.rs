//! The `rayon` feature: a fit gives the same result on any number of threads.
#![cfg(feature = "rayon")]

use oxifit::{Events, FitExt, FitMethod, FitOptions, Model, Point, Points, Toys};

fn on_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn large_fits_do_not_depend_on_the_thread_count() {
    // Enough points for many chunks of the cost sum.
    let xs: Vec<f64> = (0..50_000).map(|i| f64::from(i) * 1e-4).collect();
    // Gaussian draws (Box–Muller on two deterministic uniform sequences).
    let gauss: Vec<f64> = (1..=50_000)
        .map(|i| {
            let u1 = (f64::from(i) * 0.618_033_988_75).fract().max(1e-12);
            let u2 = (f64::from(i) * 0.754_877_666_25).fract();
            2.5 + 0.7 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect();
    let events = Events::new(&gauss);
    let shape = Model::new("g", &["mu", "s"], vec![2.4, 0.8], |x, p| {
        (-0.5 * ((x - p[0]) / p[1]).powi(2)).exp()
    })
    .lower_limit("s", 0.01);
    let unbinned = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(0.0, 5.0);
    let one = on_threads(1, || events.fit_opts(&shape, &unbinned));
    let four = on_threads(4, || events.fit_opts(&shape, &unbinned));
    assert!(one.valid, "{one:?}");
    // Bit-identical on 1 and 4 threads (field by field: an unbinned fit's
    // `chi2` is NaN, which no NaN equals).
    assert_eq!(
        (&one.params, &one.errors, &one.covariance),
        (&four.params, &four.errors, &four.covariance)
    );
    assert_eq!(
        (one.fval.to_bits(), one.edm.to_bits(), one.nfcn),
        (four.fval.to_bits(), four.edm.to_bits(), four.nfcn)
    );

    let pts = Points::from_points(
        xs.iter()
            .map(|&x| Point::new(x, 3.0 + 2.0 * x + (x * 13.0).sin() * 0.1, 0.1))
            .collect(),
    );
    let line = Model::polynomial("l", 1).with_params(vec![0.0, 1.0]);
    let a = on_threads(1, || pts.fit(&line));
    let b = on_threads(4, || pts.fit(&line));
    assert_eq!(a, b);

    // A toy study: the samples are drawn in order, the fits run in parallel.
    let flat = Model::polynomial("flat", 0).with_params(vec![40.0]);
    let grid: Vec<f64> = (0..30).map(f64::from).collect();
    let lik = FitOptions::new().method(FitMethod::Likelihood);
    let s1 = on_threads(1, || {
        Toys::binned(&flat, &grid).seed(4).run(&flat, &lik, 40)
    });
    let s4 = on_threads(4, || {
        Toys::binned(&flat, &grid).seed(4).run(&flat, &lik, 40)
    });
    assert_eq!(s1, s4);
}
