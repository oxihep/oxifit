//! 2-D and 3-D fits: binned and unbinned, products and generic models.

use oxifit::{
    Events2, Events3, FitError, FitExt, FitMethod, FitOptions, Model, PointNd, Points, Points2,
    Points3, Simultaneous,
};

/// Reproducible standard-normal pairs (Box–Muller on two Weyl sequences).
fn normals(n: usize, seed: f64) -> Vec<(f64, f64)> {
    (1..=n)
        .map(|i| {
            let i = i as f64;
            let u1 = (i * 0.618_033_988_75 + seed).fract().max(1e-12);
            let u2 = (i * 0.754_877_666_25 + seed * 0.5).fract();
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            (r * a.cos(), r * a.sin())
        })
        .collect()
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (
        m,
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n).sqrt(),
    )
}

#[test]
fn a_2d_binned_gaussian_fit_recovers_its_parameters() {
    // A product of a Gaussian (with the amplitude) and a unit normal.
    let truth = Model::product(
        "blob",
        &[
            &Model::gaussian("x").with_params(vec![500.0, 0.5, 1.2]),
            &Model::normal("y").with_params(vec![-0.3, 0.8]),
        ],
    );
    let (mut xs, mut ys, mut vs, mut ss) = (vec![], vec![], vec![], vec![]);
    for i in 0..30 {
        for j in 0..30 {
            let (x, y) = (
                -4.0 + 8.0 * (f64::from(i) + 0.5) / 30.0,
                -3.0 + 6.0 * (f64::from(j) + 0.5) / 30.0,
            );
            let v = truth.eval_at(&[x, y]);
            xs.push(x);
            ys.push(y);
            vs.push(v);
            ss.push(v.sqrt().max(1.0));
        }
    }
    let data = Points2::new(&xs, &ys, &vs, &ss).unwrap();
    let seed = truth.clone().with_params(vec![300.0, 0.2, 1.0, 0.0, 1.0]);
    for opts in [
        FitOptions::new(),
        FitOptions::new().method(FitMethod::Likelihood),
        FitOptions::new()
            .method(FitMethod::Likelihood)
            .with_gradient(false),
    ] {
        let fit = data.fit_opts(&seed, &opts);
        assert!(fit.valid, "{opts:?}: {fit:?}");
        for (got, want) in fit.params.iter().zip(&truth.params) {
            assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "{fit:?}");
        }
    }
}

#[test]
fn a_2d_unbinned_product_fit_is_the_per_variable_mle() {
    // Independent variables: the 2-D MLE is each variable's sample mean and
    // (1/n) width; the product's integral factorizes.
    let pairs = normals(3000, 0.1);
    let x: Vec<f64> = pairs.iter().map(|p| 1.0 + 0.5 * p.0).collect();
    let y: Vec<f64> = pairs.iter().map(|p| -2.0 + 1.5 * p.1).collect();
    let events = Events2::new(&x, &y).unwrap();
    let model = Model::product(
        "xy",
        &[
            &Model::normal("x").with_params(vec![0.8, 0.6]),
            &Model::normal("y").with_params(vec![-1.8, 1.2]),
        ],
    )
    .lower_limit("x.sigma", 0.01)
    .lower_limit("y.sigma", 0.01);
    let opts = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(-3.0, 5.0)
        .range_y(-12.0, 8.0)
        .tolerance(1e-3);
    let fit = events.fit_opts(&model, &opts);
    assert!(fit.valid, "{fit:?}");
    let ((mx, sx), (my, sy)) = (mean_sd(&x), mean_sd(&y));
    for (k, want) in [mx, sx, my, sy].into_iter().enumerate() {
        assert!(
            (fit.params[k] - want).abs() < 0.02 * fit.errors[k],
            "p{k}: {fit:?} vs {want}"
        );
    }
    let n = x.len() as f64;
    assert!(
        (fit.errors[0] / (sx / n.sqrt()) - 1.0).abs() < 0.02,
        "{fit:?}"
    );
}

#[test]
fn a_correlated_2d_unbinned_fit_normalizes_numerically() {
    // A bivariate Gaussian with correlation ρ — not a product, so its
    // normalization is a nested 2-D integral — fitted to correlated events:
    // ρ̂ is close to the sample correlation.
    let rho = 0.6;
    let pairs = normals(1500, 0.3);
    let x: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let y: Vec<f64> = pairs
        .iter()
        .map(|p| rho * p.0 + (1.0 - rho * rho).sqrt() * p.1)
        .collect();
    let events = Events2::new(&x, &y).unwrap();
    let model = Model::new2("bvn", &["rho"], vec![0.2], |x, y, p| {
        let r = p[0];
        (-(x * x - 2.0 * r * x * y + y * y) / (2.0 * (1.0 - r * r))).exp()
    })
    .limit("rho", -0.95, 0.95);
    let opts = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(-6.0, 6.0)
        .range_y(-6.0, 6.0);
    let fit = events.fit_opts(&model, &opts);
    assert!(fit.valid, "{fit:?}");
    let n = x.len() as f64;
    let sample =
        x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / n / (mean_sd(&x).1 * mean_sd(&y).1);
    assert!(
        (fit.params[0] - sample).abs() < 0.03,
        "{} vs {sample}",
        fit.params[0]
    );
    // Its error is about (1 − ρ²)/√n.
    let want = (1.0 - rho * rho) / n.sqrt();
    assert!(
        (fit.errors[0] / want - 1.0).abs() < 0.2,
        "{fit:?} vs {want}"
    );
}

#[test]
fn three_dimensional_fits_work_binned_and_unbinned() {
    let shape = |m: [f64; 3]| {
        Model::product(
            "xyz",
            &[
                &Model::normal("x").with_params(vec![m[0], 1.0]),
                &Model::normal("y").with_params(vec![m[1], 1.0]),
                &Model::normal("z").with_params(vec![m[2], 1.0]),
            ],
        )
        .fix("x.sigma")
        .fix("y.sigma")
        .fix("z.sigma")
    };
    // Unbinned: the MLE of each mean is the sample mean.
    let a = normals(800, 0.2);
    let b = normals(800, 0.7);
    let (x, y, z): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        a.iter().map(|p| 0.5 + p.0).collect(),
        a.iter().map(|p| -0.5 + p.1).collect(),
        b.iter().map(|p| 1.0 + p.0).collect(),
    );
    let events = Events3::new(&x, &y, &z).unwrap();
    let opts = FitOptions::new()
        .method(FitMethod::UnbinnedLikelihood)
        .range(-6.0, 7.0)
        .range_y(-7.0, 6.0)
        .range_z(-6.0, 7.0);
    let fit = events.fit_opts(&shape([0.0, 0.0, 0.0]), &opts);
    assert!(fit.valid, "{fit:?}");
    for (k, v) in [&x, &y, &z].into_iter().enumerate() {
        let i = [0, 2, 4][k];
        assert!(
            (fit.params[i] - mean_sd(v).0).abs() < 0.02 * fit.errors[i],
            "{fit:?}"
        );
    }
    // Binned: a 3-D grid of a plane in x, y, z.
    let mut pts = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let (x, y, z) = (f64::from(i), f64::from(j), f64::from(k));
                pts.push(PointNd::new([x, y, z], 1.0 + 2.0 * x - y + 0.5 * z, 0.1));
            }
        }
    }
    let plane = Model::new3("p", &["a", "b", "c", "d"], vec![0.0; 4], |x, y, z, p| {
        p[0] + p[1] * x + p[2] * y + p[3] * z
    });
    let fit = Points3::from_points(pts).fit(&plane);
    assert!(fit.valid && fit.chi2 < 1e-9, "{fit:?}");
    assert!((fit.params[1] - 2.0).abs() < 1e-6 && (fit.params[3] - 0.5).abs() < 1e-6);
}

#[test]
fn ranges_select_points_in_every_variable() {
    let data = Points2::new(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 5.0, 0.0, 5.0],
        &[1.0, 9.0, 1.0, 9.0],
        &[0.1; 4],
    )
    .unwrap();
    let flat = Model::new2("c", &["c"], vec![0.0], |_, _, p| p[0]);
    let low = data.fit_opts(&flat, &FitOptions::new().range_y(-1.0, 1.0));
    assert!(
        (low.params[0] - 1.0).abs() < 1e-6 && low.ndf == 1,
        "{low:?}"
    );
    let one = data.fit_opts(&flat, &FitOptions::new().range(0.5, 1.5).range_y(4.0, 6.0));
    assert!(
        (one.params[0] - 9.0).abs() < 1e-6 && one.ndf == 0,
        "{one:?}"
    );
}

#[test]
fn dimensions_must_match() {
    let data2 = Points2::new(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0], &[0.1; 2]).unwrap();
    let line = Model::polynomial("l", 1);
    assert!(!data2.fit(&line).valid, "a 1-D model on 2-D data");
    assert_eq!(
        data2.profile(&line, &FitOptions::new(), "p0", (0.0, 1.0), 3),
        Err(FitError::DimensionMismatch {
            expected: 1,
            found: 2
        })
    );
    let plane = Model::new2("p", &["a"], vec![0.0], |_, _, p| p[0]);
    // A sum or a product of the wrong parts.
    assert_eq!(
        (line.clone() + plane.clone()).check(),
        Err(FitError::DimensionMismatch {
            expected: 1,
            found: 2
        })
    );
    assert_eq!(
        Model::product("bad", &[&plane, &line]).check(),
        Err(FitError::DimensionMismatch {
            expected: 1,
            found: 2
        })
    );
    assert!(plane.eval(1.0).is_nan() && plane.eval_at(&[1.0, 2.0]) == 0.0);
}

#[test]
fn a_simultaneous_fit_shares_parameters_across_dimensions() {
    // A 1-D projection and a 2-D map of the same peak share its x position.
    let xs: Vec<f64> = (0..40).map(|i| -4.0 + 0.2 * f64::from(i)).collect();
    let proj: Vec<f64> = xs
        .iter()
        .map(|&x| 100.0 * (-0.5 * ((x - 0.7) / 0.9_f64).powi(2)).exp())
        .collect();
    let one_d = Points::new(
        &xs,
        &proj,
        &proj.iter().map(|v| v.sqrt().max(1.0)).collect::<Vec<_>>(),
    )
    .unwrap();
    let mut grid = Vec::new();
    for &x in &xs {
        for &y in &xs {
            let v =
                50.0 * (-0.5 * ((x - 0.7) / 0.9_f64).powi(2) - 0.5 * (y / 1.1_f64).powi(2)).exp();
            grid.push(PointNd::new([x, y, 0.0], v, v.sqrt().max(1.0)));
        }
    }
    let two_d = Points2::from_points(grid);
    let m1 = Model::new(
        "proj",
        &["a1", "mean", "sx"],
        vec![80.0, 0.0, 1.0],
        |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp(),
    );
    let m2 = Model::new2(
        "map",
        &["a2", "mean", "sx", "sy"],
        vec![40.0, 0.0, 1.0, 1.0],
        |x, y, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2) - 0.5 * (y / p[3]).powi(2)).exp(),
    );
    let sim = Simultaneous::new()
        .channel("1d", &one_d, &m1, &FitOptions::new())
        .channel("2d", &two_d, &m2, &FitOptions::new());
    assert_eq!(sim.param_names(), ["a1", "mean", "sx", "a2", "sy"]);
    let fit = sim.fit(&FitOptions::new());
    assert!(fit.valid, "{fit:?}");
    assert!(
        (fit.params[1] - 0.7).abs() < 1e-4 && (fit.params[4] - 1.1).abs() < 1e-4,
        "{fit:?}"
    );
}
