//! The built-in shapes' analytic integrals against adaptive quadrature, over
//! ranges that cross every piece (cores, power-law tails, far tails).

use oxifit::Model;

fn quad(m: &Model, lo: f64, hi: f64) -> f64 {
    oxiformula::integrate_adaptive(|x| m.eval(x), lo, hi, 1e-13, 4000).0
}

#[test]
fn analytic_integrals_match_quadrature() {
    let models = [
        Model::normal("n").with_params(vec![0.3, 0.7]),
        Model::gaussian("g").with_params(vec![12.0, -1.0, 0.4]),
        Model::exponential("e").with_params(vec![1.5, -0.8]),
        Model::exponential("flat").with_params(vec![0.2, 0.0]),
        Model::polynomial("p", 3).with_params(vec![1.0, -0.5, 0.2, 0.03]),
        Model::crystal_ball("cb").with_params(vec![3.0, 0.0, 0.5, 1.3, 3.5]),
        Model::crystal_ball("cb_hi").with_params(vec![3.0, 0.0, 0.5, -1.1, 2.2]),
        Model::crystal_ball("cb_n1").with_params(vec![1.0, 0.0, 1.0, 0.8, 1.0]),
        Model::double_crystal_ball("dcb").with_params(vec![2.0, 0.5, 0.6, 1.2, 2.5, 1.8, 6.0]),
        Model::breit_wigner("bw").with_params(vec![5.0, 1.0, 0.3]),
    ];
    let ranges = [
        (-6.0, 6.0),
        (-0.2, 0.9),
        (-5.0, -1.0),
        (0.8, 4.0),
        (2.5, 9.0),
        (-12.0, -6.0),
    ];
    for m in &models {
        for &(lo, hi) in &ranges {
            let exact = m.integral(&[(lo, hi)]).expect("a built-in integral");
            let numeric = quad(m, lo, hi);
            let scale = quad(m, -6.0, 6.0).abs().max(1e-300);
            assert!(
                (exact - numeric).abs() <= 1e-9 * scale + 1e-9 * numeric.abs(),
                "{} on [{lo}, {hi}]: {exact} vs {numeric}",
                m.name
            );
        }
    }
    // Sums and products of built-ins know theirs too.
    let sum = Model::gaussian("s").with_params(vec![5.0, 0.0, 1.0])
        + Model::exponential("b").with_params(vec![0.0, -0.5]);
    let exact = sum.integral(&[(-2.0, 3.0)]).unwrap();
    assert!((exact - quad(&sum, -2.0, 3.0)).abs() < 1e-9 * exact);
    // A closure has none.
    assert!(
        Model::new("c", &[], vec![], |x, _| x)
            .integral(&[(0.0, 1.0)])
            .is_none()
    );
}

#[test]
fn formula_models_carry_exact_gradients() {
    use oxifit::{FitExt, FitOptions, Points};
    let m = Model::from_formula("peak", "[0]*exp(-0.5*((x-[1])/[2])^2) + [3]*exp([4]*x)")
        .unwrap()
        .with_params(vec![50.0, 1.0, 0.4, 20.0, -0.5]);
    assert!(m.has_gradient());
    assert!(m.check_gradient(&[-1.0, 0.0, 0.8, 1.3, 2.5]) < 1e-6);
    // A fit with the formula's gradient equals one without it.
    let x: Vec<f64> = (0..60).map(|i| -2.0 + 0.1 * f64::from(i)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| (m.eval(v) + 3.0 * (7.0 * v).sin()).round())
        .collect();
    let e: Vec<f64> = y.iter().map(|v| v.sqrt().max(1.0)).collect();
    let data = Points::new(&x, &y, &e).unwrap();
    let with = data.fit_opts(&m, &FitOptions::new());
    let without = data.fit_opts(&m, &FitOptions::new().gradient(false));
    assert!(with.valid && without.valid);
    for k in 0..5 {
        assert!((with.params[k] - without.params[k]).abs() < 0.01 * with.errors[k]);
        assert!((with.errors[k] / without.errors[k] - 1.0).abs() < 0.01);
    }
}
