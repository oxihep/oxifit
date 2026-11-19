//! oxifit against ROOT 6.40 and iminuit 2.33 on the same data: the binned
//! resonance fit of the fitter comparison (`comparison/`), whose reference
//! values were produced by `comparison/fit_root.py` and `fit_iminuit.py`.
//! A peak `A·exp(−½((x−μ)/σ)²) + exp(c₀ + c₁x)` at the 100 bin centers of
//! `tests/data/peak_hist.txt`, seeded at (300, 5.0, 0.8, 6.5, −0.35).

use oxifit::{FitExt, FitMethod, FitOptions, FitResult, Minimizer, Model, Points};

struct Reference {
    tool: &'static str,
    params: [f64; 5],
    errors: [f64; 5],
    fval: f64,
}

/// ROOT `TH1::Fit` (option "E", for HESSE errors) and iminuit `LeastSquares`.
const CHI2: [Reference; 2] = [
    Reference {
        tool: "ROOT",
        params: [
            334.289674,
            5.18458673,
            0.5946024231,
            6.693504194,
            -0.3974987061,
        ],
        errors: [7.23668, 0.0127531, 0.0119267, 0.0108303, 0.00425878],
        fval: 94.62397278,
    },
    Reference {
        tool: "iminuit",
        params: [
            334.285905,
            5.184613336,
            0.594580062,
            6.693471443,
            -0.3974884967,
        ],
        errors: [7.23611, 0.0127531, 0.011925, 0.0108291, 0.00425801],
        fval: 94.62398964,
    },
];

/// ROOT `TH1::Fit` option "L" and iminuit with the same bin-center Poisson cost.
const LIKELIHOOD: [Reference; 2] = [
    Reference {
        tool: "ROOT",
        params: [
            333.9801805,
            5.185694156,
            0.5962985997,
            6.69246469,
            -0.3952243712,
        ],
        errors: [7.28374, 0.0127176, 0.0119496, 0.0107805, 0.00419255],
        fval: 95.76128499,
    },
    Reference {
        tool: "iminuit",
        params: [
            333.9801805,
            5.185694156,
            0.5962985997,
            6.69246469,
            -0.3952243712,
        ],
        errors: [7.28353, 0.0127176, 0.0119495, 0.0107799, 0.00419239],
        fval: 95.76128499,
    },
];

fn data() -> Points {
    let text = include_str!("data/peak_hist.txt");
    let (x, y): (Vec<f64>, Vec<f64>) = text
        .lines()
        .map(|l| {
            let mut it = l.split_whitespace().map(|v| v.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .unzip();
    let err: Vec<f64> = y.iter().map(|v| v.sqrt()).collect();
    Points::new(&x, &y, &err).unwrap()
}

fn models() -> [(&'static str, Model); 2] {
    let seed = vec![300.0, 5.0, 0.8, 6.5, -0.35];
    [
        (
            "closure",
            Model::new(
                "peak",
                &["A", "mu", "sigma", "c0", "c1"],
                seed.clone(),
                |x, p| p[0] * (-0.5 * ((x - p[1]) / p[2]).powi(2)).exp() + (p[3] + p[4] * x).exp(),
            ),
        ),
        (
            "built-in",
            (Model::gaussian("sig") + Model::exponential("bkg")).with_params(seed),
        ),
    ]
}

fn check(fit: &FitResult, refs: &[Reference], what: &str) {
    assert!(fit.valid, "{what}: {fit:?}");
    for r in refs {
        for k in 0..5 {
            // The same minimum, to well within MIGRAD's precision…
            let pull = (fit.params[k] - r.params[k]).abs() / r.errors[k];
            assert!(pull < 0.01, "{what} vs {}: p{k} off by {pull:.4}σ", r.tool);
            // …and the same HESSE errors.
            let ratio = fit.errors[k] / r.errors[k];
            assert!(
                (ratio - 1.0).abs() < 2e-3,
                "{what} vs {}: σ{k} ratio {ratio}",
                r.tool
            );
        }
        assert!(
            (fit.fval - r.fval).abs() < 1e-3,
            "{what} vs {}: fval {} vs {}",
            r.tool,
            fit.fval,
            r.fval
        );
    }
}

#[test]
fn binned_fits_match_root_and_iminuit() {
    let data = data();
    for (variant, model) in models() {
        for minimizer in [
            Minimizer::Minuit2Combined,
            Minimizer::Minuit2,
            Minimizer::LBfgsB,
        ] {
            let chi2 = data.fit_opts(&model, &FitOptions::new().minimizer(minimizer));
            check(&chi2, &CHI2, &format!("χ² {variant} {minimizer:?}"));
            let opts = FitOptions::new()
                .method(FitMethod::Likelihood)
                .minimizer(minimizer);
            let lik = data.fit_opts(&model, &opts);
            check(
                &lik,
                &LIKELIHOOD,
                &format!("likelihood {variant} {minimizer:?}"),
            );
        }
    }
}
