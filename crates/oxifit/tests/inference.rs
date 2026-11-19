//! Hypothesis tests against pyhf 0.8.0 (`tests/data/pyhf_reference.py`): a
//! three-bin counting experiment, a signal strength `mu` and a background
//! normalization with a 10 % Gaussian constraint (pyhf's `lumi` modifier);
//! then toys against the asymptotic formulae, and unbinned data.

use oxifit::{
    Events, FitError, FitMethod, FitOptions, Fitter, FnCost, HypoTest, Model, ModelCost, Points,
    TestStatistic,
};

const SIG: [f64; 3] = [5.0, 10.0, 3.0];
const BKG: [f64; 3] = [50.0, 52.0, 40.0];

fn cost(counts: [f64; 3]) -> ModelCost {
    let model = Model::new("sb", &["mu", "lumi"], vec![1.0, 1.0], |x, p| {
        let i = x as usize;
        p[0] * SIG[i] + p[1] * BKG[i]
    });
    let data = Points::unweighted(&[0.0, 1.0, 2.0], &counts).unwrap();
    ModelCost::new(
        &data,
        &model,
        &FitOptions::new().method(FitMethod::Likelihood),
    )
}

fn fitter(c: &ModelCost) -> Fitter<'_> {
    c.fitter()
        .limit("mu", 0.0, 10.0)
        .limit("lumi", 0.5, 1.5)
        .constrain("lumi", 1.0, 0.1)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-3)
}

#[test]
fn asymptotic_cls_matches_pyhf() {
    let c = cost([55.0, 62.0, 44.0]);
    let test = HypoTest::new(fitter(&c), "mu").unwrap();
    // (mu, CLs, CLs+b, CLb, expected −2σ…+2σ)
    let reference = [
        (
            0.5,
            0.703_926_391_7,
            0.5,
            0.710_301_539_9,
            [
                0.233_871_116,
                0.378_559_845,
                0.579_396_920,
                0.798_846_558,
                0.947_428_227,
            ],
        ),
        (
            1.0,
            0.578_863_042_6,
            0.5,
            0.863_762_173_8,
            [
                0.042_909_627,
                0.113_327_539,
                0.272_475_652,
                0.548_185_062,
                0.835_647_493,
            ],
        ),
        (
            2.0,
            0.167_426_180_2,
            0.144_405_708_2,
            0.862_503_749_4,
            [
                0.000_723_182,
                0.005_103_861,
                0.031_368_219,
                0.148_077_845,
                0.449_677_758,
            ],
        ),
    ];
    for (mu, cls, clsb, clb, expected) in reference {
        let r = test.cls(mu).unwrap();
        assert!(close(r.cls, cls, 1e-4), "mu {mu}: CLs {} vs {cls}", r.cls);
        assert!(
            close(r.clsb, clsb, 1e-4),
            "mu {mu}: CLs+b {} vs {clsb}",
            r.clsb
        );
        assert!(close(r.clb, clb, 1e-4), "mu {mu}: CLb {} vs {clb}", r.clb);
        for (k, (&got, &want)) in r.expected.iter().zip(&expected).enumerate() {
            assert!(close(got, want, 1e-3), "mu {mu} band {k}: {got} vs {want}");
        }
    }
}

#[test]
fn upper_limit_matches_pyhf() {
    let c = cost([55.0, 62.0, 44.0]);
    let test = HypoTest::new(fitter(&c), "mu").unwrap();
    let limit = test.upper_limit(0.95, (0.0, 10.0)).unwrap();
    assert!(
        close(limit.observed, 2.634_056_264, 1e-4),
        "{}",
        limit.observed
    );
    let expected = [
        0.957_620_066,
        1.294_228_753,
        1.814_776_291,
        2.560_962_022,
        3.491_567_422,
    ];
    for (k, (&got, &want)) in limit.expected.iter().zip(&expected).enumerate() {
        assert!(close(got, want, 1e-3), "band {k}: {got} vs {want}");
    }
    // Not bracketed: CLs never reaches 0.05 below mu = 1.
    assert_eq!(
        test.upper_limit(0.95, (0.0, 1.0)),
        Err(FitError::NoCrossing)
    );
}

#[test]
fn discovery_matches_pyhf() {
    let c = cost([62.0, 75.0, 47.0]);
    let test = HypoTest::new(fitter(&c), "mu").unwrap();
    let d = test.discovery(1.0).unwrap();
    assert!(close(d.p0, 0.006_583_290_507, 1e-3), "{}", d.p0);
    assert!((d.z - d.q0.sqrt()).abs() < 1e-12);
    // pyhf reports the expected p-value; the significance is its Φ⁻¹(1 − p).
    let z_exp = oxistats::significance_from_pvalue(0.138_915_402_571);
    assert!(
        close(d.z_expected, z_exp, 1e-3),
        "{} vs {z_exp}",
        d.z_expected
    );
}

#[test]
fn qmu_differs_from_qmu_tilde_only_below_zero() {
    // With the POI bounded at zero both statistics agree: μ̂ ≥ 0.
    let c = cost([55.0, 62.0, 44.0]);
    let tilde = HypoTest::new(fitter(&c), "mu").unwrap().cls(2.0).unwrap();
    let plain = HypoTest::new(fitter(&c), "mu")
        .unwrap()
        .statistic(TestStatistic::QMu)
        .cls(2.0)
        .unwrap();
    assert!(close(tilde.q, plain.q, 1e-9));
    // CLs+b of q̃ beyond q_A is smaller (its second regime).
    assert!(plain.cls >= tilde.cls - 1e-12);
}

#[test]
fn toys_agree_with_the_asymptotic_formulae_at_large_counts() {
    // Large counts (100× the background, no signal): the asymptotic regime,
    // where toys must reproduce it. σ(μ̂) ≈ 0.1, so μ = 0.15 is excluded at
    // about 1.5σ.
    let big = {
        let model = Model::new("sb", &["mu", "lumi"], vec![1.0, 1.0], |x, p| {
            let i = x as usize;
            100.0 * (p[0] * SIG[i] + p[1] * BKG[i])
        });
        let data = Points::unweighted(&[0.0, 1.0, 2.0], &[5_020.0, 5_190.0, 4_010.0]).unwrap();
        ModelCost::new(
            &data,
            &model,
            &FitOptions::new().method(FitMethod::Likelihood),
        )
    };
    let test = HypoTest::new(
        big.fitter()
            .limit("mu", 0.0, 10.0)
            .limit("lumi", 0.5, 1.5)
            .constrain("lumi", 1.0, 0.01),
        "mu",
    )
    .unwrap();
    let mu = 0.15;
    let asym = test.cls(mu).unwrap();
    let toys = test.cls_toys(mu, 1500, 7).unwrap();
    assert!(toys.q_sb.len() > 1400 && toys.q_b.len() > 1400);
    let pull = (toys.result.cls - asym.cls) / toys.cls_error;
    assert!(asym.cls > 0.05 && asym.cls < 0.6, "{asym:?}");
    assert!(
        pull.abs() < 4.0,
        "toys {} ± {} vs asymptotic {}",
        toys.result.cls,
        toys.cls_error,
        asym.cls
    );
    // Same seed, same result.
    let again = test.cls_toys(mu, 200, 7).unwrap();
    let again2 = test.cls_toys(mu, 200, 7).unwrap();
    assert_eq!(again.q_sb, again2.q_sb);
    assert_eq!(again.q_b, again2.q_b);
    assert_eq!(again.result.cls, again2.result.cls);
    // Background-only toys: q₀ follows ½δ(0) + ½χ²₁, so P(q₀ ≥ 1) ≈ 0.159.
    let d = test.discovery_toys(1500, 11).unwrap();
    let frac = d.q0_toys.iter().filter(|&&q| q >= 1.0).count() as f64 / d.q0_toys.len() as f64;
    assert!((frac - 0.1587).abs() < 0.03, "{frac}");
}

#[test]
fn unbinned_asimov_returns_the_generating_parameters() {
    // An extended unbinned Gaussian peak on a flat background.
    let xs: Vec<f64> = (0..3000)
        .map(|i| {
            let u = (f64::from(i) + 0.5) / 3000.0;
            10.0 * u
        })
        .collect();
    let model = Model::new(
        "m",
        &["ns", "nb", "mean"],
        vec![50.0, 3000.0, 5.0],
        |x, p| {
            p[0] * (-0.5 * ((x - p[2]) / 0.3f64).powi(2)).exp()
                / (0.3 * (std::f64::consts::TAU).sqrt())
                + p[1] / 10.0
        },
    )
    .lower_limit("ns", 0.0);
    let opts = FitOptions::new()
        .method(FitMethod::ExtendedUnbinnedLikelihood)
        .range(0.0, 10.0);
    let events = Events::new(&xs);
    let c = ModelCost::new(&events, &model, &opts);
    use oxifit::Cost;
    let truth = [120.0, 2900.0, 4.0];
    let asimov = c.asimov(&truth).unwrap();
    let fit = Fitter::new(&*asimov, vec![100.0, 3000.0, 4.2])
        .lower_limit("ns", 0.0)
        .fit(&FitOptions::new());
    assert!(fit.valid);
    for k in 0..3 {
        assert!(
            (fit.params[k] - truth[k]).abs() < 0.01 * fit.errors[k],
            "p{k}: {fit:?}"
        );
    }
    // The expected discovery significance of 120 signal events: √q₀ on the
    // Asimov dataset generated at the conditional fit to the data at ns =
    // 120 — here computed exactly, q₀ = 2∫(s + b)·ln((s + b)/b̂) with the
    // background refitted to the whole yield.
    let test = HypoTest::new(c.fitter(), "ns").unwrap();
    let d = test.discovery(120.0).unwrap();
    let g = test.fit_at(120.0).params;
    let n = 200_000;
    let flat = (g[0] + g[1]) / 10.0;
    let q0: f64 = (0..n)
        .map(|i| {
            let x = 10.0 * (f64::from(i) + 0.5) / f64::from(n);
            let s = g[0] * (-0.5 * ((x - g[2]) / 0.3f64).powi(2)).exp()
                / (0.3 * (std::f64::consts::TAU).sqrt());
            let b = g[1] / 10.0;
            2.0 * (s + b) * ((s + b) / flat).ln() * 10.0 / f64::from(n)
        })
        .sum();
    assert!(
        (d.z_expected - q0.sqrt()).abs() < 1e-3 * q0.sqrt(),
        "{} vs {}",
        d.z_expected,
        q0.sqrt()
    );
}

#[test]
fn errors_for_bad_requests() {
    let c = cost([55.0, 62.0, 44.0]);
    assert!(matches!(
        HypoTest::new(fitter(&c), "nope"),
        Err(FitError::UnknownParameter(_))
    ));
    assert!(matches!(
        HypoTest::new(fitter(&c).fix("mu"), "mu"),
        Err(FitError::FixedParameter(_))
    ));
    // A closure cost has no Asimov dataset or toys.
    let f = FnCost::new(&["mu"], |p| (p[0] - 1.0).powi(2));
    let test = HypoTest::new(Fitter::new(&f, vec![0.0]), "mu").unwrap();
    assert_eq!(test.cls(2.0), Err(FitError::NotGenerative));
    assert!(matches!(
        test.cls_toys(2.0, 10, 1),
        Err(FitError::NotGenerative)
    ));
}
