//! Template fits against pyhf 0.8.0 (`tests/data/pyhf_templates.py`): the
//! same two workspaces — normsys + histosys + staterror in one channel, and
//! shapesys + a shared normsys + lumi across a signal and a control region —
//! give the same best fit, CLs and upper limits, and the same errors as the
//! inverse of a direct finite-difference Hessian of pyhf's own `−2 ln L`
//! (pyhf's reported HESSE errors run up to 1 % small on these models; the
//! likelihoods themselves agree to 1e-13).

use oxifit::{
    Channel, Cost, FitOptions, HistoInterp, HypoTest, NormInterp, Sample, TemplateError,
    TemplateModel,
};

fn workspace_a() -> TemplateModel {
    TemplateModel::new(vec![
        Channel::new("SR", &[66.0, 58.0, 35.0])
            .sample(Sample::new("signal", &[4.0, 9.0, 6.0]).norm_factor("mu"))
            .sample(
                Sample::new("background", &[60.0, 45.0, 30.0])
                    .norm_sys("bkg_norm", 0.93, 1.08)
                    .histo_sys("jes", &[57.5, 43.0, 29.0], &[63.0, 47.0, 30.5])
                    .stat_error("mc_stat", &[3.0, 2.5, 2.0]),
            ),
    ])
    .unwrap()
}

fn workspace_b() -> TemplateModel {
    TemplateModel::new(vec![
        Channel::new("SR", &[27.0, 24.0])
            .sample(
                Sample::new("signal", &[3.0, 6.0])
                    .norm_factor("mu")
                    .lumi("lumi"),
            )
            .sample(
                Sample::new("background", &[20.0, 15.0])
                    .shape_sys("unc", &[3.0, 2.0])
                    .norm_sys("xsec", 0.85, 1.1)
                    .lumi("lumi"),
            ),
        Channel::new("CR", &[95.0, 88.0]).sample(
            Sample::new("background", &[100.0, 80.0])
                .norm_sys("xsec", 0.85, 1.1)
                .lumi("lumi"),
        ),
    ])
    .unwrap()
    .lumi_uncertainty(0.02)
    .unwrap()
}

struct Reference {
    names: [&'static str; 6],
    best: [f64; 6],
    errors: [f64; 6],
    cls: f64,
    cls_expected: [f64; 5],
    limit: f64,
    limit_expected: [f64; 5],
}

const A: Reference = Reference {
    names: [
        "jes",
        "mu",
        "bkg_norm",
        "mc_stat[0]",
        "mc_stat[1]",
        "mc_stat[2]",
    ],
    best: [
        0.059_327_246,
        1.194_978_039,
        0.037_674_358,
        1.001_796_287,
        1.004_464_853,
        0.992_515_344,
    ],
    errors: [
        1.003_586_010,
        0.800_362_211,
        0.952_779_695,
        0.047_523_712,
        0.054_293_097,
        0.064_647_241,
    ],
    cls: 0.559_641_947,
    cls_expected: [
        0.025_816_006,
        0.078_062_772,
        0.213_143_232,
        0.479_280_711,
        0.792_924_005,
    ],
    limit: 2.557_577_117,
    limit_expected: [
        0.843_571_014,
        1.135_577_013,
        1.582_848_848,
        2.217_111_145,
        3.001_693_944,
    ],
};

const B: Reference = Reference {
    names: ["lumi", "mu", "xsec", "unc[0]", "unc[1]", ""],
    best: [
        1.000_422_100,
        1.593_415_534,
        0.119_190_544,
        1.025_630_907,
        0.992_649_966,
        0.0,
    ],
    errors: [
        0.019_761_224,
        0.845_245_086,
        0.547_677_325,
        0.134_750_662,
        0.130_739_902,
        0.0,
    ],
    cls: 0.558_190_008,
    cls_expected: [
        0.024_684_183,
        0.075_520_723,
        0.208_495_342,
        0.473_425_389,
        0.788_992_766,
    ],
    limit: 3.091_412_807,
    limit_expected: [
        0.829_176_191,
        1.129_911_218,
        1.605_253_456,
        2.308_998_662,
        3.223_917_401,
    ],
};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-2)
}

fn check(model: &TemplateModel, r: &Reference, lumi_bounds: bool) {
    let mut fitter = model.fitter();
    if lumi_bounds {
        fitter = fitter.limit("lumi", 0.5, 1.5);
    }
    let fit = fitter.fit(&FitOptions::new());
    assert!(fit.valid, "{fit:?}");
    let names = model.param_names();
    for (k, name) in r.names.iter().enumerate().filter(|(_, n)| !n.is_empty()) {
        let i = names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{name}"));
        // Parameters to a hundredth of their error (MIGRAD's precision), errors to 0.5 %.
        assert!(
            (fit.params[i] - r.best[k]).abs() < 1e-2 * r.errors[k],
            "{name}: {} vs {}",
            fit.params[i],
            r.best[k]
        );
        assert!(
            close(fit.errors[i], r.errors[k], 5e-3),
            "{name} error: {} vs {}",
            fit.errors[i],
            r.errors[k]
        );
    }
    let test = HypoTest::new(fitter, "mu").unwrap();
    let cls = test.cls(1.0).unwrap();
    assert!(close(cls.cls, r.cls, 1e-3), "CLs {} vs {}", cls.cls, r.cls);
    for (k, (&got, &want)) in cls.expected.iter().zip(&r.cls_expected).enumerate() {
        assert!(close(got, want, 2e-3), "band {k}: {got} vs {want}");
    }
    let limit = test.upper_limit(0.95, (0.0, 10.0)).unwrap();
    assert!(
        close(limit.observed, r.limit, 1e-3),
        "limit {} vs {}",
        limit.observed,
        r.limit
    );
    for (k, (&got, &want)) in limit.expected.iter().zip(&r.limit_expected).enumerate() {
        assert!(
            close(got, want, 2e-3),
            "expected limit {k}: {got} vs {want}"
        );
    }
}

#[test]
fn normsys_histosys_staterror_match_pyhf() {
    check(&workspace_a(), &A, false);
}

#[test]
fn shapesys_lumi_and_two_channels_match_pyhf() {
    check(&workspace_b(), &B, true);
}

#[test]
fn asimov_returns_its_parameters_and_toys_are_reproducible() {
    let model = workspace_a();
    let truth = [1.5, 0.3, -0.4, 1.02, 0.97, 1.01];
    let names = model.param_names();
    // Reorder to the model's parameter order.
    let order = [
        "mu",
        "bkg_norm",
        "jes",
        "mc_stat[0]",
        "mc_stat[1]",
        "mc_stat[2]",
    ];
    let mut p = vec![0.0; names.len()];
    for (k, n) in order.iter().enumerate() {
        p[names.iter().position(|m| m == n).unwrap()] = truth[k];
    }
    let asimov = model.asimov(&p).unwrap();
    let fit = oxifit::Fitter::new(&*asimov, model.inits())
        .limit("mu", 0.0, 10.0)
        .fit(&FitOptions::new());
    for (i, v) in p.iter().enumerate() {
        assert!(
            (fit.params[i] - v).abs() < 1e-2 * fit.errors[i],
            "{}: {fit:?}",
            names[i]
        );
    }
    let (a, b) = (model.toy(&p, 3).unwrap(), model.toy(&p, 3).unwrap());
    assert_eq!(a.value(&p), b.value(&p));
    assert_ne!(a.value(&p), model.toy(&p, 4).unwrap().value(&p));
}

#[test]
fn pulls_impacts_and_yields() {
    let model = workspace_a();
    let fitter = model.fitter();
    let fit = fitter.fit(&FitOptions::new());
    let pulls = model.pulls(&fit);
    assert_eq!(pulls.len(), 5); // every constrained parameter, not mu
    for (name, pull, constraint) in &pulls {
        assert!(
            pull.abs() < 1.0 && *constraint > 0.5 && *constraint <= 1.01,
            "{name}"
        );
    }
    let impacts = model.impacts(&fitter, &fit, "mu");
    assert_eq!(impacts.len(), 5);
    // Ranked, and each post-fit shift moves mu the opposite way on each side
    // (or not at all).
    for w in impacts.windows(2) {
        assert!(w[0].size() >= w[1].size());
    }
    let top = &impacts[0];
    assert!(top.postfit.0 * top.postfit.1 <= 0.0, "{top:?}");
    let yields = model.yields(&fit);
    assert_eq!(yields.len(), 1);
    let y = &yields[0];
    for b in 0..3 {
        let sum: f64 = y.samples.iter().map(|(_, v)| v[b]).sum();
        assert!((sum - y.total[b]).abs() < 1e-9);
        // The fit describes the data within its errors.
        assert!(
            y.total_error[b] > 0.0
                && (y.total[b] - y.observed[b]).abs() < 3.0 * y.observed[b].sqrt()
        );
    }
}

#[test]
fn interpolation_codes_and_bad_models() {
    // Code 1 / code 0 differ from the defaults inside ±1σ.
    let a = workspace_a();
    let b = workspace_a()
        .interpolation(NormInterp::Exponential, HistoInterp::Linear)
        .unwrap();
    let mut p = a.inits();
    let names = a.param_names();
    p[names.iter().position(|n| n == "bkg_norm").unwrap()] = 0.5;
    p[names.iter().position(|n| n == "jes").unwrap()] = 0.5;
    assert!((a.value(&p) - b.value(&p)).abs() > 1e-6);
    assert!(matches!(
        TemplateModel::new(vec![
            Channel::new("SR", &[1.0, 2.0]).sample(Sample::new("s", &[1.0]))
        ]),
        Err(TemplateError::BinMismatch { .. })
    ));
    assert!(matches!(
        TemplateModel::new(vec![
            Channel::new("SR", &[1.0]).sample(
                Sample::new("s", &[1.0])
                    .norm_factor("x")
                    .norm_sys("x", 0.9, 1.1)
            )
        ]),
        Err(TemplateError::KindConflict(_))
    ));
}

#[cfg(feature = "pyhf")]
#[test]
fn pyhf_json_workspaces_read_into_the_same_models() {
    use oxifit::Workspace;
    for (file, built, r) in [
        ("pyhf_workspace_A.json", workspace_a(), &A),
        ("pyhf_workspace_B.json", workspace_b(), &B),
    ] {
        let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
        let ws = Workspace::from_json(&std::fs::read_to_string(path).unwrap(), None).unwrap();
        assert_eq!(ws.poi(), "mu");
        assert_eq!(ws.model().param_names(), built.param_names());
        let p = built.inits();
        assert_eq!(ws.model().value(&p), built.value(&p));
        // The workspace's own settings (B's lumi bounds) give pyhf's fit.
        let fit = ws.fitter().fit(&FitOptions::new());
        let i = built.param_names().iter().position(|n| n == "mu").unwrap();
        assert!(
            (fit.params[i] - r.best[1]).abs() < 1e-2 * r.errors[1],
            "{file}: {fit:?}"
        );
    }
    assert!(Workspace::from_json("{}", None).is_err());
    assert!(Workspace::from_json("not json", None).is_err());
}
