//! sWeights and the asymptotically correct covariance of weighted fits:
//! their defining identities, and coverage by toys.

use oxifit::{Events, FitExt, FitMethod, FitOptions, Model, SWeights};

/// A reproducible generator.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

const M_LO: f64 = 5.18;
const M_HI: f64 = 5.38;
const T_HI: f64 = 10.0;

/// Signal (Gaussian mass, decay time τ = 1.5) and background (flat mass,
/// τ = 0.5): `(mass, time)` per event.
fn sample(rng: &mut Rng, ns: usize, nb: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut m, mut t) = (Vec::new(), Vec::new());
    let time = |rng: &mut Rng, tau: f64| loop {
        let v = -tau * rng.uniform().ln();
        if v < T_HI {
            return v;
        }
    };
    while m.len() < ns {
        let x = 5.28 + 0.02 * rng.gauss();
        if (M_LO..M_HI).contains(&x) {
            m.push(x);
            t.push(time(rng, 1.5));
        }
    }
    for _ in 0..nb {
        m.push(M_LO + (M_HI - M_LO) * rng.uniform());
        t.push(time(rng, 0.5));
    }
    (m, t)
}

fn signal_shape() -> Model {
    Model::new("sig", &[], vec![], |m, _| {
        (-0.5 * ((m - 5.28) / 0.02f64).powi(2)).exp()
    })
}

fn background_shape() -> Model {
    Model::new("bkg", &[], vec![], |_, _| 1.0)
}

/// The extended mass fit with the shapes fixed: the yields.
fn mass_fit(masses: &[f64]) -> oxifit::FitResult {
    let norm = 0.02 * std::f64::consts::TAU.sqrt();
    let model = Model::new("mass", &["ns", "nb"], vec![300.0, 1000.0], move |m, p| {
        p[0] * (-0.5 * ((m - 5.28) / 0.02f64).powi(2)).exp() / norm + p[1] / (M_HI - M_LO)
    });
    Events::new(masses).fit_opts(
        &model,
        &FitOptions::new()
            .method(FitMethod::ExtendedUnbinnedLikelihood)
            .range(M_LO, M_HI),
    )
}

/// A weighted unbinned fit of the decay time to `exp(−t/τ)` on [0, 10].
fn lifetime_fit(times: &[f64], weights: &[f64], opts: FitOptions) -> oxifit::FitResult {
    let model =
        Model::new("decay", &["tau"], vec![1.2], |t, p| (-t / p[0]).exp()).limit("tau", 0.1, 10.0);
    Events::weighted(times, weights).unwrap().fit_opts(
        &model,
        &opts.method(FitMethod::UnbinnedLikelihood).range(0.0, T_HI),
    )
}

#[test]
fn sweights_satisfy_their_identities() {
    let mut rng = Rng(5);
    let (m, _) = sample(&mut rng, 400, 1600);
    let fit = mass_fit(&m);
    assert!(fit.valid);
    let events = Events::new(&m);
    let (sig, bkg) = (signal_shape(), background_shape());
    let sw = SWeights::compute(
        &events,
        &[&sig, &bkg],
        &fit.params,
        &FitOptions::new().range(M_LO, M_HI),
    )
    .unwrap();
    // Each species' weights sum to its fitted yield…
    for k in 0..2 {
        let total: f64 = sw.weights(k).iter().sum();
        assert!(
            (total - fit.params[k]).abs() < 1e-3 * fit.errors[k],
            "species {k}: {total}"
        );
    }
    // …each event's weights sum to one…
    for e in 0..m.len() {
        let s = sw.weights(0)[e] + sw.weights(1)[e];
        assert!((s - 1.0).abs() < 1e-3, "event {e}: {s}");
    }
    // …and V is the yields' covariance of the fit with the shapes fixed.
    let cov = fit.covariance.as_ref().unwrap();
    for a in 0..2 {
        for b in 0..2 {
            let rel =
                (sw.covariance()[a][b] - cov[a][b]).abs() / cov[a][a].sqrt() / cov[b][b].sqrt();
            assert!(
                rel < 1e-2,
                "V[{a}][{b}]: {} vs {}",
                sw.covariance()[a][b],
                cov[a][b]
            );
        }
    }
    // A bad request.
    assert!(SWeights::compute(&events, &[&sig], &[1.0, 2.0], &FitOptions::new()).is_err());
}

#[test]
fn asymptotic_errors_reduce_to_hesse_and_ignore_a_constant_weight() {
    let mut rng = Rng(9);
    let (_, t) = sample(&mut rng, 3000, 0);
    let ones = vec![1.0; t.len()];
    let plain = lifetime_fit(&t, &ones, FitOptions::new());
    let asym = lifetime_fit(&t, &ones, FitOptions::new().with_asymptotic_errors(true));
    assert!(
        (asym.errors[0] / plain.errors[0] - 1.0).abs() < 0.03,
        "{asym:?} {plain:?}"
    );
    // Every event weighted 4: the naive error halves, the right one does not.
    let fours = vec![4.0; t.len()];
    let naive = lifetime_fit(&t, &fours, FitOptions::new());
    let right = lifetime_fit(&t, &fours, FitOptions::new().with_asymptotic_errors(true));
    assert!((naive.errors[0] / plain.errors[0] - 0.5).abs() < 0.01);
    assert!(
        (right.errors[0] / plain.errors[0] - 1.0).abs() < 0.03,
        "{}",
        right.errors[0]
    );
    assert!((right.params[0] - plain.params[0]).abs() < 1e-3 * plain.errors[0]);
}

#[test]
fn sweighted_lifetime_errors_cover() {
    // Toys of the whole procedure: mass fit → sWeights → weighted lifetime
    // fit. The asymptotic errors give pulls of unit width; the naive ones
    // (Σw events) are too small.
    let (mut pulls_naive, mut pulls_asym) = (Vec::new(), Vec::new());
    let (sig, bkg) = (signal_shape(), background_shape());
    for toy in 0..150 {
        let mut rng = Rng(1000 + toy);
        let (m, t) = sample(&mut rng, 300, 900);
        let fit = mass_fit(&m);
        if !fit.valid {
            continue;
        }
        let sw = SWeights::compute(
            &Events::new(&m),
            &[&sig, &bkg],
            &fit.params,
            &FitOptions::new().range(M_LO, M_HI),
        )
        .unwrap();
        let w = sw.weights(0);
        let naive = lifetime_fit(&t, w, FitOptions::new());
        let asym = lifetime_fit(&t, w, FitOptions::new().with_asymptotic_errors(true));
        if naive.valid && asym.valid {
            pulls_naive.push((naive.params[0] - 1.5) / naive.errors[0]);
            pulls_asym.push((asym.params[0] - 1.5) / asym.errors[0]);
        }
    }
    let width = |p: &[f64]| {
        let n = p.len() as f64;
        let mean = p.iter().sum::<f64>() / n;
        (p.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };
    let (wn, wa) = (width(&pulls_naive), width(&pulls_asym));
    assert!(pulls_asym.len() > 140);
    assert!((wa - 1.0).abs() < 0.15, "asymptotic pull width {wa}");
    assert!(wn > 1.2, "naive pull width {wn}");
}
