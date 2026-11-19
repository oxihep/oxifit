//! The known minuit2 panic is caught and kept off stderr; any other panic in a
//! fit (a bug in a model closure) still reaches the program's panic hook.
//! (Its own test binary: panic hooks are process-wide.)

mod common;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use oxifit::{FitExt, FitMethod, FitOptions, Minimizer, Model, Points};

static SEEN: AtomicUsize = AtomicUsize::new(0);
static MESSAGES: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[test]
fn only_the_known_minuit2_panic_is_silenced() {
    // The program's own hook, installed before any fit.
    std::panic::set_hook(Box::new(|info| {
        SEEN.fetch_add(1, Ordering::SeqCst);
        MESSAGES
            .lock()
            .unwrap()
            .push(info.payload_as_str().unwrap_or("").to_string());
    }));

    // minuit2 0.5 panics on this dataset from this seed: caught, not reported.
    let fit = common::peak_data().fit_opts(
        &common::peak_model(-8.64),
        &FitOptions::new()
            .method(FitMethod::Likelihood)
            .minimizer(Minimizer::Minuit2),
    );
    assert!(!fit.valid);
    assert_eq!(
        SEEN.load(Ordering::SeqCst),
        0,
        "{:?}",
        MESSAGES.lock().unwrap()
    );

    // A model closure that panics: the fit is still invalid, but the panic
    // reaches the hook — a bug in user code is not hidden.
    let data = Points::new(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[0.1; 3]).unwrap();
    let buggy = Model::new("buggy", &["a"], vec![1.0], |_x, _p| {
        panic!("bug in the model")
    });
    assert!(!data.fit(&buggy).valid);
    assert!(SEEN.load(Ordering::SeqCst) >= 1);
    assert!(
        MESSAGES
            .lock()
            .unwrap()
            .iter()
            .any(|m| m.contains("bug in the model"))
    );

    // Outside a fit, the program's hook is in charge again.
    let _ = std::panic::catch_unwind(|| panic!("min > max, or either was NaN"));
    assert!(
        MESSAGES
            .lock()
            .unwrap()
            .iter()
            .filter(|m| m.contains("min > max"))
            .count()
            == 1
    );
}
