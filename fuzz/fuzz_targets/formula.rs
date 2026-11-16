//! Parse arbitrary text as a formula; whatever parses must evaluate without
//! panicking, and its ROOT spelling must parse again.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxifit_formula::Formula;

fuzz_target!(|data: &[u8]| {
    let Ok(source) = std::str::from_utf8(data) else {
        return;
    };
    // Deeply nested input is the parser's stack-depth case; keep inputs to a
    // size a real formula has.
    if source.len() > 4096 {
        return;
    }
    let Ok(f) = Formula::parse(source) else {
        return;
    };
    let vars = [0.5, -1.25, 3.0, 1e-3];
    let params: Vec<f64> = (0..f.npar()).map(|i| 0.1 + i as f64).collect();
    let _ = f.eval(&vars[..f.ndim().min(vars.len())], &params);
    let again = Formula::parse(f.root_formula());
    assert!(
        again.is_ok(),
        "{source:?} parsed, but its ROOT spelling {:?} does not: {:?}",
        f.root_formula(),
        again.err()
    );
});
