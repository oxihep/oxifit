//! Minuit's parameter transformations: an unbounded *internal* coordinate for
//! each limited parameter, so an optimizer that knows nothing of limits
//! (Nelder–Mead) can move freely while the *external* value stays inside them.
//! Exactly Minuit2's `SinParameterTransformation` (two limits) and
//! `SqrtLow`/`SqrtUpParameterTransformation` (one).

use crate::model::Constraint;

/// The transformation of one parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Transform {
    /// No limits: internal = external.
    Free,
    /// `[lo, hi]`: `ext = lo + (hi − lo)·(sin(int) + 1)/2`.
    Both(f64, f64),
    /// `≥ lo`: `ext = lo − 1 + √(int² + 1)`.
    Lower(f64),
    /// `≤ hi`: `ext = hi + 1 − √(int² + 1)`.
    Upper(f64),
}

impl Transform {
    /// The transformation for a parameter's limits.
    pub(crate) fn of(c: &Constraint) -> Transform {
        match (c.lower, c.upper) {
            (Some(lo), Some(hi)) if hi > lo => Transform::Both(lo, hi),
            (Some(lo), None) => Transform::Lower(lo),
            (None, Some(hi)) => Transform::Upper(hi),
            _ => Transform::Free,
        }
    }

    /// The external (model) value of internal coordinate `int`.
    pub(crate) fn to_ext(self, int: f64) -> f64 {
        match self {
            Transform::Free => int,
            Transform::Both(lo, hi) => lo + 0.5 * (hi - lo) * (int.sin() + 1.0),
            Transform::Lower(lo) => lo - 1.0 + (int * int + 1.0).sqrt(),
            Transform::Upper(hi) => hi + 1.0 - (int * int + 1.0).sqrt(),
        }
    }

    /// The internal coordinate of external value `ext`, which is first moved
    /// inside the limits.
    pub(crate) fn to_int(self, ext: f64) -> f64 {
        match self {
            Transform::Free => ext,
            Transform::Both(lo, hi) => {
                let u = (2.0 * (ext - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0);
                u.asin()
            }
            Transform::Lower(lo) => {
                let d = (ext - lo).max(0.0) + 1.0;
                (d * d - 1.0).sqrt()
            }
            Transform::Upper(hi) => {
                let d = (hi - ext).max(0.0) + 1.0;
                (d * d - 1.0).sqrt()
            }
        }
    }

    /// An internal step matching external step `step` at internal `int`:
    /// `step / |d ext/d int|`, kept within `[1e-3, 1]` so a parameter at its
    /// limit (where the derivative vanishes) still gets a usable step.
    pub(crate) fn int_step(self, int: f64, step: f64) -> f64 {
        let h = 1e-6 * (1.0 + int.abs());
        let slope = (self.to_ext(int + h) - self.to_ext(int - h)).abs() / (2.0 * h);
        match self {
            Transform::Free => step,
            _ if slope > 0.0 => (step / slope).clamp(1e-3, 1.0),
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Transform;

    #[test]
    fn round_trips_and_stays_inside_the_limits() {
        for t in [
            Transform::Free,
            Transform::Both(-1.0, 3.0),
            Transform::Lower(0.5),
            Transform::Upper(2.0),
        ] {
            for ext in [-0.9, 0.0, 0.6, 1.9, 2.5] {
                let inside = match t {
                    Transform::Both(lo, hi) => (lo..=hi).contains(&ext),
                    Transform::Lower(lo) => ext >= lo,
                    Transform::Upper(hi) => ext <= hi,
                    Transform::Free => true,
                };
                if inside {
                    let back = t.to_ext(t.to_int(ext));
                    assert!((back - ext).abs() < 1e-12, "{t:?}: {ext} -> {back}");
                }
            }
            // Any internal value maps inside the limits.
            for int in [-1e3, -2.0, -0.3, 0.0, 0.7, 5.0, 1e4] {
                let e = t.to_ext(int);
                match t {
                    Transform::Both(lo, hi) => assert!((lo..=hi).contains(&e)),
                    Transform::Lower(lo) => assert!(e >= lo),
                    Transform::Upper(hi) => assert!(e <= hi),
                    Transform::Free => assert_eq!(e, int),
                }
            }
        }
    }
}
