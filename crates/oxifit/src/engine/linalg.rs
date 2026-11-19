//! Small dense linear algebra for the in-house backends: a linear solve
//! (Levenberg–Marquardt) and a symmetric eigendecomposition (CMA-ES). The
//! matrices are `n × n` for `n` free parameters — a handful to a few dozen —
//! so plain O(n³) methods are plenty.

/// Solve `A·x = b` by Gaussian elimination with partial pivoting; `None` if
/// `A` is singular.
pub(crate) fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col].abs() < 1e-300 || !m[pivot][col].is_finite() {
            return None;
        }
        m.swap(col, pivot);
        let pivot_row = m[col].clone();
        for row in m.iter_mut().skip(col + 1) {
            let factor = row[col] / pivot_row[col];
            if factor != 0.0 {
                for (dst, &src) in row.iter_mut().zip(&pivot_row).skip(col) {
                    *dst -= factor * src;
                }
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let tail: f64 = (i + 1..n).map(|j| m[i][j] * x[j]).sum();
        x[i] = (m[i][n] - tail) / m[i][i];
    }
    Some(x)
}

/// Eigendecomposition of a symmetric matrix by the cyclic Jacobi method:
/// returns the eigenvalues and the eigenvectors as the *columns* of the
/// returned matrix (`a = V·diag(λ)·Vᵀ`).
pub(crate) fn sym_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j] * m[i][j])
            .sum();
        let scale: f64 = (0..n).map(|i| m[i][i] * m[i][i]).sum::<f64>().max(1e-300);
        if off <= 1e-24 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                // Rotation that zeroes m[p][q].
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in m.iter_mut() {
                    let (mkp, mkq) = (row[p], row[q]);
                    row[p] = c * mkp - s * mkq;
                    row[q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i][i]).collect(), v)
}

/// Invert a small square matrix by Gauss–Jordan elimination with partial
/// pivoting; `None` if singular. Used only for the tiny (few-parameter)
/// covariance, so an O(n³) dense solve is plenty.
#[cfg(feature = "argmin")]
pub(crate) fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    if n == 0 {
        return Some(Vec::new());
    }
    // Augment with the identity: [m | I].
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a_, &b_| a[a_][col].abs().total_cmp(&a[b_][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        let d = a[col][col];
        for x in a[col].iter_mut() {
            *x /= d;
        }
        // The pivot row is unchanged while we eliminate the others, so clone it
        // once and subtract by iterator (no double-indexing of `a`).
        let pivot_row = a[col].clone();
        for (row, a_row) in a.iter_mut().enumerate() {
            if row != col {
                let factor = a_row[col];
                if factor != 0.0 {
                    for (dst, &src) in a_row.iter_mut().zip(&pivot_row) {
                        *dst -= factor * src;
                    }
                }
            }
        }
    }
    Some(a.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// The Hessian of `f` at `x` by central differences with steps `h`.
pub(crate) fn hessian(f: &impl Fn(&[f64]) -> f64, x: &[f64], h: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let at = |di: (usize, f64), dj: Option<(usize, f64)>| {
        let mut s = x.to_vec();
        s[di.0] += di.1;
        if let Some((j, d)) = dj {
            s[j] += d;
        }
        f(&s)
    };
    let mut hess = vec![vec![0.0; n]; n];
    for i in 0..n {
        hess[i][i] = (at((i, h[i]), None) - 2.0 * f0 + at((i, -h[i]), None)) / (h[i] * h[i]);
        for j in (i + 1)..n {
            let v = (at((i, h[i]), Some((j, h[j])))
                - at((i, h[i]), Some((j, -h[j])))
                - at((i, -h[i]), Some((j, h[j])))
                + at((i, -h[i]), Some((j, -h[j]))))
                / (4.0 * h[i] * h[j]);
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    hess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_recovers_a_known_solution() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, -1.0],
            vec![0.5, -1.0, 2.0],
        ];
        let x = [1.0, -2.0, 3.0];
        let b: Vec<f64> = a
            .iter()
            .map(|r| r.iter().zip(&x).map(|(p, q)| p * q).sum())
            .collect();
        let got = solve(&a, &b).unwrap();
        for (g, w) in got.iter().zip(&x) {
            assert!((g - w).abs() < 1e-12, "{got:?}");
        }
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn sym_eigen_reconstructs_the_matrix() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, -1.0],
            vec![0.5, -1.0, 2.0],
        ];
        let (l, v) = sym_eigen(&a);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[i][k] * l[k] * v[j][k]).sum();
                assert!((r - a[i][j]).abs() < 1e-10, "({i},{j}): {r} vs {}", a[i][j]);
            }
        }
        // The trace is the sum of the eigenvalues.
        assert!((l.iter().sum::<f64>() - 9.0).abs() < 1e-10);
    }
}
