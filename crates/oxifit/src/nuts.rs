//! Hamiltonian Monte Carlo with the No-U-Turn Sampler (Hoffman and Gelman,
//! JMLR 15 (2014) 1593), with Stan's warm-up: a dual-averaging step size and a
//! diagonal metric learned in widening windows.

use crate::fitter::Fitter;
use crate::mcmc::Chain;
use crate::result::FitOptions;
use crate::rng::Rng;
use crate::scan::FitError;

/// Draws from a [`Fitter`]'s posterior with the No-U-Turn Sampler: the same
/// posterior as [`Sampler`](crate::Sampler) (the cost as `−2 ln L`, limits as
/// flat priors, Gaussian constraints as Gaussian priors), explored with the
/// cost's gradient.
///
/// Each step follows the Hamiltonian dynamics of the posterior until the path
/// starts to turn back, so the sampler moves far in one step and its draws
/// are much less correlated than the ensemble sampler's — the method of Stan
/// and PyMC, best for many parameters. A parameter with limits is sampled in
/// unconstrained coordinates (log or logit), with the Jacobian. The cost's
/// analytic gradient is used when it has one, else central differences (two
/// cost evaluations per free parameter per leapfrog step).
///
/// Warm-up (Stan's schedule): the step size is tuned by dual averaging to an
/// acceptance of [`target_accept`](Self::target_accept), and the diagonal
/// metric is estimated from the warm-up draws in windows of doubling length.
/// Chains run independently from dispersed starts — in parallel with the
/// `rayon` feature — so [`Chain::r_hat`] compares them; each depends only on
/// the seed.
///
/// ```
/// use oxifit::{FnCost, Fitter, Nuts};
///
/// // A correlated 2-D Gaussian likelihood.
/// let cost = FnCost::new(&["a", "b"], |p| {
///     let (u, v) = (p[0] - 1.0, (p[1] + 2.0) / 2.0);
///     (u * u - 1.2 * u * v + v * v) / 0.64
/// });
/// let chain = Nuts::new(Fitter::new(&cost, vec![0.0, 0.0])).seed(1).run(500, 1000)?;
/// assert!((chain.mean("a").unwrap() - 1.0).abs() < 0.1);
/// assert!(chain.r_hat("b").unwrap() < 1.05);
/// assert_eq!(chain.divergences(), 0);
/// # Ok::<(), oxifit::FitError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Nuts<'c> {
    fitter: Fitter<'c>,
    chains: usize,
    seed: u64,
    target_accept: f64,
    max_depth: usize,
}

impl<'c> Nuts<'c> {
    /// A NUTS sampler of `fitter`'s posterior: 4 chains, seed 1, target
    /// acceptance 0.8, trees of at most 2¹⁰ leapfrog steps.
    #[must_use]
    pub fn new(fitter: Fitter<'c>) -> Nuts<'c> {
        Nuts {
            fitter,
            chains: 4,
            seed: 1,
            target_accept: 0.8,
            max_depth: 10,
        }
    }

    /// The number of independent chains (at least 1).
    #[must_use]
    pub fn chains(mut self, n: usize) -> Nuts<'c> {
        self.chains = n.max(1);
        self
    }

    /// The generator seed: the same seed gives the same draws.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Nuts<'c> {
        self.seed = seed;
        self
    }

    /// The acceptance the warm-up tunes the step size for (0.8 by default;
    /// raise it towards 0.95 when there are divergences).
    #[must_use]
    pub fn target_accept(mut self, a: f64) -> Nuts<'c> {
        self.target_accept = a.clamp(0.5, 0.999);
        self
    }

    /// The deepest tree: at most `2^depth` leapfrog steps per draw (10 by
    /// default, as Stan).
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Nuts<'c> {
        self.max_depth = depth.clamp(1, 20);
        self
    }

    /// Run `warmup` adaptation iterations (discarded), then keep `draws` per
    /// chain. The [`Chain`]'s walkers are the chains, its acceptance the mean
    /// acceptance statistic of the kept draws.
    ///
    /// # Errors
    /// [`FitError::FitFailed`] if no free parameter is left or a chain cannot
    /// start at a finite posterior density; as
    /// [`Model::check`](crate::Model::check) for a parameter setting of an
    /// unknown name.
    pub fn run(&self, warmup: usize, draws: usize) -> Result<Chain, FitError> {
        self.fitter.params.check()?;
        let spec = &self.fitter.params;
        let free: Vec<usize> = (0..spec.params.len())
            .filter(|&i| !spec.constraints.get(i).is_some_and(|c| c.fixed))
            .collect();
        if free.is_empty() || spec.params.len() != self.fitter.cost.param_names().len() {
            return Err(FitError::FitFailed);
        }
        let target = Target::new(&self.fitter, &free);

        // A common center (the best fit) and scale (its errors), in the
        // unconstrained coordinates.
        let best = self.fitter.fit(&FitOptions::new().hesse(true));
        let start_x = if best.valid {
            best.params.clone()
        } else {
            spec.params.clone()
        };
        let center = target.to_u(&start_x);
        let scale: Vec<f64> = free
            .iter()
            .zip(&center)
            .enumerate()
            .map(|(k, (&i, &u))| {
                let e = best.errors.get(i).copied().unwrap_or(f64::NAN);
                let s = e / target.dx_du(k, u);
                if best.valid && s.is_finite() && s > 0.0 {
                    s
                } else {
                    1.0
                }
            })
            .collect();

        let run_chain = |c: usize| -> Result<ChainDraws, FitError> {
            ChainRun {
                target: &target,
                rng: Rng::stream(self.seed, c as u64),
                max_depth: self.max_depth,
                delta: self.target_accept,
            }
            .run(&center, &scale, warmup, draws)
        };
        let ids: Vec<usize> = (0..self.chains).collect();
        #[cfg(feature = "rayon")]
        let runs: Vec<Result<ChainDraws, FitError>> = {
            use rayon::prelude::*;
            ids.par_iter().map(|&c| run_chain(c)).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let runs: Vec<Result<ChainDraws, FitError>> = ids.iter().map(|&c| run_chain(c)).collect();
        let runs = runs.into_iter().collect::<Result<Vec<_>, _>>()?;

        let samples: Vec<Vec<Vec<f64>>> = (0..draws)
            .map(|t| runs.iter().map(|r| r.draws[t].clone()).collect())
            .collect();
        let accept = runs.iter().map(|r| r.accept).sum::<f64>() / runs.len() as f64;
        let divergences = runs.iter().map(|r| r.divergences).sum();
        Ok(Chain::from_parts(
            spec.param_names.clone(),
            free,
            samples,
            accept,
            divergences,
        ))
    }
}

/// The posterior in unconstrained coordinates `u` of the free parameters.
struct Target<'a, 'c> {
    fitter: &'a Fitter<'c>,
    free: &'a [usize],
    base: Vec<f64>,
    bounds: Vec<(Option<f64>, Option<f64>)>,
    analytic: bool,
}

impl<'a, 'c> Target<'a, 'c> {
    fn new(fitter: &'a Fitter<'c>, free: &'a [usize]) -> Target<'a, 'c> {
        let spec = &fitter.params;
        let bounds = free
            .iter()
            .map(|&i| {
                spec.constraints
                    .get(i)
                    .map_or((None, None), |c| (c.lower, c.upper))
            })
            .collect();
        Target {
            fitter,
            free,
            base: spec.params.clone(),
            bounds,
            analytic: fitter.cost.has_gradient(),
        }
    }

    /// `u` of the full parameter vector `x` (inset from the limits).
    fn to_u(&self, x: &[f64]) -> Vec<f64> {
        self.free
            .iter()
            .zip(&self.bounds)
            .map(|(&i, &b)| match b {
                (Some(lo), Some(hi)) => {
                    let t = ((x[i] - lo) / (hi - lo)).clamp(1e-6, 1.0 - 1e-6);
                    (t / (1.0 - t)).ln()
                }
                (Some(lo), None) => (x[i] - lo).max(1e-9 * lo.abs().max(1.0)).ln(),
                (None, Some(hi)) => (hi - x[i]).max(1e-9 * hi.abs().max(1.0)).ln(),
                (None, None) => x[i],
            })
            .collect()
    }

    /// `dx/du` of free parameter `k` at `u`.
    fn dx_du(&self, k: usize, u: f64) -> f64 {
        match self.bounds[k] {
            (Some(lo), Some(hi)) => {
                let s = sigmoid(u);
                (hi - lo) * s * (1.0 - s)
            }
            (Some(_), None) | (None, Some(_)) => u.exp(),
            (None, None) => 1.0,
        }
    }

    /// The full parameter vector at `u`, `ln|dx/du|` summed, and per free
    /// parameter `dx/du` and `d ln|dx/du| / du`.
    fn to_x(&self, u: &[f64]) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
        let mut x = self.base.clone();
        let mut log_jac = 0.0;
        let mut dxdu = Vec::with_capacity(u.len());
        let mut dlj = Vec::with_capacity(u.len());
        for ((&i, &b), &uk) in self.free.iter().zip(&self.bounds).zip(u) {
            match b {
                (Some(lo), Some(hi)) => {
                    let s = sigmoid(uk);
                    x[i] = lo + (hi - lo) * s;
                    let d = (hi - lo) * s * (1.0 - s);
                    log_jac += d.ln();
                    dxdu.push(d);
                    dlj.push(1.0 - 2.0 * s);
                }
                (Some(lo), None) => {
                    x[i] = lo + uk.exp();
                    log_jac += uk;
                    dxdu.push(uk.exp());
                    dlj.push(1.0);
                }
                (None, Some(hi)) => {
                    x[i] = hi - uk.exp();
                    log_jac += uk;
                    dxdu.push(-uk.exp());
                    dlj.push(1.0);
                }
                (None, None) => {
                    x[i] = uk;
                    dxdu.push(1.0);
                    dlj.push(0.0);
                }
            }
        }
        (x, log_jac, dxdu, dlj)
    }

    /// The cost's gradient in the free parameters at `x` (constraints
    /// included).
    fn cost_gradient(&self, x: &[f64]) -> Vec<f64> {
        if self.analytic {
            let mut g = vec![0.0; x.len()];
            self.fitter.cost.gradient(x, &mut g);
            return self
                .free
                .iter()
                .map(|&i| {
                    let pen = self.fitter.params.constraints.get(i).and_then(|c| c.gauss);
                    g[i] + pen.map_or(0.0, |(m, s)| 2.0 * (x[i] - m) / (s * s))
                })
                .collect();
        }
        self.free
            .iter()
            .map(|&i| {
                let h = 1e-5 * x[i].abs().max(1e-2);
                let (mut up, mut down) = (x.to_vec(), x.to_vec());
                up[i] += h;
                down[i] -= h;
                (self.fitter.value(&up) - self.fitter.value(&down)) / (2.0 * h)
            })
            .collect()
    }

    /// `ln p(u)` and its gradient; `−∞` where the cost is not finite.
    fn eval(&self, u: &[f64]) -> (f64, Vec<f64>) {
        let (x, log_jac, dxdu, dlj) = self.to_x(u);
        let v = self.fitter.value(&x);
        if !v.is_finite() || !log_jac.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; u.len()]);
        }
        let g = self.cost_gradient(&x);
        let grad: Vec<f64> = g
            .iter()
            .zip(&dxdu)
            .zip(&dlj)
            .map(|((gx, d), j)| -0.5 * gx * d + j)
            .collect();
        if grad.iter().any(|v| !v.is_finite()) {
            return (f64::NEG_INFINITY, vec![0.0; u.len()]);
        }
        (-0.5 * v + log_jac, grad)
    }
}

fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

/// A point of the trajectory: position, momentum, gradient, `ln p`.
#[derive(Clone)]
struct State {
    q: Vec<f64>,
    p: Vec<f64>,
    g: Vec<f64>,
    logp: f64,
}

/// What a subtree returns.
struct Subtree {
    minus: State,
    plus: State,
    proposal: State,
    n: f64,
    ok: bool,
    alpha: f64,
    n_alpha: f64,
}

struct ChainDraws {
    draws: Vec<Vec<f64>>,
    accept: f64,
    divergences: usize,
}

struct ChainRun<'t, 'a, 'c> {
    target: &'t Target<'a, 'c>,
    rng: Rng,
    max_depth: usize,
    delta: f64,
}

/// The largest energy error before a trajectory counts as divergent.
const MAX_ENERGY_ERROR: f64 = 1000.0;

impl ChainRun<'_, '_, '_> {
    fn kinetic(minv: &[f64], p: &[f64]) -> f64 {
        0.5 * p.iter().zip(minv).map(|(p, m)| m * p * p).sum::<f64>()
    }

    fn leapfrog(&self, s: &State, eps: f64, minv: &[f64]) -> State {
        let mut p: Vec<f64> =
            s.p.iter()
                .zip(&s.g)
                .map(|(p, g)| p + 0.5 * eps * g)
                .collect();
        let q: Vec<f64> =
            s.q.iter()
                .zip(&p)
                .zip(minv)
                .map(|((q, p), m)| q + eps * m * p)
                .collect();
        let (logp, g) = self.target.eval(&q);
        for (pi, gi) in p.iter_mut().zip(&g) {
            *pi += 0.5 * eps * gi;
        }
        State { q, p, g, logp }
    }

    fn no_u_turn(minus: &State, plus: &State, minv: &[f64]) -> bool {
        let (mut a, mut b) = (0.0, 0.0);
        for k in 0..minus.q.len() {
            let dq = plus.q[k] - minus.q[k];
            a += dq * minv[k] * minus.p[k];
            b += dq * minv[k] * plus.p[k];
        }
        a >= 0.0 && b >= 0.0
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &mut self,
        s: &State,
        log_u: f64,
        forward: bool,
        depth: usize,
        eps: f64,
        h0: f64,
        minv: &[f64],
        divergent: &mut bool,
    ) -> Subtree {
        if depth == 0 {
            let next = self.leapfrog(s, if forward { eps } else { -eps }, minv);
            let joint = next.logp - Self::kinetic(minv, &next.p);
            let joint = if joint.is_nan() {
                f64::NEG_INFINITY
            } else {
                joint
            };
            let ok = log_u < joint + MAX_ENERGY_ERROR;
            if !ok {
                *divergent = true;
            }
            return Subtree {
                minus: next.clone(),
                plus: next.clone(),
                proposal: next,
                n: if log_u <= joint { 1.0 } else { 0.0 },
                ok,
                alpha: (joint - h0).exp().min(1.0),
                n_alpha: 1.0,
            };
        }
        let mut t = self.build(s, log_u, forward, depth - 1, eps, h0, minv, divergent);
        if !t.ok {
            return t;
        }
        let edge = if forward {
            t.plus.clone()
        } else {
            t.minus.clone()
        };
        let t2 = self.build(&edge, log_u, forward, depth - 1, eps, h0, minv, divergent);
        if forward {
            t.plus = t2.plus;
        } else {
            t.minus = t2.minus;
        }
        let total = t.n + t2.n;
        if total > 0.0 && self.rng.uniform() < t2.n / total {
            t.proposal = t2.proposal;
        }
        t.alpha += t2.alpha;
        t.n_alpha += t2.n_alpha;
        t.ok = t2.ok && Self::no_u_turn(&t.minus, &t.plus, minv);
        t.n = total;
        t
    }

    /// One NUTS transition from `s`: the new state, the acceptance
    /// statistic, and whether the trajectory diverged.
    fn transition(&mut self, s: &State, eps: f64, minv: &[f64]) -> (State, f64, bool) {
        let p0: Vec<f64> = minv.iter().map(|m| self.rng.gauss() / m.sqrt()).collect();
        let start = State { p: p0, ..s.clone() };
        let h0 = start.logp - Self::kinetic(minv, &start.p);
        let log_u = h0 + self.rng.uniform().max(f64::MIN_POSITIVE).ln();
        let (mut minus, mut plus) = (start.clone(), start.clone());
        let mut current = s.clone();
        let mut n = 1.0;
        let (mut alpha, mut n_alpha) = (0.0, 0.0);
        let mut divergent = false;
        for depth in 0..self.max_depth {
            let forward = self.rng.uniform() < 0.5;
            let from = if forward { plus.clone() } else { minus.clone() };
            let t = self.build(&from, log_u, forward, depth, eps, h0, minv, &mut divergent);
            if forward {
                plus = t.plus;
            } else {
                minus = t.minus;
            }
            alpha += t.alpha;
            n_alpha += t.n_alpha;
            if t.ok && self.rng.uniform() < t.n / n {
                current = State {
                    p: s.p.clone(),
                    ..t.proposal
                };
            }
            n += t.n;
            if !t.ok || !Self::no_u_turn(&minus, &plus, minv) {
                break;
            }
        }
        (current, alpha / n_alpha.max(1.0), divergent)
    }

    /// A step size of the right order (Hoffman and Gelman, Algorithm 4).
    fn initial_step(&mut self, s: &State, minv: &[f64]) -> f64 {
        let mut eps = 1.0;
        let p: Vec<f64> = minv.iter().map(|m| self.rng.gauss() / m.sqrt()).collect();
        let start = State { p, ..s.clone() };
        let h0 = start.logp - Self::kinetic(minv, &start.p);
        let log_ratio = |run: &Self, eps: f64| {
            let next = run.leapfrog(&start, eps, minv);
            let h = next.logp - Self::kinetic(minv, &next.p);
            if h.is_finite() {
                h - h0
            } else {
                f64::NEG_INFINITY
            }
        };
        let up = log_ratio(self, eps) > 0.5f64.ln();
        for _ in 0..50 {
            let r = log_ratio(self, eps);
            if up && r <= 0.5f64.ln() || !up && r > 0.5f64.ln() {
                break;
            }
            eps = if up { eps * 2.0 } else { eps / 2.0 };
        }
        eps
    }

    fn run(
        mut self,
        center: &[f64],
        scale: &[f64],
        warmup: usize,
        draws: usize,
    ) -> Result<ChainDraws, FitError> {
        // A dispersed start: the best fit plus one error, at finite density.
        let mut state = None;
        for attempt in 0..100 {
            let spread = if attempt < 50 { 1.0 } else { 0.1 };
            let q: Vec<f64> = center
                .iter()
                .zip(scale)
                .map(|(c, s)| c + spread * s * self.rng.gauss())
                .collect();
            let (logp, g) = self.target.eval(&q);
            if logp.is_finite() {
                state = Some(State {
                    p: vec![0.0; q.len()],
                    q,
                    g,
                    logp,
                });
                break;
            }
        }
        let mut s = state.ok_or(FitError::FitFailed)?;
        let mut minv: Vec<f64> = scale.iter().map(|s| s * s).collect();

        // Stan's schedule: a fast phase, slow windows of doubling length that
        // estimate the metric, and a final fast phase.
        let (init, term) = if warmup >= 150 {
            (warmup * 15 / 100, warmup / 10)
        } else {
            (warmup, 0)
        };
        let slow_end = warmup - term;
        let mut window_ends = Vec::new();
        let (mut start, mut size) = (init, 25usize);
        while start < slow_end {
            let mut end = start + size;
            if end + 2 * size > slow_end {
                end = slow_end;
            }
            window_ends.push(end);
            start = end;
            size *= 2;
        }

        let mut eps = self.initial_step(&s, &minv);
        let mut da = DualAveraging::new(eps, self.delta);
        let mut window: Vec<Vec<f64>> = Vec::new();
        for it in 0..warmup {
            let (next, accept, _) = self.transition(&s, eps, &minv);
            s = next;
            eps = da.update(accept);
            if it >= init && it < slow_end {
                window.push(s.q.clone());
            }
            if window_ends.contains(&(it + 1)) {
                let n = window.len() as f64;
                if n >= 3.0 {
                    for (k, m) in minv.iter_mut().enumerate() {
                        let mean = window.iter().map(|q| q[k]).sum::<f64>() / n;
                        let var =
                            window.iter().map(|q| (q[k] - mean).powi(2)).sum::<f64>() / (n - 1.0);
                        *m = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0));
                    }
                }
                window.clear();
                eps = self.initial_step(&s, &minv);
                da = DualAveraging::new(eps, self.delta);
            }
        }
        if warmup > 0 {
            eps = da.final_step();
        }

        let mut out = Vec::with_capacity(draws);
        let (mut accept_sum, mut divergences) = (0.0, 0usize);
        for _ in 0..draws {
            let (next, accept, divergent) = self.transition(&s, eps, &minv);
            s = next;
            accept_sum += accept;
            divergences += usize::from(divergent);
            out.push(self.target.to_x(&s.q).0);
        }
        Ok(ChainDraws {
            draws: out,
            accept: accept_sum / draws.max(1) as f64,
            divergences,
        })
    }
}

/// Nesterov dual averaging of the log step size (Hoffman and Gelman,
/// Algorithm 5; Stan's constants).
struct DualAveraging {
    mu: f64,
    delta: f64,
    h_bar: f64,
    log_eps_bar: f64,
    m: f64,
}

impl DualAveraging {
    fn new(eps: f64, delta: f64) -> DualAveraging {
        DualAveraging {
            mu: (10.0 * eps).ln(),
            delta,
            h_bar: 0.0,
            log_eps_bar: 0.0,
            m: 0.0,
        }
    }

    /// Fold in one acceptance statistic; the next step size.
    fn update(&mut self, accept: f64) -> f64 {
        const GAMMA: f64 = 0.05;
        const T0: f64 = 10.0;
        const KAPPA: f64 = 0.75;
        self.m += 1.0;
        let w = 1.0 / (self.m + T0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.delta - accept);
        let log_eps = self.mu - self.m.sqrt() / GAMMA * self.h_bar;
        let eta = self.m.powf(-KAPPA);
        self.log_eps_bar = eta * log_eps + (1.0 - eta) * self.log_eps_bar;
        log_eps.exp()
    }

    /// The averaged step size, for sampling.
    fn final_step(&self) -> f64 {
        if self.m > 0.0 {
            self.log_eps_bar.exp()
        } else {
            self.mu.exp() / 10.0
        }
    }
}
