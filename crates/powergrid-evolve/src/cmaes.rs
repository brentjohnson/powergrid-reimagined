//! A compact (μ/μ_w, λ)-CMA-ES, following Hansen's `purecmaes` reference.
//!
//! Self-contained: the only non-trivial linear algebra is a symmetric
//! eigendecomposition of the `n×n` covariance matrix (n = 14 here), done with a
//! cyclic Jacobi solver below — no `nalgebra`/BLAS dependency. The optimizer is
//! generic over the objective; the caller evaluates the sampled points and
//! passes back fitnesses (lower = better).

// Matrix/vector math reads far clearer with explicit index loops here.
#![allow(clippy::needless_range_loop)]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Serializable optimizer state (everything needed to resume a run).
#[derive(Clone, Serialize, Deserialize)]
pub struct CmaEs {
    pub n: usize,
    pub lambda: usize,
    pub mu: usize,
    pub weights: Vec<f64>,
    pub mueff: f64,
    pub cc: f64,
    pub cs: f64,
    pub c1: f64,
    pub cmu: f64,
    pub damps: f64,
    pub chin: f64,

    pub mean: Vec<f64>,
    pub sigma: f64,
    pub pc: Vec<f64>,
    pub ps: Vec<f64>,
    pub c: Vec<Vec<f64>>, // covariance
    pub b: Vec<Vec<f64>>, // eigenvectors (columns)
    pub d: Vec<f64>,      // sqrt eigenvalues (per-axis std devs)

    pub counteval: usize,
    pub eigeneval: usize,
    pub gen: usize,
    pub rng_state: u64,

    // Pending batch between ask() and tell().
    #[serde(skip)]
    pending_x: Vec<Vec<f64>>,
    #[serde(skip)]
    xold: Vec<f64>,
}

fn eye(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

impl CmaEs {
    /// `sigma0` is the initial step size in normalized search units.
    /// `lambda` of 0 selects the default `4 + floor(3 ln n)`.
    pub fn new(n: usize, sigma0: f64, lambda: usize, rng_seed: u64) -> Self {
        let lambda = if lambda == 0 {
            4 + (3.0 * (n as f64).ln()).floor() as usize
        } else {
            lambda
        };
        let mu_f = lambda as f64 / 2.0;
        let mu = mu_f.floor() as usize;

        // Recombination weights (log-decreasing over the best mu).
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| (mu_f + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= wsum;
        }
        let sumsq: f64 = weights.iter().map(|w| w * w).sum();
        let mueff = 1.0 / sumsq;

        let nf = n as f64;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chin = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        CmaEs {
            n,
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chin,
            mean: vec![0.0; n],
            sigma: sigma0,
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            c: eye(n),
            b: eye(n),
            d: vec![1.0; n],
            counteval: 0,
            eigeneval: 0,
            gen: 0,
            rng_state: rng_seed,
            pending_x: Vec::new(),
            xold: Vec::new(),
        }
    }

    fn rng(&mut self) -> SmallRng {
        // Derive a fresh generator per ask() from the persisted state, then
        // advance the persisted state so resumed runs don't repeat samples.
        let r = SmallRng::seed_from_u64(self.rng_state);
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        r
    }

    /// Standard normal via Box–Muller (avoids a rand_distr dependency).
    fn randn(rng: &mut SmallRng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-12);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Sample `lambda` candidate vectors `x_k = mean + sigma * B (D ⊙ z_k)`.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        let mut rng = self.rng();
        let n = self.n;
        let mut batch = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            let z: Vec<f64> = (0..n).map(|_| Self::randn(&mut rng)).collect();
            let dz: Vec<f64> = (0..n).map(|i| self.d[i] * z[i]).collect();
            // y = B * dz
            let mut x = vec![0.0; n];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += self.b[i][j] * dz[j];
                }
                x[i] = self.mean[i] + self.sigma * acc;
            }
            batch.push(x);
        }
        self.xold = self.mean.clone();
        self.pending_x = batch.clone();
        batch
    }

    /// Update the distribution from the fitnesses of the last [`ask`] batch
    /// (lower fitness = better). Order must match the returned batch.
    pub fn tell(&mut self, fitnesses: &[f64]) {
        let n = self.n;
        assert_eq!(fitnesses.len(), self.pending_x.len(), "tell arity mismatch");
        self.counteval += self.lambda;

        let mut idx: Vec<usize> = (0..fitnesses.len()).collect();
        idx.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

        // Weighted mean of the best mu, and the corresponding normalized step.
        let mut newmean = vec![0.0; n];
        for (w, &k) in self.weights.iter().zip(idx.iter().take(self.mu)) {
            for i in 0..n {
                newmean[i] += w * self.pending_x[k][i];
            }
        }
        // (newmean - xold) / sigma = B (D ⊙ zmean)
        let ymean: Vec<f64> = (0..n)
            .map(|i| (newmean[i] - self.xold[i]) / self.sigma)
            .collect();
        // zmean = B^T ymean ./ D
        let bt_ymean = self.bt_mul(&ymean);
        let zmean: Vec<f64> = (0..n).map(|i| bt_ymean[i] / self.d[i]).collect();
        // C^{-1/2} (newmean - xold)/sigma = B * zmean
        let c_inv_sqrt_step = self.b_mul(&zmean);

        // ps update
        let cs_factor = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for i in 0..n {
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + cs_factor * c_inv_sqrt_step[i];
        }
        let ps_norm = norm(&self.ps);
        let hsig = ps_norm
            / (1.0 - (1.0 - self.cs).powi(2 * (self.counteval / self.lambda) as i32)).sqrt()
            / self.chin
            < 1.4 + 2.0 / (n as f64 + 1.0);
        let hsig_f = if hsig { 1.0 } else { 0.0 };

        // pc update
        let cc_factor = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        for i in 0..n {
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + hsig_f * cc_factor * ymean[i];
        }

        // Covariance update: rank-one (pc) + rank-mu (best steps).
        let delta_hsig = (1.0 - hsig_f) * self.cc * (2.0 - self.cc);
        for i in 0..n {
            for j in 0..n {
                let rank_one = self.pc[i] * self.pc[j] + delta_hsig * self.c[i][j];
                let mut rank_mu = 0.0;
                for (w, &k) in self.weights.iter().zip(idx.iter().take(self.mu)) {
                    let yi = (self.pending_x[k][i] - self.xold[i]) / self.sigma;
                    let yj = (self.pending_x[k][j] - self.xold[j]) / self.sigma;
                    rank_mu += w * yi * yj;
                }
                self.c[i][j] = (1.0 - self.c1 - self.cmu) * self.c[i][j]
                    + self.c1 * rank_one
                    + self.cmu * rank_mu;
            }
        }

        // Step-size update.
        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chin - 1.0)).exp();

        self.mean = newmean;
        self.gen += 1;

        // Refresh eigendecomposition occasionally (amortized O(n^2) per eval).
        let interval = (self.lambda as f64 / (self.c1 + self.cmu) / n as f64 / 10.0).max(1.0);
        if (self.counteval - self.eigeneval) as f64 > interval {
            self.eigeneval = self.counteval;
            self.refresh_eigen();
        }
    }

    fn refresh_eigen(&mut self) {
        // Symmetrize to kill accumulated round-off, then decompose.
        let n = self.n;
        let mut sym = self.c.clone();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (sym[i][j] + sym[j][i]);
                sym[i][j] = avg;
                sym[j][i] = avg;
            }
        }
        let (evals, evecs) = jacobi_eig(&sym);
        self.b = evecs;
        self.d = evals.iter().map(|&e| e.max(1e-20).sqrt()).collect();
    }

    /// B * v
    fn b_mul(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.b[i][j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// B^T * v
    fn bt_mul(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.b[j][i] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Restore the transient ask/tell scratch fields after deserialization.
    /// (A resumed run simply starts a fresh generation, so these begin empty.)
    pub fn after_load(&mut self) {
        self.pending_x = Vec::new();
        self.xold = self.mean.clone();
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix.
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored as
/// columns: `V[i][k]` is component `i` of the `k`-th eigenvector.
pub fn jacobi_eig(input: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = input.len();
    let mut a = input.to_vec();
    let mut v = eye(n);

    for _sweep in 0..100 {
        // Sum of squared off-diagonal entries.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-20 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if tau == 0.0 {
                    1.0
                } else {
                    tau.signum() / (tau.abs() + (tau * tau + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Rotate columns p,q of A.
                for i in 0..n {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[i][q] = s * aip + c * aiq;
                }
                // Rotate rows p,q of A.
                for j in 0..n {
                    let apj = a[p][j];
                    let aqj = a[q][j];
                    a[p][j] = c * apj - s * aqj;
                    a[q][j] = s * apj + c * aqj;
                }
                // Accumulate the rotation into V.
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes_known_matrix() {
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3.
        let (evals, _v) = jacobi_eig(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        let mut e = evals.clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] - 1.0).abs() < 1e-9);
        assert!((e[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn minimizes_sphere() {
        // Sphere shifted to (1, -2, 0.5, ...): CMA should drive the mean there.
        let n = 5;
        let target: Vec<f64> = vec![1.0, -2.0, 0.5, -1.5, 0.0];
        let mut es = CmaEs::new(n, 0.5, 0, 7);
        for _ in 0..200 {
            let batch = es.ask();
            let fits: Vec<f64> = batch
                .iter()
                .map(|x| x.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum())
                .collect();
            es.tell(&fits);
        }
        let err: f64 = es
            .mean
            .iter()
            .zip(&target)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(err < 1e-2, "did not converge: mean={:?}", es.mean);
    }

    #[test]
    fn serde_round_trip() {
        let es = CmaEs::new(4, 0.3, 0, 1);
        let json = serde_json::to_string(&es).unwrap();
        let mut back: CmaEs = serde_json::from_str(&json).unwrap();
        back.after_load();
        assert_eq!(back.n, 4);
        assert_eq!(back.mean.len(), 4);
    }
}
