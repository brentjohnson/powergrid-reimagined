//! Native inference for the RL-trained Expert policy.
//!
//! The network is the policy path of an sb3 MaskablePPO `MlpPolicy`:
//! `obs(OBS_SIZE) → Linear → tanh → Linear → tanh → Linear → logits(N_ACTIONS)`.
//! Weights are exported by `python/scripts/export_policy.py` into a flat
//! little-endian binary (`assets/policies/expert.bin`, embedded at compile
//! time; override at runtime via `RL_POLICY_FILE`).
//!
//! Action selection samples from the masked softmax rather than taking the
//! argmax: the policy was trained and evaluated stochastically, and a greedy
//! pass-everything policy can stall a game forever.

use std::sync::{Arc, OnceLock};

use rand::Rng;
use tracing::warn;

use crate::encoding::{N_ACTIONS, OBS_SIZE};

const MAGIC: &[u8; 8] = b"PGRLPOL1";
const HEADER_LEN: usize = 8 + 3 * 4;

const EMBEDDED_POLICY: &[u8] = include_bytes!("../../../assets/policies/expert.bin");

#[derive(Debug, PartialEq, Eq)]
pub enum PolicyLoadError {
    BadMagic,
    /// File length doesn't match the dimensions declared in the header.
    BadLength {
        expected: usize,
        actual: usize,
    },
    /// Header dimensions don't match the encoding this build was compiled with.
    DimMismatch {
        obs_size: usize,
        n_actions: usize,
    },
}

/// The policy-path MLP. Weights are row-major (torch layout: `weight[out][in]`).
pub struct MlpPolicy {
    obs_size: usize,
    hidden: usize,
    n_actions: usize,
    l1_w: Vec<f32>,
    l1_b: Vec<f32>,
    l2_w: Vec<f32>,
    l2_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

fn read_f32s(bytes: &[u8], count: usize, cursor: &mut usize) -> Vec<f32> {
    let slice = &bytes[*cursor..*cursor + count * 4];
    *cursor += count * 4;
    slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn linear(w: &[f32], b: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(out_dim);
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let sum: f32 = row.iter().zip(x).map(|(wi, xi)| wi * xi).sum();
        y.push(b[o] + sum);
    }
    y
}

impl MlpPolicy {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PolicyLoadError> {
        if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
            return Err(PolicyLoadError::BadMagic);
        }
        let dim = |i: usize| {
            u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize
        };
        let (obs_size, hidden, n_actions) = (dim(0), dim(1), dim(2));
        if obs_size != OBS_SIZE || n_actions != N_ACTIONS {
            return Err(PolicyLoadError::DimMismatch {
                obs_size,
                n_actions,
            });
        }
        let n_params =
            hidden * obs_size + hidden + hidden * hidden + hidden + n_actions * hidden + n_actions;
        let expected = HEADER_LEN + n_params * 4;
        if bytes.len() != expected {
            return Err(PolicyLoadError::BadLength {
                expected,
                actual: bytes.len(),
            });
        }

        let mut cursor = HEADER_LEN;
        Ok(Self {
            obs_size,
            hidden,
            n_actions,
            l1_w: read_f32s(bytes, hidden * obs_size, &mut cursor),
            l1_b: read_f32s(bytes, hidden, &mut cursor),
            l2_w: read_f32s(bytes, hidden * hidden, &mut cursor),
            l2_b: read_f32s(bytes, hidden, &mut cursor),
            out_w: read_f32s(bytes, n_actions * hidden, &mut cursor),
            out_b: read_f32s(bytes, n_actions, &mut cursor),
        })
    }

    /// Forward pass: observation vector → unnormalised action logits.
    pub fn logits(&self, obs: &[f32]) -> Vec<f32> {
        debug_assert_eq!(obs.len(), self.obs_size);
        let mut h = linear(&self.l1_w, &self.l1_b, obs, self.hidden, self.obs_size);
        h.iter_mut().for_each(|v| *v = v.tanh());
        let mut h = linear(&self.l2_w, &self.l2_b, &h, self.hidden, self.hidden);
        h.iter_mut().for_each(|v| *v = v.tanh());
        linear(&self.out_w, &self.out_b, &h, self.n_actions, self.hidden)
    }
}

/// Sample an action index from the softmax over legal (mask = 1) logits,
/// replicating sb3's `MaskableCategorical` at temperature 1.0.
/// Returns `None` when the mask is all-zero.
pub fn sample_masked(logits: &[f32], mask: &[u8], rng: &mut impl Rng) -> Option<usize> {
    let legal_max = logits
        .iter()
        .zip(mask)
        .filter(|(_, &m)| m != 0)
        .map(|(&l, _)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    if legal_max == f32::NEG_INFINITY {
        return None;
    }

    let weights: Vec<f32> = logits
        .iter()
        .zip(mask)
        .map(|(&l, &m)| if m != 0 { (l - legal_max).exp() } else { 0.0 })
        .collect();
    let total: f32 = weights.iter().sum();
    let mut threshold = rng.gen::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        if *w == 0.0 {
            continue;
        }
        threshold -= w;
        if threshold <= 0.0 {
            return Some(i);
        }
    }
    // Float round-off: fall back to the last legal index.
    mask.iter().rposition(|&m| m != 0)
}

/// The Expert policy: embedded weights (or `RL_POLICY_FILE` override), parsed
/// once. `None` — with a warning — if the weights are missing or invalid, in
/// which case Expert bots fall back to the hard heuristic.
pub fn default_policy() -> Option<Arc<MlpPolicy>> {
    static POLICY: OnceLock<Option<Arc<MlpPolicy>>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let (bytes, source) = match std::env::var("RL_POLICY_FILE") {
                Ok(path) => match std::fs::read(&path) {
                    Ok(b) => (b, path),
                    Err(e) => {
                        warn!("failed to read RL_POLICY_FILE {}: {}", path, e);
                        return None;
                    }
                },
                Err(_) => (EMBEDDED_POLICY.to_vec(), "embedded".to_string()),
            };
            match MlpPolicy::from_bytes(&bytes) {
                Ok(p) => Some(Arc::new(p)),
                Err(e) => {
                    warn!("invalid RL policy weights ({}): {:?}", source, e);
                    None
                }
            }
        })
        .clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Build policy bytes for arbitrary dims with the given parameter values.
    fn make_bytes(obs: u32, hidden: u32, actions: u32, params: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::from(*MAGIC);
        bytes.extend(obs.to_le_bytes());
        bytes.extend(hidden.to_le_bytes());
        bytes.extend(actions.to_le_bytes());
        for p in params {
            bytes.extend(p.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        assert_eq!(
            MlpPolicy::from_bytes(b"NOTRIGHT").err().unwrap(),
            PolicyLoadError::BadMagic
        );
        assert_eq!(
            MlpPolicy::from_bytes(&[]).err().unwrap(),
            PolicyLoadError::BadMagic
        );
    }

    #[test]
    fn from_bytes_rejects_wrong_dims() {
        let bytes = make_bytes(10, 4, 5, &[]);
        assert!(matches!(
            MlpPolicy::from_bytes(&bytes).err().unwrap(),
            PolicyLoadError::DimMismatch { .. }
        ));
    }

    #[test]
    fn from_bytes_rejects_truncated_payload() {
        let bytes = make_bytes(OBS_SIZE as u32, 4, N_ACTIONS as u32, &[0.0; 16]);
        assert!(matches!(
            MlpPolicy::from_bytes(&bytes).err().unwrap(),
            PolicyLoadError::BadLength { .. }
        ));
    }

    #[test]
    #[allow(clippy::neg_multiply)] // the literal w*x form mirrors the math being checked
    fn forward_matches_hand_computed_tanh_math() {
        // Tiny net (obs 2, hidden 2, actions 3) checked against by-hand math.
        // Dims bypass from_bytes validation, so construct directly.
        let policy = MlpPolicy {
            obs_size: 2,
            hidden: 2,
            n_actions: 3,
            l1_w: vec![1.0, 0.5, -1.0, 2.0], // rows: [1.0, 0.5], [-1.0, 2.0]
            l1_b: vec![0.1, -0.2],
            l2_w: vec![0.3, -0.4, 0.7, 0.2],
            l2_b: vec![0.0, 0.5],
            out_w: vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5],
            out_b: vec![0.0, 0.1, -0.1],
        };
        let x = [0.5f32, -1.0];
        let h1 = [
            (1.0f32 * 0.5 + 0.5 * -1.0 + 0.1).tanh(),
            (-1.0f32 * 0.5 + 2.0 * -1.0 - 0.2).tanh(),
        ];
        let h2 = [
            (0.3f32 * h1[0] - 0.4 * h1[1]).tanh(),
            (0.7f32 * h1[0] + 0.2 * h1[1] + 0.5).tanh(),
        ];
        let expected = [h2[0], h2[1] + 0.1, 0.5 * h2[0] - 0.5 * h2[1] - 0.1];
        let logits = policy.logits(&x);
        for (got, want) in logits.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn embedded_policy_matches_torch_golden_logits() {
        #[derive(serde::Deserialize)]
        struct Golden {
            obs: Vec<f32>,
            logits: Vec<f32>,
            zeros_logits: Vec<f32>,
        }
        let golden: Golden =
            serde_json::from_str(include_str!("../../../assets/policies/expert.golden.json"))
                .expect("parse expert.golden.json");
        let policy = default_policy().expect("embedded policy must load");

        for (obs, want) in [
            (&golden.obs, &golden.logits),
            (&vec![0.0f32; OBS_SIZE], &golden.zeros_logits),
        ] {
            let got = policy.logits(obs);
            assert_eq!(got.len(), want.len());
            let max_diff = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-3, "logits diverge from torch: {max_diff}");
        }
    }

    #[test]
    fn sample_masked_respects_mask() {
        let mut rng = SmallRng::seed_from_u64(7);
        let logits = vec![5.0, 1.0, -2.0, 3.0];

        assert_eq!(sample_masked(&logits, &[0, 0, 0, 0], &mut rng), None);
        // Single legal action wins regardless of its logit.
        for _ in 0..20 {
            assert_eq!(sample_masked(&logits, &[0, 0, 1, 0], &mut rng), Some(2));
        }
        // Only legal indices are ever sampled.
        for _ in 0..200 {
            let s = sample_masked(&logits, &[1, 0, 0, 1], &mut rng).unwrap();
            assert!(s == 0 || s == 3, "sampled masked-out index {s}");
        }
    }
}
