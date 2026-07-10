//! Mapping between a CMA-ES search vector and a concrete [`BotProfile`].
//!
//! CMA-ES optimizes a **normalized** vector `x ∈ R^N_PARAMS` initialized at the
//! origin, which corresponds exactly to the shipped `hard` profile. Each
//! coordinate maps to one strategy weight via
//! `actual_i = clamp(init_i + x_i * scale_i, min_i, max_i)`, so `x = 0` reproduces
//! `hard` and `scale_i` sets how many real units one unit of search space spans
//! (keeping every coordinate ~comparably sensitive for the optimizer).
//!
//! Only the 14 *strategy* weights are evolved. `temperature`, `jitter`, and
//! `max_jitter` are deliberately excluded: fitness is measured with those forced
//! to zero (fully deterministic, paired games — see the training journal on why
//! jittered A/B has a ±5pp noise floor), and the shipped champion simply re-uses
//! `hard`'s noise values.

// Parameter tables read clearest as explicit indexed loops over SPECS.
#![allow(clippy::needless_range_loop)]

use powergrid_bot_strategy::BotProfile;

pub const N_PARAMS: usize = 14;

pub struct ParamSpec {
    pub name: &'static str,
    /// Real units spanned by one unit of normalized search space.
    pub scale: f64,
    pub min: f64,
    pub max: f64,
    /// Rounded to the nearest integer before being written to the profile.
    pub integer: bool,
}

const fn spec(name: &'static str, scale: f64, min: f64, max: f64, integer: bool) -> ParamSpec {
    ParamSpec {
        name,
        scale,
        min,
        max,
        integer,
    }
}

/// Order is load-bearing: it must match [`profile_to_raw`] and [`apply_raw`].
pub const SPECS: [ParamSpec; N_PARAMS] = [
    spec("auction.city_reserve", 10.0, 0.0, 100.0, false),
    spec("auction.safety_buffer", 5.0, 0.0, 40.0, false),
    spec("auction.upgrade_margin", 8.0, 0.0, 60.0, false),
    spec("auction.min_open_score", 8.0, 0.0, 60.0, false),
    spec("auction.buildable_lookahead", 1.0, 0.0, 4.0, true),
    spec("auction.endgame_weight", 6.0, 0.0, 50.0, false),
    spec("auction.denial_weight", 0.15, 0.0, 1.5, false),
    spec("auction.operating_cost_weight", 0.3, 0.0, 3.0, false),
    spec("auction.fuel_risk_weight", 1.0, 0.0, 8.0, false),
    spec("auction.replacement_waste_weight", 0.25, 0.0, 3.0, false),
    spec("buy.fuel_reserve_multiplier", 1.0, 0.5, 10.0, false),
    spec("buy.stockpile_rounds", 0.5, 1.0, 5.0, false),
    spec("build.block_weight", 1.0, 0.0, 8.0, false),
    spec("bureaucracy.oil_preference", 0.4, 0.0, 1.0, false),
];

/// Read the 14 evolved weights out of a profile, in [`SPECS`] order.
pub fn profile_to_raw(p: &BotProfile) -> [f64; N_PARAMS] {
    [
        p.auction.city_reserve as f64,
        p.auction.safety_buffer as f64,
        p.auction.upgrade_margin as f64,
        p.auction.min_open_score as f64,
        p.auction.buildable_lookahead as f64,
        p.auction.endgame_weight as f64,
        p.auction.denial_weight as f64,
        p.auction.operating_cost_weight as f64,
        p.auction.fuel_risk_weight as f64,
        p.auction.replacement_waste_weight as f64,
        p.buy.fuel_reserve_multiplier as f64,
        p.buy.stockpile_rounds as f64,
        p.build.block_weight as f64,
        p.bureaucracy.oil_preference as f64,
    ]
}

/// Convert a normalized search vector to real parameter values (clamped/rounded).
pub fn x_to_raw(init_raw: &[f64; N_PARAMS], x: &[f64]) -> [f64; N_PARAMS] {
    let mut raw = *init_raw;
    for i in 0..N_PARAMS {
        let s = &SPECS[i];
        let mut v = init_raw[i] + x[i] * s.scale;
        if s.integer {
            v = v.round();
        }
        raw[i] = v.clamp(s.min, s.max);
    }
    raw
}

/// Write real parameter values onto a clone of `base` (fields not in [`SPECS`],
/// e.g. `display_name`/`temperature`/`jitter`, are inherited from `base`).
pub fn apply_raw(base: &BotProfile, raw: &[f64; N_PARAMS]) -> BotProfile {
    let mut p = base.clone();
    p.auction.city_reserve = raw[0] as f32;
    p.auction.safety_buffer = raw[1] as f32;
    p.auction.upgrade_margin = raw[2] as f32;
    p.auction.min_open_score = raw[3] as f32;
    p.auction.buildable_lookahead = raw[4].round() as u8;
    p.auction.endgame_weight = raw[5] as f32;
    p.auction.denial_weight = raw[6] as f32;
    p.auction.operating_cost_weight = raw[7] as f32;
    p.auction.fuel_risk_weight = raw[8] as f32;
    p.auction.replacement_waste_weight = raw[9] as f32;
    p.buy.fuel_reserve_multiplier = raw[10] as f32;
    p.buy.stockpile_rounds = raw[11] as f32;
    p.build.block_weight = raw[12] as f32;
    p.bureaucracy.oil_preference = raw[13] as f32;
    p
}

/// Build the evaluation-ready profile for a normalized vector: the 14 evolved
/// weights applied on top of `base`, with noise forced off for deterministic,
/// paired games.
pub fn x_to_eval_profile(base: &BotProfile, init_raw: &[f64; N_PARAMS], x: &[f64]) -> BotProfile {
    let raw = x_to_raw(init_raw, x);
    let mut p = apply_raw(base, &raw);
    silence_noise(&mut p);
    p
}

/// Force a profile to play deterministically (argmax, no bid jitter).
pub fn silence_noise(p: &mut BotProfile) {
    p.temperature = 0.0;
    p.jitter = 0.0;
    p.max_jitter = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_bot_strategy::embedded_registry;

    #[test]
    fn origin_reproduces_base_weights() {
        let hard = embedded_registry().hard;
        let init = profile_to_raw(&hard);
        let raw = x_to_raw(&init, &[0.0; N_PARAMS]);
        // x = 0 → every evolved weight equals the base (within f32 round-trip).
        for i in 0..N_PARAMS {
            assert!(
                (raw[i] - init[i]).abs() < 1e-6,
                "param {} drifted at origin: {} vs {}",
                SPECS[i].name,
                raw[i],
                init[i]
            );
        }
    }

    #[test]
    fn round_trip_apply_then_read() {
        let hard = embedded_registry().hard;
        let init = profile_to_raw(&hard);
        let x: Vec<f64> = (0..N_PARAMS).map(|i| 0.1 * i as f64).collect();
        let raw = x_to_raw(&init, &x);
        let prof = apply_raw(&hard, &raw);
        let back = profile_to_raw(&prof);
        for i in 0..N_PARAMS {
            // integer params round; compare after rounding.
            let expect = if SPECS[i].integer {
                raw[i].round()
            } else {
                raw[i]
            };
            assert!(
                (back[i] - expect).abs() < 1e-4,
                "param {} did not round-trip",
                SPECS[i].name
            );
        }
    }

    #[test]
    fn clamps_to_bounds() {
        let hard = embedded_registry().hard;
        let init = profile_to_raw(&hard);
        let big = vec![1e6; N_PARAMS];
        let raw = x_to_raw(&init, &big);
        for i in 0..N_PARAMS {
            assert!(
                raw[i] <= SPECS[i].max + 1e-9,
                "param {} exceeded max",
                SPECS[i].name
            );
        }
        let small = vec![-1e6; N_PARAMS];
        let raw = x_to_raw(&init, &small);
        for i in 0..N_PARAMS {
            assert!(
                raw[i] >= SPECS[i].min - 1e-9,
                "param {} below min",
                SPECS[i].name
            );
        }
    }
}
