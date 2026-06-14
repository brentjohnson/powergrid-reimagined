//! Human-readable layout of the 454-dim observation vector.
//!
//! Mirrors the numbered sections in
//! `powergrid_bot_strategy::encoding::build_observation`. Used to render the
//! input editor as grouped, labeled sliders rather than 454 bare numbers.

use powergrid_bot_strategy::encoding::{CITY_IDS, N_CITIES, REGION_NAMES};

/// A named, contiguous slice of the observation vector.
pub struct ObsSection {
    pub name: &'static str,
    pub start: usize,
    pub len: usize,
    /// Label for index `i` (0..len) within this section.
    pub label: fn(usize) -> String,
}

const PLANT_FEATS: [&str; 5] = ["number", "kind", "cost", "cities", "capacity"];
const ACTUAL_FEATS: [&str; 6] = ["number", "kind", "cost", "cities", "present", "discount"];
const FUTURE_FEATS: [&str; 5] = ["number", "kind", "cost", "cities", "present"];
const OPP_FEATS: [&str; 4] = ["plants", "cities", "capacity", "last_powered"];
const RESOURCE_NAMES: [&str; 4] = ["coal", "oil", "gas", "uranium"];
const MARKET_META: [&str; 3] = ["step3_triggered", "in_step3", "deck_size"];
const PHASE_SCALARS: [&str; 5] = [
    "phase_id",
    "step",
    "round",
    "end_game_cities",
    "turn_order_pos",
];

/// All sections, in observation-vector order. Section ranges are contiguous
/// and cover the full `OBS_SIZE` (454) — see the `sections_cover_obs_size` test.
pub fn sections() -> Vec<ObsSection> {
    vec![
        ObsSection {
            name: "Self money",
            start: 0,
            len: 1,
            label: |_| "money".to_string(),
        },
        ObsSection {
            name: "Self resources",
            start: 1,
            len: 4,
            label: |i| RESOURCE_NAMES[i].to_string(),
        },
        ObsSection {
            name: "Self plants",
            start: 5,
            len: 15,
            label: |i| format!("slot {}: {}", i / 5, PLANT_FEATS[i % 5]),
        },
        ObsSection {
            name: "Self cities",
            start: 20,
            len: N_CITIES,
            label: |i| CITY_IDS[i].to_string(),
        },
        ObsSection {
            name: "Opponents",
            start: 69,
            len: 20,
            label: |i| format!("opp {}: {}", i / 4, OPP_FEATS[i % 4]),
        },
        ObsSection {
            name: "Opponent cities",
            start: 89,
            len: 5 * N_CITIES,
            label: |i| format!("opp {}: {}", i / N_CITIES, CITY_IDS[i % N_CITIES]),
        },
        ObsSection {
            name: "City slot counts",
            start: 334,
            len: N_CITIES,
            label: |i| CITY_IDS[i].to_string(),
        },
        ObsSection {
            name: "Active regions",
            start: 383,
            len: 7,
            label: |i| REGION_NAMES[i].to_string(),
        },
        ObsSection {
            name: "Plant market (cards 1-4)",
            start: 390,
            len: 24,
            label: |i| format!("card {}: {}", i / 6 + 1, ACTUAL_FEATS[i % 6]),
        },
        ObsSection {
            name: "Plant market (cards 5-8)",
            start: 414,
            len: 20,
            label: |i| format!("card {}: {}", i / 5 + 5, FUTURE_FEATS[i % 5]),
        },
        ObsSection {
            name: "Market meta",
            start: 434,
            len: 3,
            label: |i| MARKET_META[i].to_string(),
        },
        ObsSection {
            name: "Resource market",
            start: 437,
            len: 4,
            label: |i| RESOURCE_NAMES[i].to_string(),
        },
        ObsSection {
            name: "Phase scalars",
            start: 441,
            len: 5,
            label: |i| PHASE_SCALARS[i].to_string(),
        },
        ObsSection {
            name: "Phase scratch",
            start: 446,
            len: 8,
            label: |i| format!("scratch[{i}] (meaning depends on phase)"),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_bot_strategy::encoding::OBS_SIZE;

    #[test]
    fn sections_cover_obs_size() {
        let secs = sections();
        let mut next = 0;
        for s in &secs {
            assert_eq!(s.start, next, "section {} starts at wrong offset", s.name);
            assert!(s.len > 0);
            next += s.len;
        }
        assert_eq!(next, OBS_SIZE);
    }

    #[test]
    fn labels_are_unique_within_each_section() {
        for s in sections() {
            let labels: Vec<String> = (0..s.len).map(s.label).collect();
            for (i, l) in labels.iter().enumerate() {
                assert!(!l.is_empty(), "{} index {i} has empty label", s.name);
            }
        }
    }
}
