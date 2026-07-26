//! Human-readable names for the macro action space
//! (`powergrid_bot_strategy::macro_actions`), used by the netviz output panel.

use powergrid_bot_strategy::macro_actions::{
    AUCTION_PASS, AUCTION_RAISE, BUILD_COUNT_BASE, BUY_DEFAULT, BUY_DONE, BUY_PLANT1_BASE,
    BUY_PLANT2_BASE, DISCARD_PLANT_BASE, NOMINATE_BASE, N_BUILD_COUNT, N_BUY_PLANT_SLOTS,
    N_DISCARD_PLANT, N_NOMINATE,
};

/// Human-readable label for macro id `id` (0..N_ACTIONS).
pub fn action_label(id: usize) -> String {
    let id = id as u16;
    match id {
        i if (NOMINATE_BASE..NOMINATE_BASE + N_NOMINATE).contains(&i) => {
            format!("Nominate[slot {}]", i - NOMINATE_BASE)
        }
        AUCTION_PASS => "Auction:Pass".to_string(),
        AUCTION_RAISE => "Auction:Raise+1".to_string(),
        i if (BUILD_COUNT_BASE..BUILD_COUNT_BASE + N_BUILD_COUNT).contains(&i) => {
            match i - BUILD_COUNT_BASE {
                0 => "Build:Nothing".to_string(),
                n => format!("Build:{n} cheapest"),
            }
        }
        BUY_DONE => "Buy:Done".to_string(),
        BUY_DEFAULT => "Buy:Default".to_string(),
        i if (BUY_PLANT1_BASE..BUY_PLANT1_BASE + N_BUY_PLANT_SLOTS).contains(&i) => {
            format!("Buy:1 set[plant {}]", i - BUY_PLANT1_BASE)
        }
        i if (BUY_PLANT2_BASE..BUY_PLANT2_BASE + N_BUY_PLANT_SLOTS).contains(&i) => {
            format!("Buy:2 sets[plant {}]", i - BUY_PLANT2_BASE)
        }
        i if (DISCARD_PLANT_BASE..DISCARD_PLANT_BASE + N_DISCARD_PLANT).contains(&i) => {
            format!("DiscardPlant[slot {}]", i - DISCARD_PLANT_BASE)
        }
        _ => format!("unknown[{id}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_bot_strategy::encoding::N_ACTIONS;

    #[test]
    fn every_action_id_has_a_label() {
        for id in 0..N_ACTIONS {
            let label = action_label(id);
            assert!(!label.starts_with("unknown"), "no label for action {id}");
        }
    }
}
