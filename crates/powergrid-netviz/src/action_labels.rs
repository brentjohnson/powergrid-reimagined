//! Human-readable names for the macro action space
//! (`powergrid_bot_strategy::macro_actions`), used by the netviz output panel.

use powergrid_bot_strategy::macro_actions::{
    AUCTION_PASS, AUCTION_RAISE, BUILD_COUNT_BASE, BUILD_DEFAULT, BUY_DEFAULT, BUY_EXTRA_1,
    BUY_EXTRA_2, BUY_FILL, BUY_NOTHING, BUY_SHORT_1, BUY_SHORT_2, DISCARD_PLANT_BASE,
    NOMINATE_BASE, N_BUILD_COUNT, N_DISCARD_PLANT, N_NOMINATE,
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
        BUILD_DEFAULT => "Build:Default".to_string(),
        i if (BUILD_COUNT_BASE..BUILD_COUNT_BASE + N_BUILD_COUNT).contains(&i) => {
            match i - BUILD_COUNT_BASE {
                0 => "Build:Nothing".to_string(),
                n => format!("Build:{n} cheapest"),
            }
        }
        BUY_NOTHING => "Buy:Nothing".to_string(),
        BUY_DEFAULT => "Buy:Full set".to_string(),
        BUY_SHORT_1 => "Buy:Set-1".to_string(),
        BUY_SHORT_2 => "Buy:Set-2".to_string(),
        BUY_EXTRA_1 => "Buy:Set+1".to_string(),
        BUY_EXTRA_2 => "Buy:Set+2".to_string(),
        BUY_FILL => "Buy:Fill storage".to_string(),
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
