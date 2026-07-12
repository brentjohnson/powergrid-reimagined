//! Human-readable names for the macro action space
//! (`powergrid_bot_strategy::macro_actions`), used by the netviz output panel.

use powergrid_bot_strategy::macro_actions::{
    AUCTION_PASS, AUCTION_RAISE, BUILD_BLOCK, BUILD_CHEAPEST_1, BUILD_CHEAPEST_2, BUILD_CHEAPEST_3,
    BUILD_DEFAULT, BUILD_MAX, BUILD_NOTHING, BUILD_RACE, BUY_DEFAULT, BUY_DENIAL, BUY_NOTHING,
    BUY_STOCKPILE2, BUY_STOCKPILE3, DISCARD_PLANT_BASE, NOMINATE_BASE, N_DISCARD_PLANT, N_NOMINATE,
    POWER_NOTHING, POWER_OPTIMAL,
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
        BUILD_NOTHING => "Build:Nothing".to_string(),
        BUILD_DEFAULT => "Build:Default".to_string(),
        BUILD_CHEAPEST_1 => "Build:Cheapest1".to_string(),
        BUILD_CHEAPEST_2 => "Build:Cheapest2".to_string(),
        BUILD_CHEAPEST_3 => "Build:Cheapest3".to_string(),
        BUILD_MAX => "Build:Max".to_string(),
        BUILD_BLOCK => "Build:Block".to_string(),
        BUILD_RACE => "Build:Race".to_string(),
        BUY_NOTHING => "Buy:Nothing".to_string(),
        BUY_DEFAULT => "Buy:Default".to_string(),
        BUY_STOCKPILE2 => "Buy:Stockpile2".to_string(),
        BUY_STOCKPILE3 => "Buy:Stockpile3".to_string(),
        BUY_DENIAL => "Buy:Denial".to_string(),
        i if (DISCARD_PLANT_BASE..DISCARD_PLANT_BASE + N_DISCARD_PLANT).contains(&i) => {
            format!("DiscardPlant[slot {}]", i - DISCARD_PLANT_BASE)
        }
        POWER_OPTIMAL => "Power:Optimal".to_string(),
        POWER_NOTHING => "Power:Nothing".to_string(),
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
