//! Human-readable names for the 143-entry action space, mirroring the
//! action-id ranges in `powergrid_bot_strategy::encoding`.

use powergrid_bot_strategy::encoding::{
    BUILD_CITY_BASE, BUY_RESOURCE_BASE, CITY_IDS, DISCARD_PLANT_BASE, DISCARD_RESOURCE_BASE,
    DONE_BUILDING_IDX, DONE_BUYING_IDX, N_ACTIONS, PASS_AUCTION_IDX, PLACE_BID_BASE,
    POWER_CITIES_BASE, POWER_FUEL_BASE, SELECT_PLANT_BASE,
};

const RESOURCE_NAMES: [&str; 4] = ["coal", "oil", "gas", "uranium"];

/// Human-readable label for action id `id` (0..N_ACTIONS).
pub fn action_label(id: usize) -> String {
    match id {
        PASS_AUCTION_IDX => "PassAuction".to_string(),
        DONE_BUYING_IDX => "DoneBuying".to_string(),
        DONE_BUILDING_IDX => "DoneBuilding".to_string(),
        id if (SELECT_PLANT_BASE..PLACE_BID_BASE).contains(&id) => {
            format!("SelectPlant[slot {}]", id - SELECT_PLANT_BASE)
        }
        id if (PLACE_BID_BASE..DISCARD_PLANT_BASE).contains(&id) => {
            format!("PlaceBid[bid+1+{}]", id - PLACE_BID_BASE)
        }
        id if (DISCARD_PLANT_BASE..BUILD_CITY_BASE).contains(&id) => {
            format!("DiscardPlant[slot {}]", id - DISCARD_PLANT_BASE)
        }
        id if (BUILD_CITY_BASE..BUY_RESOURCE_BASE).contains(&id) => {
            format!("BuildCity:{}", CITY_IDS[id - BUILD_CITY_BASE])
        }
        id if (BUY_RESOURCE_BASE..POWER_CITIES_BASE).contains(&id) => {
            format!("BuyResource:{}", RESOURCE_NAMES[id - BUY_RESOURCE_BASE])
        }
        id if (POWER_CITIES_BASE..DISCARD_RESOURCE_BASE).contains(&id) => {
            format!("PowerCities[mask={:03b}]", id - POWER_CITIES_BASE)
        }
        id if (DISCARD_RESOURCE_BASE..POWER_FUEL_BASE).contains(&id) => {
            format!("DiscardResource[gas={}]", id - DISCARD_RESOURCE_BASE)
        }
        id if (POWER_FUEL_BASE..N_ACTIONS).contains(&id) => {
            format!("PowerCitiesFuel[gas={}]", id - POWER_FUEL_BASE)
        }
        _ => format!("unknown[{id}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_action_id_has_a_label() {
        for id in 0..N_ACTIONS {
            let label = action_label(id);
            assert!(!label.starts_with("unknown"), "no label for action {id}");
        }
    }
}
