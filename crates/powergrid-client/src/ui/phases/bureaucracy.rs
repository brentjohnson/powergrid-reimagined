use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    check_plant_feasibility, income_for,
    types::{Phase, PlantKind, PlayerId},
    GameStateView,
};

use crate::{state::AppState, theme, ws::WsChannels};

use super::super::helpers::{neon_button, send};

pub(in crate::ui) fn bureaucracy_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let Phase::Bureaucracy { remaining } = &gs.phase else {
        return;
    };

    let room_owned = state.current_room.clone();
    let room = room_owned.as_deref();

    if remaining.contains(&my_id) {
        if let Some(player) = gs.player(my_id) {
            if !state.power_selected_initialised {
                let (default_selection, _, _) =
                    player.optimal_firing_subset(gs.player_city_count(my_id) as u8);
                state.power_selected_plants = default_selection.into_iter().collect();
                state.power_selected_initialised = true;
            }

            ui.label(
                RichText::new("Select plants to fire:")
                    .color(theme::TEXT_BRIGHT)
                    .monospace(),
            );
            ui.add_space(4.0);

            for plant in &player.plants {
                let kind_str = match plant.kind {
                    PlantKind::Coal => "coal",
                    PlantKind::Oil => "oil",
                    PlantKind::GasOrOil => "gas/oil",
                    PlantKind::Gas => "gas",
                    PlantKind::Uranium => "uranium",
                    PlantKind::Wind => "wind",
                };
                let city_word = if plant.cities == 1 { "city" } else { "cities" };
                let label = format!(
                    "Plant {:>2}: {} {}  → {} {}",
                    plant.number, plant.cost, kind_str, plant.cities, city_word
                );
                let mut checked = state.power_selected_plants.contains(&plant.number);
                if ui
                    .checkbox(
                        &mut checked,
                        RichText::new(label).monospace().color(theme::TEXT_BRIGHT),
                    )
                    .changed()
                {
                    if checked {
                        state.power_selected_plants.insert(plant.number);
                    } else {
                        state.power_selected_plants.remove(&plant.number);
                    }
                }
            }

            ui.add_space(4.0);

            let selected: Vec<_> = player
                .plants
                .iter()
                .filter(|p| state.power_selected_plants.contains(&p.number))
                .collect();
            let cities_owned = gs.player_city_count(my_id) as u8;
            let feasibility = check_plant_feasibility(&selected, &player.resources);

            match feasibility {
                Some((powered, remaining_res)) => {
                    let capped = powered.min(cities_owned);
                    let income = income_for(capped);
                    ui.label(
                        RichText::new(format!(
                            "Powering {}/{} cities → {} elektro",
                            capped, cities_owned, income
                        ))
                        .monospace()
                        .color(theme::NEON_GREEN),
                    );
                    let spent_coal = player.resources.coal - remaining_res.coal;
                    let spent_oil = player.resources.oil - remaining_res.oil;
                    let spent_gas = player.resources.gas - remaining_res.gas;
                    let spent_uranium = player.resources.uranium - remaining_res.uranium;
                    let mut parts: Vec<String> = Vec::new();
                    if spent_coal > 0 {
                        parts.push(format!("{} coal", spent_coal));
                    }
                    if spent_oil > 0 {
                        parts.push(format!("{} oil", spent_oil));
                    }
                    if spent_gas > 0 {
                        parts.push(format!("{} gas", spent_gas));
                    }
                    if spent_uranium > 0 {
                        parts.push(format!("{} uranium", spent_uranium));
                    }
                    let spend_str = if parts.is_empty() {
                        "Spends: nothing".to_string()
                    } else {
                        format!("Spends: {}", parts.join(", "))
                    };
                    ui.label(RichText::new(spend_str).monospace().color(theme::TEXT_DIM));
                    ui.add_space(4.0);
                    if ui
                        .add(neon_button("[ POWER CITIES ]", theme::NEON_GREEN))
                        .clicked()
                    {
                        let plant_numbers: Vec<u8> =
                            state.power_selected_plants.iter().copied().collect();
                        send(Action::PowerCities { plant_numbers }, room, channels);
                    }
                }
                None => {
                    ui.label(
                        RichText::new("⚠ Selection is infeasible — not enough resources")
                            .monospace()
                            .color(theme::NEON_AMBER),
                    );
                    ui.add_space(4.0);
                    ui.add_enabled(false, neon_button("[ POWER CITIES ]", theme::NEON_GREEN));
                }
            }
        }
    } else {
        ui.label(
            RichText::new("● Waiting for other operators…")
                .color(theme::TEXT_DIM)
                .monospace(),
        );
    }
}
