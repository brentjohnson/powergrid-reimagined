use egui::{RichText, Ui};
use powergrid_core::{actions::Action, types::Phase, types::PlayerId, GameStateView};

use crate::{state::AppState, theme, ws::WsChannels};

use super::super::helpers::{neon_button, send};

pub(in crate::ui) fn build_cities_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let Phase::BuildCities { remaining } = &gs.phase else {
        return;
    };

    let room_owned = state.current_room.clone();
    let room = room_owned.as_deref();

    if remaining.first() == Some(&my_id) {
        let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);
        ui.label(
            RichText::new("Click cities on map to select build targets.")
                .color(theme::TEXT_BRIGHT)
                .monospace(),
        );

        if !state.selected_build_cities.is_empty() {
            let bp = &state.build_preview;
            let cost_color = if bp.total_cost > my_money {
                theme::NEON_RED
            } else {
                theme::NEON_GREEN
            };
            ui.label(
                RichText::new(format!(
                    "Selected: {}  Route: ${}  Slots: ${}  Total: ${}",
                    state.selected_build_cities.len(),
                    bp.total_route_cost,
                    bp.total_slot_cost,
                    bp.total_cost,
                ))
                .color(cost_color)
                .monospace(),
            );

            let balance = my_money as i64 - bp.total_cost as i64;
            let balance_color = if balance < 0 {
                theme::NEON_RED
            } else {
                theme::NEON_GREEN
            };
            ui.label(
                RichText::new(format!("BALANCE: ${balance}"))
                    .color(balance_color)
                    .monospace(),
            );
        }

        ui.horizontal(|ui| {
            if ui
                .add(neon_button("[ CLEAR ]", theme::NEON_AMBER))
                .clicked()
            {
                state.clear_build_selection();
            }
            if ui
                .add(neon_button("[ DONE BUILDING ]", theme::NEON_CYAN))
                .clicked()
            {
                if state.selected_build_cities.is_empty() {
                    send(Action::DoneBuilding, room, channels);
                } else {
                    let city_ids = state.build_preview.ordered.clone();
                    send(Action::BuildCities { city_ids }, room, channels);
                }
            }
        });
    } else {
        ui.label(
            RichText::new("● Waiting for other operators to build…")
                .color(theme::TEXT_DIM)
                .monospace(),
        );
    }
}
