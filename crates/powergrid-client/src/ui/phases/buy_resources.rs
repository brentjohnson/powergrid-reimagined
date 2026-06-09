use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerId},
    GameStateView,
};

use crate::{state::AppState, theme, ws::WsChannels};

use super::super::helpers::{neon_button, send};

pub(in crate::ui) fn buy_resources_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let Phase::BuyResources { remaining } = &gs.phase else {
        return;
    };

    let room_owned = state.current_room.clone();
    let room = room_owned.as_deref();

    if remaining.first() == Some(&my_id) {
        let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);
        let player = gs.player(my_id);

        // Per-resource counts/capacity are intentionally omitted here — the
        // resource market overlay already highlights cart contents (filled
        // squares in the resource color) and capacity (lit vs. dimmed squares).
        ui.horizontal(|ui| {
            let has_fuel_plants =
                player.is_some_and(|p| p.plants.iter().any(|pl| pl.kind.needs_resources()));
            if has_fuel_plants {
                if ui.add(neon_button("[ 1 SET ]", theme::NEON_CYAN)).clicked() {
                    state.fill_cart_for_sets(1);
                }
                if ui
                    .add(neon_button("[ 2 SETS ]", theme::NEON_CYAN))
                    .clicked()
                {
                    state.fill_cart_for_sets(2);
                }
            }

            let unaffordable = state.resource_cart_cost.is_some_and(|c| c > my_money);
            if ui
                .add(neon_button("[ CLEAR ]", theme::NEON_AMBER))
                .clicked()
            {
                state.clear_cart();
            }
            if ui
                .add_enabled(
                    !unaffordable,
                    neon_button("[ DONE BUYING ]", theme::NEON_CYAN),
                )
                .clicked()
            {
                let purchases = state.cart_purchases();
                if purchases.is_empty() {
                    send(Action::DoneBuying, room, channels);
                } else {
                    send(Action::BuyResourceBatch { purchases }, room, channels);
                }
            }

            let cost = state.resource_cart_cost.unwrap_or(0);
            let balance = my_money as i64 - cost as i64;
            let balance_color = if balance < 0 {
                theme::NEON_RED
            } else {
                theme::NEON_GREEN
            };
            ui.label(
                RichText::new(format!("TOTAL: ${cost}  BALANCE: ${balance}"))
                    .color(balance_color)
                    .monospace(),
            );
        });
    } else {
        ui.label(
            RichText::new("● Waiting for other operators to buy…")
                .color(theme::TEXT_DIM)
                .monospace(),
        );
    }
}
