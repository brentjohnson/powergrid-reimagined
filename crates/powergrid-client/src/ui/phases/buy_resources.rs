use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlantKind, PlayerId, Resource},
    GameStateView,
};

use crate::{state::AppState, theme, ws::WsChannels};

use super::super::helpers::{neon_button, resource_name, send};

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

        for resource in [
            Resource::Coal,
            Resource::Oil,
            Resource::Gas,
            Resource::Uranium,
        ] {
            let count = state.resource_cart.get(&resource).copied().unwrap_or(0);

            let (owned, cap_lo, cap_hi) = player
                .map(|p| {
                    let has_hybrid = p.plants.iter().any(|pl| pl.kind == PlantKind::GasOrOil);
                    match resource {
                        Resource::Gas | Resource::Oil if has_hybrid => {
                            let gas_only: u8 = p
                                .plants
                                .iter()
                                .filter(|pl| pl.kind == PlantKind::Gas)
                                .map(|pl| pl.cost * 2)
                                .sum();
                            let oil_only: u8 = p
                                .plants
                                .iter()
                                .filter(|pl| pl.kind == PlantKind::Oil)
                                .map(|pl| pl.cost * 2)
                                .sum();
                            let hybrid: u8 = p
                                .plants
                                .iter()
                                .filter(|pl| pl.kind == PlantKind::GasOrOil)
                                .map(|pl| pl.cost * 2)
                                .sum();
                            let (dedicated, owned) = if resource == Resource::Gas {
                                (gas_only, p.resources.gas)
                            } else {
                                (oil_only, p.resources.oil)
                            };
                            (owned, dedicated, dedicated + hybrid)
                        }
                        _ => {
                            let cap = p.resource_capacity(resource);
                            (p.resources.get(resource), cap, cap)
                        }
                    }
                })
                .unwrap_or((0, 0, 0));

            let cap_str = if cap_lo == cap_hi {
                format!("{owned}/{cap_hi}")
            } else {
                format!("{owned}/{cap_lo}-{cap_hi}")
            };

            ui.label(
                RichText::new(format!(
                    "{:>8}: {count:>2}  {cap_str}",
                    resource_name(resource)
                ))
                .color(theme::TEXT_BRIGHT)
                .monospace(),
            );
        }

        if let Some(cost) = state.resource_cart_cost {
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
        }

        let has_fuel_plants =
            player.is_some_and(|p| p.plants.iter().any(|pl| pl.kind.needs_resources()));
        if has_fuel_plants {
            ui.horizontal(|ui| {
                if ui.add(neon_button("[ 1 SET ]", theme::NEON_CYAN)).clicked() {
                    state.fill_cart_for_sets(1);
                }
                if ui
                    .add(neon_button("[ 2 SETS ]", theme::NEON_CYAN))
                    .clicked()
                {
                    state.fill_cart_for_sets(2);
                }
            });
        }

        let unaffordable = state.resource_cart_cost.is_some_and(|c| c > my_money);
        ui.horizontal(|ui| {
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
        });
    } else {
        ui.label(
            RichText::new("● Waiting for other operators to buy…")
                .color(theme::TEXT_DIM)
                .monospace(),
        );
    }
}
