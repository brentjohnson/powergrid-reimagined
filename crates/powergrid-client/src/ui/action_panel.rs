use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlantKind, PlayerId, Resource},
    GameState,
};

use crate::{card_painter, state::player_color_to_egui, state::AppState, theme, ws::WsChannels};

use super::helpers::{neon_button, resource_name, send};

pub(super) fn action_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    match &gs.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => {
            let my_nominate_turn = gs.player_order.get(*current_bidder_idx) == Some(&my_id);

            if let Some(bid) = active_bid {
                // Target plant card
                let target_plant = gs
                    .market
                    .actual
                    .iter()
                    .chain(gs.market.future.iter())
                    .find(|p| p.number == bid.plant_number);
                if let Some(plant) = target_plant {
                    card_painter::draw_plant_card(ui, plant);
                    ui.add_space(4.0);
                }

                // Leading bid info
                if let Some(leader) = gs.player(bid.highest_bidder) {
                    let color = player_color_to_egui(leader.color);
                    ui.label(
                        RichText::new(format!("Leading bid: ${} by {}", bid.amount, leader.name))
                            .color(color)
                            .monospace(),
                    );
                }

                // Remaining bidders
                let remaining_names: Vec<&str> = bid
                    .remaining_bidders
                    .iter()
                    .filter_map(|id| gs.player(*id).map(|p| p.name.as_str()))
                    .collect();
                ui.label(
                    RichText::new(format!("Still bidding: {}", remaining_names.join(", ")))
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
                ui.separator();

                let is_my_bid_turn = bid.remaining_bidders.first() == Some(&my_id);
                if is_my_bid_turn {
                    let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);
                    let min_bid = bid.amount + 1;
                    let max_bid = my_money;

                    if state.bid_amount < min_bid {
                        state.bid_amount = min_bid;
                    }
                    if state.bid_amount > max_bid {
                        state.bid_amount = max_bid;
                    }

                    ui.label(
                        RichText::new("Your turn to bid:")
                            .color(theme::TEXT_BRIGHT)
                            .monospace(),
                    );
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                state.bid_amount > min_bid,
                                neon_button("[ - ]", theme::NEON_AMBER),
                            )
                            .clicked()
                        {
                            state.bid_amount -= 1;
                        }
                        ui.label(
                            RichText::new(format!("${}", state.bid_amount))
                                .color(theme::TEXT_BRIGHT)
                                .monospace(),
                        );
                        if ui
                            .add_enabled(
                                state.bid_amount < max_bid,
                                neon_button("[ + ]", theme::NEON_AMBER),
                            )
                            .clicked()
                        {
                            state.bid_amount += 1;
                        }
                        let can_bid = min_bid <= max_bid;
                        if ui
                            .add_enabled(can_bid, neon_button("[ BID ]", theme::NEON_CYAN))
                            .clicked()
                        {
                            send(
                                Action::PlaceBid {
                                    amount: state.bid_amount,
                                },
                                channels,
                            );
                            state.bid_amount = 0;
                        }
                        if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
                            send(Action::PassAuction, channels);
                        }
                    });
                } else {
                    let next_bidder = bid
                        .remaining_bidders
                        .first()
                        .and_then(|id| gs.player(*id))
                        .map(|p| p.name.as_str())
                        .unwrap_or("???");
                    ui.label(
                        RichText::new(format!("● Waiting for {} to bid…", next_bidder))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                }
            } else if my_nominate_turn {
                ui.label(
                    RichText::new("Your turn — select a plant from the market, or pass.")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );
                if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
                    send(Action::PassAuction, channels);
                }
            } else {
                let nominator_name = gs
                    .player_order
                    .get(*current_bidder_idx)
                    .and_then(|id| gs.player(*id))
                    .map(|p| p.name.as_str())
                    .unwrap_or("???");
                ui.label(
                    RichText::new(format!(
                        "● Waiting for {} to select a plant…",
                        nominator_name
                    ))
                    .color(theme::TEXT_DIM)
                    .monospace(),
                );
            }

            // Bought / passed summary
            if !bought.is_empty() || !passed.is_empty() {
                ui.add_space(4.0);
                if !bought.is_empty() {
                    let names: Vec<&str> = bought
                        .iter()
                        .filter_map(|id| gs.player(*id).map(|p| p.name.as_str()))
                        .collect();
                    ui.label(
                        RichText::new(format!("Bought: {}", names.join(", ")))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                }
                if !passed.is_empty() {
                    let names: Vec<&str> = passed
                        .iter()
                        .filter_map(|id| gs.player(*id).map(|p| p.name.as_str()))
                        .collect();
                    ui.label(
                        RichText::new(format!("Passed: {}", names.join(", ")))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                }
            }
        }

        Phase::DiscardPlant {
            player, new_plant, ..
        } => {
            if *player == my_id {
                ui.label(
                    RichText::new(
                        "You won a 4th plant! Choose one of your existing plants to discard:",
                    )
                    .color(theme::NEON_AMBER)
                    .monospace(),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("Incoming: plant #{}", new_plant.number))
                        .color(theme::NEON_GREEN)
                        .monospace(),
                );
                card_painter::draw_plant_card(ui, new_plant);
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Click a plant to discard it:")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );
                if let Some(player_data) = gs.player(my_id) {
                    for plant in &player_data.plants {
                        let resp = card_painter::draw_plant_card(ui, plant);
                        if resp.clicked() {
                            send(
                                Action::DiscardPlant {
                                    plant_number: plant.number,
                                },
                                channels,
                            );
                        }
                    }
                }
            } else {
                let name = gs.player(*player).map(|p| p.name.as_str()).unwrap_or("???");
                ui.label(
                    RichText::new(format!("● Waiting for {} to discard a plant…", name))
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            }
        }

        Phase::BuyResources { remaining } => {
            if remaining.first() == Some(&my_id) {
                let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);
                let player = gs.player(my_id);

                for resource in [
                    Resource::Coal,
                    Resource::Oil,
                    Resource::Garbage,
                    Resource::Uranium,
                ] {
                    let count = state.resource_cart.get(&resource).copied().unwrap_or(0);

                    let (owned, cap_lo, cap_hi) = player
                        .map(|p| {
                            let has_hybrid =
                                p.plants.iter().any(|pl| pl.kind == PlantKind::CoalOrOil);
                            match resource {
                                Resource::Coal | Resource::Oil if has_hybrid => {
                                    let coal_only: u8 = p
                                        .plants
                                        .iter()
                                        .filter(|pl| pl.kind == PlantKind::Coal)
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
                                        .filter(|pl| pl.kind == PlantKind::CoalOrOil)
                                        .map(|pl| pl.cost * 2)
                                        .sum();
                                    let (dedicated, owned) = if resource == Resource::Coal {
                                        (coal_only, p.resources.coal)
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

                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{:>8}: {:>2}", resource_name(resource), count))
                                .color(theme::TEXT_BRIGHT)
                                .monospace(),
                        );
                        if ui.add(neon_button("[-]", theme::NEON_AMBER)).clicked() {
                            state.remove_from_cart(resource);
                        }
                        if ui.add(neon_button("[+]", theme::NEON_GREEN)).clicked() {
                            state.add_to_cart(resource);
                        }
                        ui.label(RichText::new(cap_str).color(theme::TEXT_DIM).monospace());
                    });
                }

                if let Some(cost) = state.resource_cart_cost {
                    let cost_color = if cost > my_money {
                        theme::NEON_RED
                    } else {
                        theme::NEON_GREEN
                    };
                    ui.label(
                        RichText::new(format!("TOTAL: ${cost}  BALANCE: ${my_money}"))
                            .color(cost_color)
                            .monospace(),
                    );
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
                            send(Action::DoneBuying, channels);
                        } else {
                            send(Action::BuyResourceBatch { purchases }, channels);
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

        Phase::BuildCities { remaining } => {
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
                            send(Action::DoneBuilding, channels);
                        } else {
                            let city_ids = state.build_preview.ordered.clone();
                            send(Action::BuildCities { city_ids }, channels);
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

        Phase::Bureaucracy { remaining } => {
            if remaining.contains(&my_id) {
                ui.label(
                    RichText::new("Fire all plants you can to power cities.")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );
                if ui
                    .add(neon_button("[ POWER CITIES ]", theme::NEON_GREEN))
                    .clicked()
                {
                    if let Some(player) = gs.player(my_id) {
                        let plant_numbers: Vec<u8> =
                            player.plants.iter().map(|p| p.number).collect();
                        send(Action::PowerCities { plant_numbers }, channels);
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

        Phase::GameOver { winner } => {
            let name = gs
                .player(*winner)
                .map(|p| p.name.as_str())
                .unwrap_or("UNKNOWN");
            ui.label(
                RichText::new(format!("GRID CONTROLLED BY: {name}"))
                    .size(20.0)
                    .color(theme::NEON_GREEN)
                    .monospace(),
            );
        }

        _ => {}
    }
}
