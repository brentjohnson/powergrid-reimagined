use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlantKind, PlayerId, Resource},
    GameState,
};

use crate::{card_painter, state::player_color_to_egui, state::AppState, theme, ws::WsChannels};

use super::helpers::{neon_button, resource_counter_row, resource_name, send};

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

                    if state.bid_plant_number != Some(bid.plant_number) {
                        state.bid_plant_number = Some(bid.plant_number);
                        state.bid_amount = min_bid;
                    }
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

        Phase::DiscardResource {
            player, drop_total, ..
        } => {
            if *player == my_id {
                let drop_total = *drop_total;
                let (coal_held, oil_held) = gs
                    .player(my_id)
                    .map(|p| (p.resources.coal, p.resources.oil))
                    .unwrap_or((0, 0));

                ui.label(
                    RichText::new(format!(
                        "Hybrid plants can't hold all your fuel — discard {} total:",
                        drop_total
                    ))
                    .color(theme::NEON_AMBER)
                    .monospace(),
                );
                ui.add_space(4.0);

                // Cap each counter at min(held, drop_total - other_selection) so
                // coal+oil cannot exceed the required total.
                let coal_max = coal_held.min(drop_total - state.discard_oil);
                match resource_counter_row(
                    ui,
                    "    COAL",
                    state.discard_coal,
                    0,
                    coal_max,
                    &format!("held: {}", coal_held),
                ) {
                    d if d < 0 => state.discard_coal -= 1,
                    d if d > 0 => state.discard_coal += 1,
                    _ => {}
                }

                let oil_max = oil_held.min(drop_total - state.discard_coal);
                match resource_counter_row(
                    ui,
                    "     OIL",
                    state.discard_oil,
                    0,
                    oil_max,
                    &format!("held: {}", oil_held),
                ) {
                    d if d < 0 => state.discard_oil -= 1,
                    d if d > 0 => state.discard_oil += 1,
                    _ => {}
                }

                let selected = state.discard_coal + state.discard_oil;
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("{} / {} selected", selected, drop_total))
                        .color(if selected == drop_total {
                            theme::NEON_GREEN
                        } else {
                            theme::TEXT_DIM
                        })
                        .monospace(),
                );

                if ui
                    .add_enabled(
                        selected == drop_total,
                        neon_button("[ CONFIRM ]", theme::NEON_CYAN),
                    )
                    .clicked()
                {
                    send(
                        Action::DiscardResource {
                            coal: state.discard_coal,
                            oil: state.discard_oil,
                        },
                        channels,
                    );
                }
            } else {
                let name = gs.player(*player).map(|p| p.name.as_str()).unwrap_or("???");
                ui.label(
                    RichText::new(format!("● Waiting for {} to discard fuel…", name))
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

                    match resource_counter_row(
                        ui,
                        &format!("{:>8}", resource_name(resource)),
                        count,
                        0,
                        u8::MAX,
                        &cap_str,
                    ) {
                        d if d < 0 => state.remove_from_cart(resource),
                        d if d > 0 => state.add_to_cart(resource),
                        _ => {}
                    }
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

        Phase::PowerCitiesFuel {
            player,
            hybrid_cost,
            plant_numbers,
            ..
        } => {
            if *player == my_id {
                let hybrid_cost = *hybrid_cost;
                let (coal_held, oil_held) = gs
                    .player(my_id)
                    .map(|p| (p.resources.coal, p.resources.oil))
                    .unwrap_or((0, 0));

                // Compute how much pure-fuel pure plants in the chosen subset need.
                let (pure_coal, pure_oil) = gs
                    .player(my_id)
                    .map(|p| {
                        plant_numbers.iter().fold((0u8, 0u8), |(pc, po), &num| {
                            if let Some(pl) = p.plants.iter().find(|pl| pl.number == num) {
                                match pl.kind {
                                    PlantKind::Coal => (pc + pl.cost, po),
                                    PlantKind::Oil => (pc, po + pl.cost),
                                    _ => (pc, po),
                                }
                            } else {
                                (pc, po)
                            }
                        })
                    })
                    .unwrap_or((0, 0));

                let coal_avail = coal_held.saturating_sub(pure_coal);
                let oil_avail = oil_held.saturating_sub(pure_oil);

                // Clamp coal selection to valid range.
                let min_coal = hybrid_cost.saturating_sub(oil_avail);
                let max_coal = hybrid_cost.min(coal_avail);
                if state.power_fuel_coal < min_coal {
                    state.power_fuel_coal = min_coal;
                }
                if state.power_fuel_coal > max_coal {
                    state.power_fuel_coal = max_coal;
                }
                let oil_used = hybrid_cost - state.power_fuel_coal;

                ui.label(
                    RichText::new(format!(
                        "Hybrid plants need {} fuel — choose coal/oil split:",
                        hybrid_cost
                    ))
                    .color(theme::NEON_AMBER)
                    .monospace(),
                );
                ui.add_space(4.0);

                match resource_counter_row(
                    ui,
                    "    COAL",
                    state.power_fuel_coal,
                    min_coal,
                    max_coal,
                    &format!("avail: {}", coal_avail),
                ) {
                    d if d < 0 => state.power_fuel_coal -= 1,
                    d if d > 0 => state.power_fuel_coal += 1,
                    _ => {}
                }

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("     OIL: {:>2}", oil_used))
                            .color(theme::TEXT_BRIGHT)
                            .monospace(),
                    );
                    ui.label(
                        RichText::new(format!("avail: {}", oil_avail))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                });

                ui.add_space(4.0);
                if ui
                    .add(neon_button("[ CONFIRM ]", theme::NEON_CYAN))
                    .clicked()
                {
                    send(
                        Action::PowerCitiesFuel {
                            coal: state.power_fuel_coal,
                            oil: oil_used,
                        },
                        channels,
                    );
                }
            } else {
                let name = gs.player(*player).map(|p| p.name.as_str()).unwrap_or("???");
                ui.label(
                    RichText::new(format!("● Waiting for {} to choose fuel split…", name))
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
