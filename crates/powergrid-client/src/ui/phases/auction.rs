use egui::{RichText, Ui};
use powergrid_core::{actions::Action, types::Phase, types::PlayerId, GameStateView};

use crate::{state::player_color_to_egui, state::AppState, theme, ws::WsChannels};

use super::super::helpers::{dim_color, is_active_player, neon_button, send};

pub(in crate::ui) fn auction_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let Phase::Auction {
        current_bidder_idx,
        active_bid,
        bought,
        passed,
    } = &gs.phase
    else {
        return;
    };

    let room_owned = state.current_room.clone();
    let room = room_owned.as_deref();

    // Per-player status column. During bidding the rotation goes clockwise around
    // the table (seating order, `gs.players`), so order the column that way while a
    // bid is live; otherwise fall back to turn order for the nominate step.
    let column_order: Vec<PlayerId> = if active_bid.is_some() {
        gs.players.iter().map(|p| p.id).collect()
    } else {
        gs.player_order.clone()
    };

    for pid in &column_order {
        if let Some(p) = gs.player(*pid) {
            let is_me = p.id == my_id;
            let active = is_active_player(gs, p.id);
            let player_color = player_color_to_egui(p.color);
            let swatch_color = if active {
                player_color
            } else {
                dim_color(player_color)
            };
            let name_color = if active {
                player_color
            } else {
                dim_color(player_color)
            };

            let (status_text, status_color) = if bought.contains(&p.id) {
                ("PURCHASED".to_string(), theme::NEON_GREEN)
            } else if passed.contains(&p.id) {
                ("PASSED".to_string(), theme::TEXT_DIM)
            } else if let Some(bid) = active_bid {
                let last_bid = state.auction_last_bids.get(&p.id).copied();
                if bid.highest_bidder == p.id {
                    (format!("BID ${}  ◀ leading", bid.amount), theme::NEON_AMBER)
                } else if bid.remaining_bidders.first() == Some(&p.id) {
                    match last_bid {
                        Some(a) => (format!("▶ to bid  ${a}"), theme::NEON_CYAN),
                        None => ("▶ to bid".to_string(), theme::NEON_CYAN),
                    }
                } else if bid.remaining_bidders.contains(&p.id) {
                    match last_bid {
                        Some(a) => (format!("in  ${a}"), theme::TEXT_MID),
                        None => ("in".to_string(), theme::TEXT_MID),
                    }
                } else {
                    ("passed bid".to_string(), theme::TEXT_DIM)
                }
            } else if gs.player_order.get(*current_bidder_idx) == Some(&p.id) {
                ("▶ to nominate".to_string(), theme::NEON_CYAN)
            } else {
                ("—".to_string(), theme::TEXT_DIM)
            };

            ui.horizontal(|ui| {
                ui.label(RichText::new("■").color(swatch_color).monospace());
                ui.label(RichText::new(&p.name).color(name_color).monospace());
                if is_me {
                    ui.label(RichText::new("(you)").color(theme::TEXT_DIM).small());
                }
                ui.label(RichText::new(status_text).color(status_color).monospace());
            });
        }
    }

    ui.add_space(4.0);

    if let Some(bid) = active_bid {
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
                        room,
                        channels,
                    );
                    state.bid_amount = 0;
                }
                if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
                    send(Action::PassAuction, room, channels);
                }
            });
        }
    } else if gs.player_order.get(*current_bidder_idx) == Some(&my_id) {
        ui.label(
            RichText::new("Your turn — select a plant from the market, or pass.")
                .color(theme::TEXT_BRIGHT)
                .monospace(),
        );
        if let Some(tok) = gs.market.discount_token {
            ui.label(
                RichText::new(format!("Plant #{tok} has the discount token — min bid $1."))
                    .color(theme::NEON_CYAN)
                    .monospace()
                    .small(),
            );
        }
        if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
            send(Action::PassAuction, room, channels);
        }
    }
}
