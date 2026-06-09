use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerId},
    GameStateView,
};

use crate::{
    card_painter,
    state::{player_color_to_egui, AppState},
    theme,
    ws::WsChannels,
};

use super::helpers::{dim_color, send};

/// AUCTION PLANTS overlay — floating top-right, always visible during the game.
/// Shows the plant market (actual + future slots) and captures its rect so the
/// auction action panel can anchor directly below it.
pub(super) fn plant_market_overlay(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let room = state.current_room.clone();
    let room = room.as_deref();

    let is_auction = matches!(
        &gs.phase,
        Phase::Auction { .. } | Phase::DiscardPlant { .. }
    );

    let resp = egui::Area::new(egui::Id::new("plant_market_overlay"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                auction_col_header(ui, is_auction, gs);
                theme::neon_frame().show(ui, |ui| {
                    ui.vertical(|ui| {
                        let discount_token = gs.market.discount_token;
                        if gs.step >= 3 {
                            let (top, bottom) = gs
                                .market
                                .actual
                                .split_at(gs.market.actual.len().div_ceil(2));
                            plant_row(
                                ui,
                                top,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                            ui.add_space(4.0);
                            plant_row(
                                ui,
                                bottom,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                        } else {
                            ui.label(
                                RichText::new("ACTUAL")
                                    .color(theme::TEXT_DIM)
                                    .small()
                                    .monospace(),
                            );
                            plant_row(
                                ui,
                                &gs.market.actual,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new("FUTURE")
                                    .color(theme::TEXT_DIM)
                                    .small()
                                    .monospace(),
                            );
                            plant_row(
                                ui,
                                &gs.market.future,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                None, // future market never holds the discount token
                            );
                        }
                    });
                });
            })
        });

    let rect = resp.response.rect;
    state.phase_column_rects[0] = Some(rect);
    state.plant_market_bottom = rect.bottom();
}

// ── Header with label + auction turn-dots ─────────────────────────────────────

fn auction_col_header(ui: &mut Ui, active: bool, gs: &GameStateView) {
    let color = if active {
        theme::NEON_AMBER
    } else {
        theme::TEXT_DIM
    };
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("AUCTION PLANTS")
                .color(color)
                .small()
                .monospace()
                .strong(),
        );
        ui.add_space(4.0);
        auction_turn_dots(ui, gs);
    });
    ui.add_space(2.0);
}

fn auction_turn_dots(ui: &mut Ui, gs: &GameStateView) {
    // Determine where we are in the turn sequence.
    let current_phase_idx: Option<u8> = match &gs.phase {
        Phase::Auction { .. } | Phase::DiscardPlant { .. } => Some(0),
        Phase::BuyResources { .. } | Phase::DiscardResource { .. } => Some(1),
        Phase::BuildCities { .. } => Some(2),
        Phase::Bureaucracy { .. } | Phase::PowerCitiesFuel { .. } => Some(3),
        _ => None,
    };

    let is_current = current_phase_idx == Some(0);
    let is_past = current_phase_idx.is_some_and(|idx| idx > 0);

    // Auction runs in forward player order.
    let player_ids: Vec<PlayerId> = gs.player_order.clone();

    let phase_active: Option<PlayerId> = if !is_current {
        None
    } else {
        match &gs.phase {
            Phase::Auction {
                current_bidder_idx, ..
            } => gs.player_order.get(*current_bidder_idx).copied(),
            _ => None,
        }
    };

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for pid in &player_ids {
            if let Some(p) = gs.player(*pid) {
                let base = player_color_to_egui(p.color);
                let is_active = phase_active == Some(*pid);
                let is_completed = is_current
                    && match &gs.phase {
                        Phase::Auction { bought, passed, .. } => {
                            bought.contains(pid) || passed.contains(pid)
                        }
                        _ => false,
                    };
                let dimmed = is_past || is_completed;

                let size = egui::Vec2::splat(12.0);
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                if ui.is_rect_visible(rect) {
                    let painter = ui.painter();
                    if is_active {
                        painter.rect_filled(rect, 2.0, dim_color(base));
                        painter.rect_stroke(
                            rect,
                            2.0,
                            egui::Stroke::new(2.0, base),
                            egui::StrokeKind::Outside,
                        );
                    } else {
                        let fill = if dimmed { dim_color(base) } else { base };
                        painter.rect_filled(rect, 2.0, fill);
                    }
                }
            }
        }
    });
}

// ── Plant card rows ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn plant_row(
    ui: &mut Ui,
    plants: &[powergrid_core::types::PowerPlant],
    channels: Option<&WsChannels>,
    phase: &Phase,
    my_id: PlayerId,
    player_order: &[PlayerId],
    room: Option<&str>,
    discount_token: Option<u8>,
) {
    let is_my_auction_turn = matches!(phase, Phase::Auction { current_bidder_idx, active_bid, .. }
        if active_bid.is_none() && player_order.get(*current_bidder_idx) == Some(&my_id));

    let nominated_number = if let Phase::Auction {
        active_bid: Some(bid),
        ..
    } = phase
    {
        Some(bid.plant_number)
    } else {
        None
    };

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for plant in plants {
            let discounted = discount_token == Some(plant.number);
            let nominated = nominated_number == Some(plant.number);
            let resp = card_painter::draw_plant_card_full(ui, plant, discounted, nominated);
            if is_my_auction_turn && resp.clicked() {
                send(
                    Action::SelectPlant {
                        plant_number: plant.number,
                    },
                    room,
                    channels,
                );
            }
            egui::Tooltip::for_enabled(&resp).show(|ui| {
                plant_tooltip(ui, plant, discounted);
            });
        }
    });
}

fn plant_tooltip(ui: &mut Ui, plant: &powergrid_core::types::PowerPlant, discounted: bool) {
    let min_bid_text = if discounted {
        "  Min bid: $1 (discount token)".to_string()
    } else {
        format!("  Min bid: ${}", plant.number)
    };
    ui.label(
        RichText::new(format!(
            "#{} {:?}\nCost: {}  Cities: {}{}",
            plant.number, plant.kind, plant.cost, plant.cities, min_bid_text
        ))
        .monospace()
        .color(theme::TEXT_BRIGHT),
    );
}
