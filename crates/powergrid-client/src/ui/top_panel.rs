use egui::{Align2, Color32, FontId, Rect, RichText, Sense, Stroke, StrokeKind, Ui};
use powergrid_core::{
    actions::{Action, HintPayload},
    income_for, price_table,
    rules::replenishment_amounts,
    types::{Phase, PlayerColor, PlayerId, Resource, ResourceMarket},
    GameStateView,
};
use std::collections::HashMap;

use crate::{
    card_painter,
    state::{player_color_to_egui, AppState, CitySnapshot},
    theme,
    ws::WsChannels,
};

use super::helpers::{dim_color, send};
use super::player_summary::player_summary;

pub(super) fn top_panel_contents(
    ui: &mut Ui,
    gs: GameStateView,
    state: &mut AppState,
    channels: Option<&WsChannels>,
    my_id: PlayerId,
) {
    let room = state.current_room.clone();
    let room = room.as_deref();
    let my_buy_turn = matches!(&gs.phase, Phase::BuyResources { remaining }
        if remaining.first() == Some(&my_id));

    let is_auction = matches!(
        &gs.phase,
        Phase::Auction { .. } | Phase::DiscardPlant { .. }
    );
    let is_buy = matches!(
        &gs.phase,
        Phase::BuyResources { .. } | Phase::DiscardResource { .. }
    );
    let is_build = matches!(&gs.phase, Phase::BuildCities { .. });
    let is_bureau = matches!(
        &gs.phase,
        Phase::Bureaucracy { .. } | Phase::PowerCitiesFuel { .. }
    );

    ui.horizontal_top(|ui| {
        // ── Phase 1: Determine Player Order ───────────────────────────────
        ui.vertical(|ui| {
            phase_col_header(ui, "DETERMINE ORDER", false, &gs, PhaseKind::DetermineOrder);
            theme::neon_frame_bright().show(ui, |ui| {
                ui.label(
                    RichText::new(format!("ROUND {}", gs.round))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
            });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Phase 2: Auction Power Plants ─────────────────────────────────
        let auction_col = ui.vertical(|ui| {
            phase_col_header(ui, "AUCTION PLANTS", is_auction, &gs, PhaseKind::Auction);
            theme::neon_frame().show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let discount_token = gs.market.discount_token;
                    if gs.step >= 3 {
                        let mid = gs.market.actual.len().div_ceil(2);
                        let (left, right) = gs.market.actual.split_at(mid);
                        ui.vertical(|ui| {
                            plant_column(
                                ui,
                                left,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                        });
                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            plant_column(
                                ui,
                                right,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                        });
                    } else {
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new("ACTUAL")
                                    .color(theme::TEXT_DIM)
                                    .small()
                                    .monospace(),
                            );
                            plant_column(
                                ui,
                                &gs.market.actual,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                discount_token,
                            );
                        });
                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new("FUTURE")
                                    .color(theme::TEXT_DIM)
                                    .small()
                                    .monospace(),
                            );
                            plant_column(
                                ui,
                                &gs.market.future,
                                channels,
                                &gs.phase,
                                my_id,
                                &gs.player_order,
                                room,
                                None, // future market never holds the discount token
                            );
                        });
                    }
                });
            });
        });
        state.phase_column_rects[0] = Some(auction_col.response.rect);

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Phase 3: Buy Resources ─────────────────────────────────────────
        let buy_col = ui.vertical(|ui| {
            phase_col_header(ui, "BUY RESOURCES", is_buy, &gs, PhaseKind::BuyResources);

            let cart_snapshot = state.resource_cart.clone();
            let peer_carts: Vec<(Color32, HashMap<Resource, u8>)> = state
                .peer_hints
                .hints
                .iter()
                .filter_map(|(pid, hint)| {
                    if let HintPayload::Cart { items } = hint {
                        let color = gs
                            .player(*pid)
                            .map(|p| player_color_to_egui(p.color))
                            .unwrap_or(Color32::GRAY);
                        let cart: HashMap<Resource, u8> = items.iter().cloned().collect();
                        Some((color, cart))
                    } else {
                        None
                    }
                })
                .collect();

            let replenish = if matches!(&gs.phase, Phase::Lobby | Phase::GameOver { .. }) {
                (0, 0, 0, 0)
            } else {
                replenishment_amounts(gs.step, gs.players.len())
            };
            let click = theme::neon_frame().show(ui, |ui| {
                resource_market_grid(
                    ui,
                    &gs.resources,
                    &cart_snapshot,
                    &peer_carts,
                    my_buy_turn,
                    replenish,
                )
            });
            if let Some((resource, amount)) = click.inner {
                state.set_cart_amount(resource, amount);
            }
        });
        state.phase_column_rects[1] = Some(buy_col.response.rect);

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Phase 4: Build Generators ──────────────────────────────────────
        let build_col = ui.vertical(|ui| {
            phase_col_header(
                ui,
                "BUILD GENERATORS",
                is_build,
                &gs,
                PhaseKind::BuildCities,
            );
            theme::neon_frame().show(ui, |ui| {
                city_count_list(ui, &gs);
            });
        });
        state.phase_column_rects[2] = Some(build_col.response.rect);

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Phase 5: Bureaucracy ───────────────────────────────────────────
        let bureau_col = ui.vertical(|ui| {
            phase_col_header(ui, "BUREAUCRACY", is_bureau, &gs, PhaseKind::Bureaucracy);
            theme::neon_frame().show(ui, |ui| {
                player_summary(ui, &gs, my_id);
            });
        });
        state.phase_column_rects[3] = Some(bureau_col.response.rect);
    });
}

#[derive(Clone, Copy, PartialEq)]
enum PhaseKind {
    DetermineOrder,
    Auction,
    BuyResources,
    BuildCities,
    Bureaucracy,
}

fn phase_col_header(ui: &mut Ui, label: &str, active: bool, gs: &GameStateView, kind: PhaseKind) {
    let color = if active {
        theme::NEON_AMBER
    } else {
        theme::TEXT_DIM
    };
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .color(color)
                .small()
                .monospace()
                .strong(),
        );
        if kind != PhaseKind::DetermineOrder {
            ui.add_space(4.0);
            phase_turn_dots(ui, gs, kind);
        }
    });
    ui.add_space(2.0);
}

fn phase_turn_dots(ui: &mut Ui, gs: &GameStateView, kind: PhaseKind) {
    #[derive(Clone, Copy, PartialEq)]
    enum Dp {
        Auction,
        Resource,
        Build,
        Bureaucracy,
    }

    let phase_idx = |d: Dp| -> u8 {
        match d {
            Dp::Auction => 0,
            Dp::Resource => 1,
            Dp::Build => 2,
            Dp::Bureaucracy => 3,
        }
    };

    let current_dp = match &gs.phase {
        Phase::Auction { .. } | Phase::DiscardPlant { .. } => Some(Dp::Auction),
        Phase::BuyResources { .. } | Phase::DiscardResource { .. } => Some(Dp::Resource),
        Phase::BuildCities { .. } => Some(Dp::Build),
        Phase::Bureaucracy { .. } | Phase::PowerCitiesFuel { .. } => Some(Dp::Bureaucracy),
        _ => None,
    };

    let dp = match kind {
        PhaseKind::Auction => Dp::Auction,
        PhaseKind::BuyResources => Dp::Resource,
        PhaseKind::BuildCities => Dp::Build,
        PhaseKind::Bureaucracy => Dp::Bureaucracy,
        PhaseKind::DetermineOrder => return,
    };

    let is_current = current_dp == Some(dp);
    let is_past = current_dp.is_some_and(|cd| phase_idx(dp) < phase_idx(cd));

    // Auction runs in forward order; all other phases run in reverse (fewest cities acts first)
    let player_ids: Vec<PlayerId> = if dp == Dp::Auction {
        gs.player_order.clone()
    } else {
        gs.player_order.iter().rev().cloned().collect()
    };

    let phase_active: Option<PlayerId> = if !is_current {
        None
    } else {
        match &gs.phase {
            Phase::Auction {
                current_bidder_idx, ..
            } => gs.player_order.get(*current_bidder_idx).copied(),
            Phase::BuyResources { remaining } | Phase::BuildCities { remaining } => {
                remaining.first().copied()
            }
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
                        Phase::BuyResources { remaining }
                        | Phase::BuildCities { remaining }
                        | Phase::Bureaucracy { remaining } => !remaining.contains(pid),
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

fn city_count_list(ui: &mut Ui, gs: &GameStateView) {
    // Display in reverse auction order (fewest cities → builds first)
    for pid in gs.player_order.iter().rev() {
        if let Some(p) = gs.player(*pid) {
            let is_building = matches!(&gs.phase,
                Phase::BuildCities { remaining } if remaining.first() == Some(pid));
            let base = player_color_to_egui(p.color);
            let color = if is_building { base } else { dim_color(base) };
            ui.horizontal(|ui| {
                ui.label(RichText::new("■").color(color).monospace());
                ui.label(
                    RichText::new(format!(
                        "{}: {}/{}",
                        p.name,
                        gs.player_city_count(p.id),
                        gs.end_game_cities
                    ))
                    .color(color)
                    .small()
                    .monospace(),
                );
            });
        }
    }
}

// ── Plant market helpers ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn plant_column(
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

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        for plant in plants {
            let discounted = discount_token == Some(plant.number);
            let nominated = nominated_number == Some(plant.number);
            let resp = card_painter::draw_plant_card_ex(ui, plant, discounted, nominated);
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

// ── Step/replenish table ───────────────────────────────────────────────────────

pub(super) fn replenish_rates(step: u8, n: usize) -> (u8, u8, u8, u8) {
    match step {
        1 => match n {
            2 => (3, 2, 1, 1),
            3 => (4, 2, 1, 1),
            4 => (5, 3, 2, 1),
            5 => (5, 4, 3, 2),
            _ => (7, 5, 3, 2),
        },
        2 => match n {
            2 => (4, 2, 1, 1),
            3 => (5, 3, 2, 1),
            4 => (6, 4, 3, 2),
            5 => (7, 5, 3, 3),
            _ => (9, 6, 5, 3),
        },
        _ => match n {
            2 => (3, 4, 3, 1),
            3 => (3, 4, 3, 1),
            4 => (4, 5, 4, 2),
            5 => (5, 6, 5, 3),
            _ => (7, 7, 6, 3),
        },
    }
}

pub(super) fn step_replenish_columns(ui: &mut Ui, current_step: u8, n_players: usize) {
    let coal_color = theme::RES_COAL;
    let oil_color = theme::RES_OIL;
    let gas_color = theme::RES_GAS;
    let uran_color = theme::RES_URANIUM;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        for step in 1u8..=3 {
            let (coal, oil, gas, uran) = replenish_rates(step, n_players);
            let active = step == current_step;
            let hdr = if active {
                theme::NEON_CYAN
            } else {
                theme::TEXT_DIM
            };
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(
                    RichText::new(format!("{step}"))
                        .color(hdr)
                        .monospace()
                        .small(),
                );
                ui.label(
                    RichText::new(format!("{coal}"))
                        .color(coal_color)
                        .monospace()
                        .small(),
                );
                ui.label(
                    RichText::new(format!("{gas}"))
                        .color(gas_color)
                        .monospace()
                        .small(),
                );
                ui.label(
                    RichText::new(format!("{oil}"))
                        .color(oil_color)
                        .monospace()
                        .small(),
                );
                ui.label(
                    RichText::new(format!("{uran}"))
                        .color(uran_color)
                        .monospace()
                        .small(),
                );
            });
        }
    });
}

// ── Resource market grid ───────────────────────────────────────────────────────

fn resource_market_grid(
    ui: &mut Ui,
    market: &ResourceMarket,
    cart: &HashMap<Resource, u8>,
    peer_carts: &[(Color32, HashMap<Resource, u8>)],
    clickable: bool,
    replenish: (u8, u8, u8, u8),
) -> Option<(Resource, u8)> {
    const SQ: f32 = 14.0;
    const INNER_GAP: f32 = 2.0;
    const GROUP_GAP: f32 = 8.0;
    const LABEL_W: f32 = 36.0;
    const HEADER_H: f32 = 20.0;
    const ROW_H: f32 = SQ;
    const ROW_GAP: f32 = 4.0;

    let rows: &[(Resource, &str, Color32)] = &[
        (Resource::Coal, "COAL", theme::RES_COAL),
        (Resource::Gas, "GAS", theme::RES_GAS),
        (Resource::Oil, "OIL", theme::RES_OIL),
        (Resource::Uranium, "URAN", theme::RES_URANIUM),
    ];

    let resource_groups: Vec<Vec<(u8, usize)>> = rows
        .iter()
        .map(|(r, _, _)| {
            let mut groups: Vec<(u8, usize)> = Vec::new();
            for &p in price_table(*r).iter().rev() {
                match groups.last_mut() {
                    Some(last) if last.0 == p => last.1 += 1,
                    _ => groups.push((p, 1)),
                }
            }
            groups
        })
        .collect();

    let mut all_prices: Vec<u8> = resource_groups
        .iter()
        .flat_map(|groups| groups.iter().map(|&(p, _)| p))
        .collect();
    all_prices.sort_unstable();
    all_prices.dedup();

    let col_widths: Vec<usize> = all_prices
        .iter()
        .map(|&p| {
            resource_groups
                .iter()
                .filter_map(|groups| groups.iter().find(|&&(gp, _)| gp == p).map(|&(_, gs)| gs))
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut col_x: Vec<f32> = Vec::with_capacity(all_prices.len());
    let mut x = 0.0f32;
    for (i, &w) in col_widths.iter().enumerate() {
        col_x.push(x);
        let col_w = w as f32 * (SQ + INNER_GAP) - INNER_GAP;
        x += col_w;
        if i + 1 < col_widths.len() {
            x += GROUP_GAP;
        }
    }
    let content_w = x;

    let total_w = LABEL_W + content_w;
    let n = rows.len() as f32;
    let total_h = HEADER_H + ROW_GAP + n * ROW_H + (n - 1.0) * ROW_GAP;

    let sense = if clickable {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(total_w, total_h), sense);

    if !ui.is_rect_visible(rect) {
        return None;
    }

    let painter = ui.painter();
    let ox = rect.min.x;
    let oy = rect.min.y;

    for (col_idx, (&price, &w)) in all_prices.iter().zip(col_widths.iter()).enumerate() {
        let gx = ox + LABEL_W + col_x[col_idx];
        let col_w = w as f32 * (SQ + INNER_GAP) - INNER_GAP;
        painter.text(
            egui::pos2(gx + col_w / 2.0, oy + HEADER_H / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(10.0),
            theme::TEXT_DIM,
        );
    }

    let mut click_result: Option<(Resource, u8)> = None;
    let clicked_pos = if response.clicked() {
        response.interact_pointer_pos()
    } else {
        None
    };

    for (row_idx, ((resource, label, color), rgroups)) in
        rows.iter().zip(resource_groups.iter()).enumerate()
    {
        let row_y = oy + HEADER_H + ROW_GAP + row_idx as f32 * (ROW_H + ROW_GAP);
        let count = market.available(*resource) as usize;
        let total = price_table(*resource).len();
        let cart_amount = cart.get(resource).copied().unwrap_or(0) as usize;
        let cheapest_filled = total.saturating_sub(count);
        let replenish_amount = match resource {
            Resource::Coal => replenish.0,
            Resource::Oil => replenish.1,
            Resource::Gas => replenish.2,
            Resource::Uranium => replenish.3,
        } as usize;

        painter.text(
            egui::pos2(ox + LABEL_W - 2.0, row_y + ROW_H / 2.0),
            Align2::RIGHT_CENTER,
            *label,
            FontId::monospace(10.0),
            *color,
        );

        let mut display_pos = 0usize;
        for (col_idx, &price) in all_prices.iter().enumerate() {
            let group_size = rgroups
                .iter()
                .find(|&&(p, _)| p == price)
                .map_or(0, |&(_, gs)| gs);
            let gx = ox + LABEL_W + col_x[col_idx];

            for s in 0..group_size {
                let dp = display_pos + s;
                let array_idx = total - 1 - dp;
                let filled = array_idx < count;
                let in_cart = filled && dp >= cheapest_filled && dp < cheapest_filled + cart_amount;

                let sq_x = gx + s as f32 * (SQ + INNER_GAP);
                let sq_rect = Rect::from_min_size(egui::pos2(sq_x, row_y), egui::vec2(SQ, ROW_H));

                if in_cart {
                    painter.rect_stroke(sq_rect, 1.0, Stroke::new(1.5, *color), StrokeKind::Inside);
                } else if filled {
                    painter.rect_filled(sq_rect, 1.0, *color);
                } else {
                    painter.rect_filled(sq_rect, 1.0, dim_color(*color));
                }

                if dp < cheapest_filled && (cheapest_filled - dp) <= replenish_amount {
                    painter.circle_filled(sq_rect.center(), 2.5, *color);
                }

                for (peer_color, peer_cart) in peer_carts {
                    let peer_amount = peer_cart.get(resource).copied().unwrap_or(0) as usize;
                    let peer_in_cart =
                        filled && dp >= cheapest_filled && dp < cheapest_filled + peer_amount;
                    if peer_in_cart {
                        painter.rect_stroke(
                            sq_rect.expand(1.5),
                            2.0,
                            Stroke::new(1.0, *peer_color),
                            StrokeKind::Outside,
                        );
                    }
                }

                if let Some(pos) = clicked_pos {
                    if pos.y >= row_y && pos.y < row_y + ROW_H && pos.x >= sq_x && pos.x < sq_x + SQ
                    {
                        let amount = if filled {
                            (dp.saturating_sub(cheapest_filled) + 1) as u8
                        } else {
                            0u8
                        };
                        click_result = Some((*resource, amount));
                    }
                }
            }
            display_pos += group_size;
        }
    }

    click_result
}

// ── City history graph (used by the CITIES popup window in mod.rs) ─────────────

pub(super) fn city_history_graph(
    ui: &mut Ui,
    history: &[CitySnapshot],
    players_info: &[(PlayerId, PlayerColor)],
    end_game_cities: u8,
    gs: &GameStateView,
) {
    const PAD_L: f32 = 26.0;
    const PAD_B: f32 = 18.0;
    const H: f32 = 342.0;
    const DOT_R: f32 = 4.0;
    const STEP2_CITIES: usize = 7;

    let w = (ui.available_width() - PAD_L).max(100.0);
    let total_w = PAD_L + w;
    let total_h = PAD_B + H;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(total_w, total_h), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }

    let painter = ui.painter();
    let ox = rect.min.x + PAD_L;
    let oy = rect.min.y;

    let max_cities = history
        .iter()
        .flat_map(|snap| snap.iter().map(|(_, c)| *c))
        .chain(gs.players.iter().map(|p| gs.player_city_count(p.id)))
        .max()
        .unwrap_or(1)
        .max(end_game_cities as usize)
        .max(1);

    let rounds = history.len();
    let total_points = rounds + 1;
    let x_for = |idx: usize| -> f32 {
        if total_points <= 1 {
            ox
        } else {
            ox + (idx as f32 / (total_points - 1) as f32) * w
        }
    };

    painter.line_segment(
        [egui::pos2(ox, oy), egui::pos2(ox, oy + H)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );
    painter.line_segment(
        [egui::pos2(ox, oy + H), egui::pos2(ox + w, oy + H)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );

    painter.text(
        egui::pos2(ox - 2.0, oy),
        Align2::RIGHT_TOP,
        format!("{max_cities}"),
        FontId::monospace(13.0),
        theme::TEXT_DIM,
    );
    painter.text(
        egui::pos2(ox - 2.0, oy + H),
        Align2::RIGHT_BOTTOM,
        "0",
        FontId::monospace(13.0),
        theme::TEXT_DIM,
    );

    painter.text(
        egui::pos2(ox, oy + H + PAD_B),
        Align2::LEFT_BOTTOM,
        "1",
        FontId::monospace(13.0),
        theme::TEXT_DIM,
    );
    if rounds > 1 {
        painter.text(
            egui::pos2(ox + w, oy + H + PAD_B),
            Align2::RIGHT_BOTTOM,
            format!("{}", rounds + 1),
            FontId::monospace(13.0),
            theme::TEXT_DIM,
        );
    }

    let step2_y = oy + H - (STEP2_CITIES as f32 / max_cities as f32) * H;
    let step2_color = theme::city_graph_step2();
    let dash_len = 4.0_f32;
    let gap_len = 3.0_f32;
    let mut x = ox;
    while x < ox + w {
        let x_end = (x + dash_len).min(ox + w);
        painter.line_segment(
            [egui::pos2(x, step2_y), egui::pos2(x_end, step2_y)],
            Stroke::new(1.0, step2_color),
        );
        x += dash_len + gap_len;
    }
    painter.text(
        egui::pos2(ox - 2.0, step2_y),
        Align2::RIGHT_CENTER,
        "S2",
        FontId::monospace(11.0),
        step2_color,
    );

    let end_y = oy + H - (end_game_cities as f32 / max_cities as f32) * H;
    let end_color = theme::city_graph_end();
    let mut x = ox;
    while x < ox + w {
        let x_end = (x + dash_len).min(ox + w);
        painter.line_segment(
            [egui::pos2(x, end_y), egui::pos2(x_end, end_y)],
            Stroke::new(1.0, end_color),
        );
        x += dash_len + gap_len;
    }
    painter.text(
        egui::pos2(ox - 2.0, end_y),
        Align2::RIGHT_CENTER,
        "E",
        FontId::monospace(11.0),
        end_color,
    );

    for (player_id, player_color) in players_info {
        let color = player_color_to_egui(*player_color);

        let points: Vec<egui::Pos2> = history
            .iter()
            .enumerate()
            .filter_map(|(round_idx, snap)| {
                snap.iter()
                    .find(|(id, _)| id == player_id)
                    .map(|(_, count)| {
                        let x = x_for(round_idx);
                        let y = oy + H - (*count as f32 / max_cities as f32) * H;
                        egui::pos2(x, y)
                    })
            })
            .collect();

        for pair in points.windows(2) {
            painter.line_segment([pair[0], pair[1]], Stroke::new(2.5, color));
        }

        for pt in &points {
            painter.circle_filled(*pt, DOT_R, color);
        }

        if let Some(&last_pt) = points.last() {
            if let Some(player) = gs.players.iter().find(|p| p.id == *player_id) {
                let proj_count = gs.player_city_count(player.id);
                let proj_x = x_for(rounds);
                let proj_y = oy + H - (proj_count as f32 / max_cities as f32) * H;
                let proj_pt = egui::pos2(proj_x, proj_y);
                let dim = dim_color(color);
                painter.line_segment([last_pt, proj_pt], Stroke::new(2.5, dim));
                painter.circle_filled(proj_pt, DOT_R, dim);
            }
        }
    }
}

// ── City payout table (used by the CITIES popup window in mod.rs) ──────────────

pub(super) fn city_payout_table(ui: &mut Ui, gs: &GameStateView) {
    use crate::state::player_color_to_egui;

    // Compute effective powerable cities per player.
    let highlights: Vec<(u8, egui::Color32)> = gs
        .players
        .iter()
        .map(|p| {
            let city_count = gs.player_city_count(p.id) as u8;
            let (_, max_powered, _) = p.optimal_firing_subset(city_count);
            let effective = max_powered.min(city_count);
            (effective, player_color_to_egui(p.color))
        })
        .collect();

    // Render two columns side-by-side (rows 0-9 left, 10-18 right).
    const MAX_ROW: u8 = 18;
    const SPLIT: u8 = 9;

    let render_col = |ui: &mut Ui, start: u8, end: u8| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(
                RichText::new("  C  $")
                    .color(theme::NEON_CYAN)
                    .monospace()
                    .small(),
            );
            for c in start..=end {
                let income = income_for(c);
                let row_colors: Vec<egui::Color32> = highlights
                    .iter()
                    .filter(|(eff, _)| *eff == c)
                    .map(|(_, col)| *col)
                    .collect();

                let text_color = if row_colors.is_empty() {
                    theme::TEXT_DIM
                } else {
                    row_colors[0]
                };

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    ui.label(
                        RichText::new(format!("{c:>3}{income:>4}"))
                            .color(text_color)
                            .monospace()
                            .small(),
                    );
                    for &col in &row_colors {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(5.0, 5.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 2.5, col);
                    }
                });
            }
        });
    };

    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        render_col(ui, 0, SPLIT);
        ui.add(egui::Separator::default().vertical());
        render_col(ui, SPLIT + 1, MAX_ROW);
    });
}
