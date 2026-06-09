use egui::{Align2, Color32, FontId, Rect, RichText, Sense, Stroke, StrokeKind, Ui};
use powergrid_core::{
    actions::HintPayload,
    income_for, price_table,
    rules::replenishment_amounts,
    types::{Phase, PlayerColor, PlayerId, Resource, ResourceMarket},
    GameStateView,
};
use std::collections::HashMap;

use crate::{
    state::{player_color_to_egui, AppState, CitySnapshot},
    theme,
};

use super::helpers::{dim_color, resource_image};

// ── Resource market overlay (fixed, bottom-right corner) ──────────────────────

/// Slightly larger than the old "zoomed in" buy-phase scale (1.0) — the market
/// lives in its own corner overlay at a constant size; no more animated zoom
/// between phases.
const MARKET_SCALE: f32 = 1.15;

/// Fixed-size resource market, pinned to the bottom-right corner of the
/// screen. Always visible (every phase), matching the old always-on behavior —
/// just rendered as a floating overlay independent of any panel.
///
/// The market's measured size is stashed in `state.resource_market_width` /
/// `state.resource_market_height` so `buy_cart_panel` can anchor directly above
/// it, matching its width.
pub(super) fn resource_market_overlay(
    ctx: &egui::Context,
    state: &mut AppState,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let my_buy_turn = matches!(&gs.phase, Phase::BuyResources { remaining }
        if remaining.first() == Some(&my_id));

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

    egui::Area::new(egui::Id::new("resource_market_overlay"))
        .anchor(Align2::RIGHT_BOTTOM, egui::vec2(-8.0, -8.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let frame = theme::neon_frame().show(ui, |ui| {
                resource_market_grid(
                    ui,
                    &gs.resources,
                    &cart_snapshot,
                    &peer_carts,
                    my_buy_turn,
                    replenish,
                    MARKET_SCALE,
                )
            });
            state.resource_market_width = frame.response.rect.width();
            state.resource_market_height = frame.response.rect.height();
            if let Some((resource, amount)) = frame.inner {
                state.set_cart_amount(resource, amount);
            }
        });
}

fn resource_market_grid(
    ui: &mut Ui,
    market: &ResourceMarket,
    cart: &HashMap<Resource, u8>,
    peer_carts: &[(Color32, HashMap<Resource, u8>)],
    clickable: bool,
    replenish: (u8, u8, u8, u8),
    scale: f32,
) -> Option<(Resource, u8)> {
    let sq = 22.0 * scale;
    let inner_gap = 3.0 * scale;
    let group_gap = 10.0 * scale;
    let label_w = 44.0 * scale;
    let header_h = 22.0 * scale;
    let row_h = sq;
    let row_gap = 5.0 * scale;
    let font_size = 12.0 * scale;

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
        let col_w = w as f32 * (sq + inner_gap) - inner_gap;
        x += col_w;
        if i + 1 < col_widths.len() {
            x += group_gap;
        }
    }
    let content_w = x;

    let total_w = label_w + content_w;
    let n = rows.len() as f32;
    let total_h = header_h + row_gap + n * row_h + (n - 1.0) * row_gap;

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
        let gx = ox + label_w + col_x[col_idx];
        let col_w = w as f32 * (sq + inner_gap) - inner_gap;
        painter.text(
            egui::pos2(gx + col_w / 2.0, oy + header_h / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(font_size),
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
        let row_y = oy + header_h + row_gap + row_idx as f32 * (row_h + row_gap);
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
            egui::pos2(ox + label_w - 2.0 * scale, row_y + row_h / 2.0),
            Align2::RIGHT_CENTER,
            *label,
            FontId::monospace(font_size),
            *color,
        );

        let mut display_pos = 0usize;
        for (col_idx, &price) in all_prices.iter().enumerate() {
            let group_size = rgroups
                .iter()
                .find(|&&(p, _)| p == price)
                .map_or(0, |&(_, gs)| gs);
            let gx = ox + label_w + col_x[col_idx];

            for s in 0..group_size {
                let dp = display_pos + s;
                let array_idx = total - 1 - dp;
                let filled = array_idx < count;
                let in_cart = filled && dp >= cheapest_filled && dp < cheapest_filled + cart_amount;

                let sq_x = gx + s as f32 * (sq + inner_gap);
                let sq_rect = Rect::from_min_size(egui::pos2(sq_x, row_y), egui::vec2(sq, row_h));

                let will_refill =
                    dp < cheapest_filled && (cheapest_filled - dp) <= replenish_amount;
                if in_cart {
                    painter.rect_filled(sq_rect, 1.0, *color);
                    egui::Image::new(resource_image(*resource))
                        .tint(Color32::BLACK)
                        .paint_at(ui, sq_rect);
                } else {
                    let icon_tint = if filled {
                        *color
                    } else if will_refill {
                        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 120)
                    } else {
                        dim_color(*color)
                    };
                    egui::Image::new(resource_image(*resource))
                        .tint(icon_tint)
                        .paint_at(ui, sq_rect);
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
                    if pos.y >= row_y && pos.y < row_y + row_h && pos.x >= sq_x && pos.x < sq_x + sq
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

// ── City history graph (used by the CITIES popup window in mod.rs) ─────────────

pub(super) fn city_history_graph(
    ui: &mut Ui,
    history: &[CitySnapshot],
    players_info: &[(PlayerId, PlayerColor)],
    end_game_cities: u8,
    gs: &GameStateView,
    max_height: f32,
) {
    const PAD_L: f32 = 26.0;
    const PAD_B: f32 = 18.0;
    const FRAME_V: f32 = 16.0;
    const TOP_PAD: f32 = 6.0;
    const MIN_H: f32 = 120.0;
    const DOT_R: f32 = 4.0;
    const STEP2_CITIES: usize = 7;

    let h = (max_height - PAD_B - FRAME_V - TOP_PAD).max(MIN_H);
    let w = (ui.available_width() - PAD_L).max(100.0);
    let total_w = PAD_L + w;
    let total_h = PAD_B + h;

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
        [egui::pos2(ox, oy), egui::pos2(ox, oy + h)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );
    painter.line_segment(
        [egui::pos2(ox, oy + h), egui::pos2(ox + w, oy + h)],
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
        egui::pos2(ox - 2.0, oy + h),
        Align2::RIGHT_BOTTOM,
        "0",
        FontId::monospace(13.0),
        theme::TEXT_DIM,
    );

    painter.text(
        egui::pos2(ox, oy + h + PAD_B),
        Align2::LEFT_BOTTOM,
        "1",
        FontId::monospace(13.0),
        theme::TEXT_DIM,
    );
    if rounds > 1 {
        painter.text(
            egui::pos2(ox + w, oy + h + PAD_B),
            Align2::RIGHT_BOTTOM,
            format!("{}", rounds + 1),
            FontId::monospace(13.0),
            theme::TEXT_DIM,
        );
    }

    let step2_y = oy + h - (STEP2_CITIES as f32 / max_cities as f32) * h;
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

    let end_y = oy + h - (end_game_cities as f32 / max_cities as f32) * h;
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
                        let y = oy + h - (*count as f32 / max_cities as f32) * h;
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
                let proj_y = oy + h - (proj_count as f32 / max_cities as f32) * h;
                let proj_pt = egui::pos2(proj_x, proj_y);
                let dim = dim_color(color);
                painter.line_segment([last_pt, proj_pt], Stroke::new(2.5, dim));
                painter.circle_filled(proj_pt, DOT_R, dim);
            }
        }
    }
}

// ── City payout table (used by the info panel in mod.rs) ──────────────────────

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
