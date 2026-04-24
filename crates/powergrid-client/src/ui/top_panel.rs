use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{Align2, Color32, FontId, Rect, RichText, Sense, Stroke, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerColor, PlayerId, ResourceMarket},
    GameState,
};

use crate::{
    card_painter,
    state::{player_color_to_egui, AppState, CitySnapshot},
    theme,
    ws::WsChannels,
};

use super::helpers::{dim_color, section_header, send};
use super::phase_tracker::phase_tracker;

pub(super) fn top_panel_contents(
    ui: &mut Ui,
    gs: GameState,
    state: &AppState,
    channels: &Option<Res<WsChannels>>,
    my_id: PlayerId,
) {
    ui.horizontal(|ui| {
        // Round / Step header
        ui.vertical(|ui| {
            theme::neon_frame_bright().show(ui, |ui| {
                ui.label(
                    RichText::new(format!("ROUND {}", gs.round))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new(format!("STEP  {}", gs.step))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
            });
        });

        ui.add_space(8.0);

        // Phase tracker
        phase_tracker(ui, &gs);

        ui.add_space(8.0);

        // Plant market
        ui.vertical(|ui| {
            section_header(ui, "PLANT MARKET");
            theme::neon_frame().show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    if gs.step >= 3 {
                        // Step 3: two columns of 3, all plants purchasable, no labels.
                        let mid = gs.market.actual.len().div_ceil(2);
                        let (left, right) = gs.market.actual.split_at(mid);
                        ui.vertical(|ui| {
                            plant_column(ui, left, channels, &gs.phase, my_id, &gs.player_order);
                        });
                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            plant_column(ui, right, channels, &gs.phase, my_id, &gs.player_order);
                        });
                    } else {
                        // Steps 1 & 2: ACTUAL and FUTURE columns.
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
                            );
                        });
                    }
                });
            });
        });

        ui.add_space(8.0);

        // Resource market
        ui.vertical(|ui| {
            section_header(ui, "RESOURCE MARKET");
            theme::neon_frame().show(ui, |ui| {
                resource_market_grid(ui, &gs.resources);
            });
        });

        ui.add_space(8.0);

        // Cities graph
        if !state.city_history.is_empty() {
            let players_info: Vec<(PlayerId, PlayerColor)> =
                gs.players.iter().map(|p| (p.id, p.color)).collect();
            ui.vertical(|ui| {
                section_header(ui, "CITIES");
                theme::neon_frame().show(ui, |ui| {
                    city_history_graph(
                        ui,
                        &state.city_history,
                        &players_info,
                        gs.end_game_cities,
                        &gs,
                    );
                });
            });
        }
    });
}

fn plant_column(
    ui: &mut Ui,
    plants: &[powergrid_core::types::PowerPlant],
    channels: &Option<Res<WsChannels>>,
    phase: &Phase,
    my_id: PlayerId,
    player_order: &[PlayerId],
) {
    let is_my_auction_turn = matches!(phase, Phase::Auction { current_bidder_idx, active_bid, .. }
        if active_bid.is_none() && player_order.get(*current_bidder_idx) == Some(&my_id));

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        for plant in plants {
            let resp = card_painter::draw_plant_card(ui, plant);
            if is_my_auction_turn && resp.clicked() {
                send(
                    Action::SelectPlant {
                        plant_number: plant.number,
                    },
                    channels,
                );
            }
            if resp.hovered() {
                egui::show_tooltip_at_pointer(
                    ui.ctx(),
                    ui.layer_id(),
                    egui::Id::new(plant.number),
                    |ui| {
                        plant_tooltip(ui, plant);
                    },
                );
            }
        }
    });
}

fn plant_tooltip(ui: &mut Ui, plant: &powergrid_core::types::PowerPlant) {
    ui.label(
        RichText::new(format!(
            "#{} {:?}\nCost: {}  Cities: {}",
            plant.number, plant.kind, plant.cost, plant.cities
        ))
        .monospace()
        .color(theme::TEXT_BRIGHT),
    );
}

fn resource_market_grid(ui: &mut Ui, market: &ResourceMarket) {
    const SQ: f32 = 14.0; // square size (height and base width)
    const INNER_GAP: f32 = 2.0; // gap between squares within a COG price group
    const GROUP_GAP: f32 = 8.0; // gap between price groups (COG) or uranium slots
    const URAN_W: f32 = 24.0; // uranium slot width (slightly wider to fit price labels)
    const LABEL_W: f32 = 36.0; // width of row labels (COAL, OIL, etc.)
    const HEADER_H: f32 = 20.0; // height of price label header row
    const ROW_H: f32 = SQ;
    const ROW_GAP: f32 = 4.0; // gap between resource rows
    const SECTION_GAP: f32 = 12.0; // gap between COG section and uranium section

    // COG: 8 price groups × 3 slots each
    let cog_group_w = 3.0 * SQ + 2.0 * INNER_GAP; // 26px per group
    let cog_total_w = 8.0 * cog_group_w + 7.0 * GROUP_GAP; // 208+35 = 243px

    // Uranium: 12 individual slots, each treated as its own group
    let uran_total_w = 12.0 * URAN_W + 11.0 * GROUP_GAP; // 132+55 = 187px

    let content_w = cog_total_w.max(uran_total_w);
    let total_w = LABEL_W + content_w;

    // Total height: COG header + 3 rows + uranium header + uranium row
    let total_h = HEADER_H
        + ROW_GAP
        + ROW_H
        + ROW_GAP
        + ROW_H
        + ROW_GAP
        + ROW_H
        + SECTION_GAP
        + HEADER_H
        + ROW_GAP
        + ROW_H;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(total_w, total_h), Sense::hover());

    if !ui.is_rect_visible(rect) {
        return;
    }

    let painter = ui.painter();
    let ox = rect.min.x;
    let oy = rect.min.y;

    let coal_color = Color32::from_rgb(150, 100, 55);
    let oil_color = Color32::from_rgb(110, 110, 140);
    let garb_color = Color32::from_rgb(200, 170, 20);
    let uran_color = Color32::from_rgb(200, 30, 30);

    // ── COG price header ($1 on left = cheapest, $8 on right = most expensive) ──
    for g in 0..8usize {
        let price = g + 1;
        let gx = ox + LABEL_W + g as f32 * (cog_group_w + GROUP_GAP);
        let cx = gx + cog_group_w / 2.0;
        painter.text(
            egui::pos2(cx, oy + HEADER_H / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(10.0),
            theme::TEXT_DIM,
        );
    }

    // ── COG rows (coal, oil, garbage) ──
    // Price table: index 0 = most expensive ($8), index 23 = cheapest ($1).
    // `count` resources occupy indices 0..(count-1).
    // Display pos 0 (leftmost, $1) maps to array index (total-1 - display_pos).
    // Slot is filled when array_idx < count.
    let cog_rows: &[(&str, Color32, u8, usize)] = &[
        ("COAL", coal_color, market.coal, 24),
        ("OIL", oil_color, market.oil, 24),
        ("GARB", garb_color, market.garbage, 24),
    ];

    for (i, (label, color, count, total)) in cog_rows.iter().enumerate() {
        let row_y = oy + HEADER_H + ROW_GAP + i as f32 * (ROW_H + ROW_GAP);

        painter.text(
            egui::pos2(ox + LABEL_W - 2.0, row_y + ROW_H / 2.0),
            Align2::RIGHT_CENTER,
            *label,
            FontId::monospace(10.0),
            *color,
        );

        for g in 0..8usize {
            let gx = ox + LABEL_W + g as f32 * (cog_group_w + GROUP_GAP);
            for s in 0..3usize {
                let display_pos = g * 3 + s;
                let array_idx = (total - 1) - display_pos;
                let filled = array_idx < *count as usize;
                let sq_x = gx + s as f32 * (SQ + INNER_GAP);
                let sq_rect = Rect::from_min_size(egui::pos2(sq_x, row_y), egui::vec2(SQ, ROW_H));
                painter.rect_filled(
                    sq_rect,
                    1.0,
                    if filled { *color } else { dim_color(*color) },
                );
            }
        }
    }

    // ── Uranium section ──
    // Price table (cheap→expensive display): [1,2,3,4,5,6,7,8,10,12,14,16]
    // Array index 0 = $16, index 11 = $1.
    // Display pos i (0=cheapest) maps to array index 11-i.
    let usec_y = oy + HEADER_H + ROW_GAP + 3.0 * (ROW_H + ROW_GAP) + SECTION_GAP;
    let uran_prices: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];

    for (i, &price) in uran_prices.iter().enumerate() {
        let sx = ox + LABEL_W + i as f32 * (URAN_W + GROUP_GAP);
        let cx = sx + URAN_W / 2.0;
        painter.text(
            egui::pos2(cx, usec_y + HEADER_H / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(9.0),
            theme::TEXT_DIM,
        );
    }

    let uran_row_y = usec_y + HEADER_H + ROW_GAP;

    painter.text(
        egui::pos2(ox + LABEL_W - 2.0, uran_row_y + ROW_H / 2.0),
        Align2::RIGHT_CENTER,
        "URAN",
        FontId::monospace(10.0),
        uran_color,
    );

    for i in 0..12usize {
        let array_idx = 11 - i; // display 0 ($1) maps to array 11
        let filled = array_idx < market.uranium as usize;
        let sx = ox + LABEL_W + i as f32 * (URAN_W + GROUP_GAP);
        let sq_rect = Rect::from_min_size(egui::pos2(sx, uran_row_y), egui::vec2(URAN_W, ROW_H));
        painter.rect_filled(
            sq_rect,
            1.0,
            if filled {
                uran_color
            } else {
                dim_color(uran_color)
            },
        );
    }
}

fn city_history_graph(
    ui: &mut Ui,
    history: &[CitySnapshot],
    players_info: &[(PlayerId, PlayerColor)],
    end_game_cities: u8,
    gs: &GameState,
) {
    const PAD_L: f32 = 14.0; // left padding for y-axis label
    const PAD_B: f32 = 10.0; // bottom padding for x-axis label
    const H: f32 = 114.0; // plot area height — sized to match plant market
    const DOT_R: f32 = 2.0;
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

    // Determine y range — ensure indicator lines are always visible
    let max_cities = history
        .iter()
        .flat_map(|snap| snap.iter().map(|(_, c)| *c))
        .chain(gs.players.iter().map(|p| p.city_count()))
        .max()
        .unwrap_or(1)
        .max(end_game_cities as usize)
        .max(1);

    let rounds = history.len();
    // Include a projected point slot beyond the historical rounds
    let total_points = rounds + 1;
    let x_for = |idx: usize| -> f32 {
        if total_points <= 1 {
            ox
        } else {
            ox + (idx as f32 / (total_points - 1) as f32) * w
        }
    };

    // Draw axes
    painter.line_segment(
        [egui::pos2(ox, oy), egui::pos2(ox, oy + H)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );
    painter.line_segment(
        [egui::pos2(ox, oy + H), egui::pos2(ox + w, oy + H)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );

    // Y-axis label (max value)
    painter.text(
        egui::pos2(ox - 2.0, oy),
        Align2::RIGHT_TOP,
        format!("{max_cities}"),
        FontId::monospace(7.0),
        theme::TEXT_DIM,
    );
    painter.text(
        egui::pos2(ox - 2.0, oy + H),
        Align2::RIGHT_BOTTOM,
        "0",
        FontId::monospace(7.0),
        theme::TEXT_DIM,
    );

    // X-axis round labels (first and last)
    painter.text(
        egui::pos2(ox, oy + H + PAD_B),
        Align2::LEFT_BOTTOM,
        "1",
        FontId::monospace(7.0),
        theme::TEXT_DIM,
    );
    if rounds > 1 {
        painter.text(
            egui::pos2(ox + w, oy + H + PAD_B),
            Align2::RIGHT_BOTTOM,
            format!("{}", rounds + 1),
            FontId::monospace(7.0),
            theme::TEXT_DIM,
        );
    }

    // Draw Step 2 indicator line at 7 cities
    let step2_y = oy + H - (STEP2_CITIES as f32 / max_cities as f32) * H;
    let step2_color = Color32::from_rgba_unmultiplied(180, 180, 60, 180);
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
        FontId::monospace(6.0),
        step2_color,
    );

    // Draw end game indicator line
    let end_y = oy + H - (end_game_cities as f32 / max_cities as f32) * H;
    let end_color = Color32::from_rgba_unmultiplied(220, 80, 80, 200);
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
        FontId::monospace(6.0),
        end_color,
    );

    // Draw one line per player
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

        // Draw line segments
        for pair in points.windows(2) {
            painter.line_segment([pair[0], pair[1]], Stroke::new(1.5, color));
        }

        // Draw dots
        for pt in &points {
            painter.circle_filled(*pt, DOT_R, color);
        }

        // Draw projected next-round point (dimmer) using live city count
        if let Some(&last_pt) = points.last() {
            if let Some(player) = gs.players.iter().find(|p| p.id == *player_id) {
                let proj_count = player.city_count();
                let proj_x = x_for(rounds); // one slot beyond historical points
                let proj_y = oy + H - (proj_count as f32 / max_cities as f32) * H;
                let proj_pt = egui::pos2(proj_x, proj_y);
                let dim = dim_color(color);
                painter.line_segment([last_pt, proj_pt], Stroke::new(1.5, dim));
                painter.circle_filled(proj_pt, DOT_R, dim);
            }
        }
    }
}
