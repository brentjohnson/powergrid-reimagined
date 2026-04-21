use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{Align2, Color32, FontId, RichText, Sense, Stroke, Ui};
use powergrid_core::{
    types::{PlayerColor, PlayerId},
    GameState,
};

use crate::{
    state::{player_color_to_egui, AppState, CitySnapshot},
    theme,
    ws::WsChannels,
};

use super::action_panel::action_panel;
use super::helpers::{dim_color, section_header};

pub(super) fn right_panel_contents(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    // ---- City count graph ----
    if !state.city_history.is_empty() {
        let players_info: Vec<(PlayerId, PlayerColor)> =
            gs.players.iter().map(|p| (p.id, p.color)).collect();
        section_header(ui, "CITIES");
        theme::neon_frame().show(ui, |ui| {
            city_history_graph(
                ui,
                &state.city_history,
                &players_info,
                gs.end_game_cities,
                gs,
            );
        });
        ui.add_space(4.0);
    }

    // ---- My action panel ----
    if gs.player(my_id).is_some() {
        section_header(ui, "ACTION CONSOLE");
        theme::neon_frame_bright().show(ui, |ui| {
            if let Some(err) = &state.error_message.clone() {
                ui.label(
                    RichText::new(format!("⚠ {err}"))
                        .color(theme::NEON_RED)
                        .small()
                        .monospace(),
                );
                ui.add_space(4.0);
            }
            action_panel(ui, state, channels, gs, my_id);
        });

        ui.add_space(4.0);
    }

    // ---- Event log ----
    section_header(ui, "EVENT LOG");
    theme::neon_frame().show(ui, |ui| {
        for entry in gs.event_log.iter().rev().take(8) {
            ui.label(
                RichText::new(entry)
                    .color(theme::TEXT_DIM)
                    .small()
                    .monospace(),
            );
        }
    });

    ui.add_space(8.0);
}

fn city_history_graph(
    ui: &mut Ui,
    history: &[CitySnapshot],
    players_info: &[(PlayerId, PlayerColor)],
    end_game_cities: u8,
    gs: &GameState,
) {
    const W: f32 = 360.0;
    const H: f32 = 72.0;
    const PAD_L: f32 = 14.0; // left padding for y-axis label
    const PAD_B: f32 = 10.0; // bottom padding for x-axis label
    const DOT_R: f32 = 2.0;
    const STEP2_CITIES: usize = 7;

    let total_w = PAD_L + W;
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
            ox + (idx as f32 / (total_points - 1) as f32) * W
        }
    };

    // Draw axes
    painter.line_segment(
        [egui::pos2(ox, oy), egui::pos2(ox, oy + H)],
        Stroke::new(1.0, theme::TEXT_DIM),
    );
    painter.line_segment(
        [egui::pos2(ox, oy + H), egui::pos2(ox + W, oy + H)],
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
            egui::pos2(ox + W, oy + H + PAD_B),
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
    while x < ox + W {
        let x_end = (x + dash_len).min(ox + W);
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
    while x < ox + W {
        let x_end = (x + dash_len).min(ox + W);
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
