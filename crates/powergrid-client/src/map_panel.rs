/// Renders the Germany map inside an egui panel using the registered Bevy texture,
/// with zoom/pan and interactive overlays (resource slots, city markers, build edges).
use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use powergrid_core::{
    map::City,
    types::{Phase, PlayerColor, PlayerId},
    GameState,
};
use std::collections::HashMap;

use crate::{state::AppState, theme};

/// Original map image dimensions (germany.png is 1869 × 2593).
const IMG_W: f32 = 1869.0;
const IMG_H: f32 = 2593.0;

/// Fraction of displayed image width used as hit-test radius for city clicks.
const CITY_HIT_FRAC: f32 = 0.030;
/// Fraction of displayed image width used as draw radius for city dots.
const CITY_R_FRAC: f32 = 0.011;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Draws the full map panel and handles zoom/pan/click input.
/// Returns `true` if a city was clicked (with city_id via `state.toggle_build_city`).
pub fn draw(ui: &mut Ui, state: &mut AppState, game_state: &GameState, my_id: PlayerId) {
    let available = ui.available_rect_before_wrap();

    // Reserve the entire available area for map interaction.
    let (response, painter) = ui.allocate_painter(available.size(), Sense::click_and_drag());

    let map_rect = response.rect;

    // ---- zoom / pan input ----
    let scroll = ui.input(|i| i.smooth_scroll_delta);
    let hover_pos = ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO));
    if map_rect.contains(hover_pos) && scroll.y != 0.0 {
        let hover = hover_pos;
        let factor = 1.15_f32.powf(scroll.y / 20.0);
        let new_zoom = (state.map_zoom * factor).clamp(0.3, 8.0);
        let ratio = new_zoom / state.map_zoom;
        let cx = hover.x - map_rect.left();
        let cy = hover.y - map_rect.top();
        state.map_offset.x = cx - (cx - state.map_offset.x) * ratio;
        state.map_offset.y = cy - (cy - state.map_offset.y) * ratio;
        state.map_zoom = new_zoom;
    }
    if response.dragged() {
        let delta = response.drag_delta();
        state.map_offset += bevy::prelude::Vec2::new(delta.x, delta.y);
    }

    // ---- city click ----
    let is_my_build_turn = matches!(&game_state.phase,
        Phase::BuildCities { remaining } if remaining.first() == Some(&my_id));

    if response.clicked() && is_my_build_turn {
        if let Some(click) = response.interact_pointer_pos() {
            let (img_w, img_h, ox, oy) = image_layout(map_rect);
            let lx = (click.x - map_rect.left() - state.map_offset.x) / state.map_zoom;
            let ly = (click.y - map_rect.top() - state.map_offset.y) / state.map_zoom;
            let xp = (lx - ox) / img_w;
            let yp = (ly - oy) / img_h;

            for (city_id, city) in &game_state.map.cities {
                if let (Some(cx), Some(cy)) = (city.x, city.y) {
                    let dx = xp - cx;
                    let dy = yp - cy;
                    if dx * dx + dy * dy <= CITY_HIT_FRAC * CITY_HIT_FRAC {
                        state.toggle_build_city(city_id.clone());
                        break;
                    }
                }
            }
        }
    }

    // ---- overlays ----
    let (img_w, img_h, ox, oy) = image_layout(map_rect);
    let city_r = CITY_R_FRAC * img_w * state.map_zoom;

    let to_screen = |xp: f32, yp: f32| -> Pos2 {
        Pos2::new(
            map_rect.left() + state.map_offset.x + (ox + xp * img_w) * state.map_zoom,
            map_rect.top() + state.map_offset.y + (oy + yp * img_h) * state.map_zoom,
        )
    };

    // Player color lookup
    let player_colors: HashMap<PlayerId, PlayerColor> = game_state
        .player_order
        .iter()
        .filter_map(|pid| game_state.player(*pid).map(|p| (*pid, p.color)))
        .collect();

    // Build preview edges
    if !state.build_preview.edges.is_empty() {
        let stroke_w = (city_r * 0.6).max(2.0);
        let edge_color = Color32::from_rgba_unmultiplied(0, 220, 255, 220);
        for (from_id, to_id) in &state.build_preview.edges {
            let fc = game_state.map.cities.get(from_id);
            let tc = game_state.map.cities.get(to_id);
            if let (
                Some(City {
                    x: Some(fx),
                    y: Some(fy),
                    ..
                }),
                Some(City {
                    x: Some(tx),
                    y: Some(ty),
                    ..
                }),
            ) = (fc, tc)
            {
                let fp = to_screen(*fx, *fy);
                let tp = to_screen(*tx, *ty);
                // Glow: draw thicker dim line behind the bright one
                painter.line_segment(
                    [fp, tp],
                    Stroke::new(
                        stroke_w * 2.5,
                        Color32::from_rgba_unmultiplied(0, 180, 220, 60),
                    ),
                );
                painter.line_segment([fp, tp], Stroke::new(stroke_w, edge_color));
            }
        }
    }

    // City markers
    for (city_id, city) in &game_state.map.cities {
        if let (Some(cx), Some(cy)) = (city.x, city.y) {
            let center = to_screen(cx, cy);
            let is_selected = state.selected_build_cities.contains(city_id);

            if !city.owners.is_empty() {
                let n = city.owners.len() as f32;
                let spacing = city_r * 2.3;
                let total_w = spacing * (n - 1.0);
                for (i, owner_id) in city.owners.iter().enumerate() {
                    let ox2 = center.x + (i as f32 * spacing) - total_w / 2.0;
                    let c = Pos2::new(ox2, center.y);
                    painter.circle_filled(c, city_r + 1.5, Color32::WHITE);
                    let color = player_colors
                        .get(owner_id)
                        .copied()
                        .map(player_color_to_egui)
                        .unwrap_or(theme::NEON_GREEN);
                    painter.circle_filled(c, city_r, color);
                }
            } else if is_selected {
                // Neon cyan glow for selected build target
                painter.circle_filled(
                    center,
                    city_r * 2.0,
                    Color32::from_rgba_unmultiplied(0, 200, 230, 40),
                );
                painter.circle_filled(center, city_r + 1.5, Color32::WHITE);
                painter.circle_filled(center, city_r, theme::NEON_CYAN);
            } else {
                let alpha = if is_my_build_turn { 180 } else { 90 };
                painter.circle_filled(
                    center,
                    city_r,
                    Color32::from_rgba_unmultiplied(240, 240, 240, alpha),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the "contain" layout of the map image within a rect.
/// Returns (displayed_w, displayed_h, offset_x, offset_y) in local image coords.
fn image_layout(rect: Rect) -> (f32, f32, f32, f32) {
    let img_ratio = IMG_W / IMG_H;
    let rect_ratio = rect.width() / rect.height();
    let (w, h) = if rect_ratio < img_ratio {
        let s = rect.width() / IMG_W;
        (rect.width(), IMG_H * s)
    } else {
        let s = rect.height() / IMG_H;
        (IMG_W * s, rect.height())
    };
    (w, h, (rect.width() - w) / 2.0, (rect.height() - h) / 2.0)
}

fn player_color_to_egui(color: PlayerColor) -> Color32 {
    crate::state::player_color_to_egui(color)
}
