/// Renders the Germany map inside an egui panel using the registered Bevy texture,
/// with zoom/pan and interactive overlays (resource slots, city markers, build edges).
use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use powergrid_core::{
    types::{Phase, PlayerColor, PlayerId},
    GameStateView,
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
pub fn draw(ui: &mut Ui, state: &mut AppState, game_state: &GameStateView, my_id: PlayerId) {
    // Clone the Arc so we can freely call &mut state methods below without borrow conflicts.
    let Some(map) = state.map.clone() else {
        return;
    };

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

            let mut clicked_city: Option<String> = None;
            for (city_id, city) in &map.cities {
                if !game_state.is_city_active(city_id, &map) {
                    continue;
                }
                if let (Some(cx), Some(cy)) = (city.x, city.y) {
                    let dx = xp - cx;
                    let dy = yp - cy;
                    if dx * dx + dy * dy <= CITY_HIT_FRAC * CITY_HIT_FRAC {
                        clicked_city = Some(city_id.clone());
                        break;
                    }
                }
            }
            // Release the map borrow before the mutable state call.
            drop(map);
            if let Some(city_id) = clicked_city {
                state.toggle_build_city(city_id);
            }
            return;
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

    // Base connection lines (all map edges)
    {
        let conn_stroke_w = (city_r * 0.12).max(0.8);
        let conn_color = Color32::from_rgba_unmultiplied(90, 80, 65, 180);
        let conn_glow = Color32::from_rgba_unmultiplied(180, 160, 120, 60);
        let mut drawn = std::collections::HashSet::<(String, String)>::new();
        for (from_id, neighbors) in &map.edges {
            for (to_id, cost) in neighbors {
                let key = if from_id <= to_id {
                    (from_id.clone(), to_id.clone())
                } else {
                    (to_id.clone(), from_id.clone())
                };
                if !drawn.insert(key) {
                    continue;
                }
                let fc = map.cities.get(from_id);
                let tc = map.cities.get(to_id);
                if let (
                    Some(powergrid_core::map::City {
                        x: Some(fx),
                        y: Some(fy),
                        ..
                    }),
                    Some(powergrid_core::map::City {
                        x: Some(tx),
                        y: Some(ty),
                        ..
                    }),
                ) = (fc, tc)
                {
                    let fp = to_screen(*fx, *fy);
                    let tp = to_screen(*tx, *ty);
                    // Shadow
                    painter.line_segment([fp, tp], Stroke::new(conn_stroke_w + 2.0, conn_glow));
                    // Line
                    painter.line_segment([fp, tp], Stroke::new(conn_stroke_w, conn_color));
                    // Cost label at midpoint (only when zoomed in enough)
                    if state.map_zoom >= 1.2 {
                        let mid = Pos2::new((fp.x + tp.x) / 2.0, (fp.y + tp.y) / 2.0);
                        let font_size = (city_r * 1.6).clamp(11.0, 22.0);
                        // Dark outline for legibility
                        for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                            painter.text(
                                Pos2::new(mid.x + dx, mid.y + dy),
                                egui::Align2::CENTER_CENTER,
                                cost.to_string(),
                                egui::FontId::monospace(font_size),
                                Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                            );
                        }
                        painter.text(
                            mid,
                            egui::Align2::CENTER_CENTER,
                            cost.to_string(),
                            egui::FontId::monospace(font_size),
                            Color32::from_rgba_unmultiplied(255, 240, 160, 255),
                        );
                    }
                }
            }
        }
    }

    // Build preview edges
    if !state.build_preview.edges.is_empty() {
        let stroke_w = (city_r * 0.6).max(2.0);
        let edge_color = Color32::from_rgba_unmultiplied(0, 220, 255, 220);
        for (from_id, to_id) in &state.build_preview.edges {
            let fc = map.cities.get(from_id);
            let tc = map.cities.get(to_id);
            if let (
                Some(powergrid_core::map::City {
                    x: Some(fx),
                    y: Some(fy),
                    ..
                }),
                Some(powergrid_core::map::City {
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

    // City markers — always show 3 slots (one per game step).
    // Slot states: filled (owner color), available (outline), locked (faint dot).
    // Inactive region cities render as a single dim house with no slots.
    for (city_id, city) in &map.cities {
        if let (Some(cx), Some(cy)) = (city.x, city.y) {
            let center = to_screen(cx, cy);

            // Inactive region: single dim house, no interaction.
            if !game_state.is_city_active(city_id, &map) {
                painter.add(egui::Shape::convex_polygon(
                    house_points(center, city_r * 0.5),
                    Color32::from_rgba_unmultiplied(60, 60, 60, 100),
                    Stroke::NONE,
                ));
                continue;
            }

            let is_selected = state.selected_build_cities.contains(city_id);
            let spacing = city_r * 2.3;
            let total_w = spacing * 2.0; // 3 slots → 2 gaps

            // Cyan glow behind all slots when this city is selected for building.
            if is_selected {
                painter.circle_filled(
                    center,
                    total_w / 2.0 + city_r * 2.0,
                    Color32::from_rgba_unmultiplied(0, 200, 230, 30),
                );
            }

            // Owners come from city_owners in the view (map.cities[].owners kept in sync too).
            let owners = game_state
                .city_owners
                .get(city_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            for slot in 0usize..3 {
                let x = center.x + (slot as f32 * spacing) - total_w / 2.0;
                let pos = Pos2::new(x, center.y);

                if slot < owners.len() {
                    // Filled: white border + player color house.
                    let color = player_colors
                        .get(&owners[slot])
                        .copied()
                        .map(player_color_to_egui)
                        .unwrap_or(theme::NEON_GREEN);
                    painter.add(egui::Shape::convex_polygon(
                        house_points(pos, city_r),
                        color,
                        Stroke::new(2.0, Color32::WHITE),
                    ));
                } else if slot < game_state.step as usize {
                    // Available this step: outline house, brighter during build turn.
                    let (r, g, b) = if is_my_build_turn {
                        (120, 180, 165) // TEXT_MID
                    } else {
                        (60, 100, 90) // TEXT_DIM
                    };
                    painter.add(egui::Shape::convex_polygon(
                        house_points(pos, city_r),
                        Color32::TRANSPARENT,
                        Stroke::new(1.2, Color32::from_rgb(r, g, b)),
                    ));
                } else {
                    // Locked: tiny faint house indicating the slot exists.
                    painter.add(egui::Shape::convex_polygon(
                        house_points(pos, city_r * 0.45),
                        Color32::from_rgba_unmultiplied(60, 100, 90, 40),
                        Stroke::NONE,
                    ));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// House shape helper
// ---------------------------------------------------------------------------

/// Returns the 5 vertices of a house shape (square body + triangular roof)
/// centered at `center` with approximate half-width `r`.
fn house_points(center: Pos2, r: f32) -> Vec<Pos2> {
    let hw = r * 0.9; // half-width of walls
    let bottom = center.y + r * 0.7;
    let wall_top = center.y - r * 0.1;
    let peak = center.y - r * 1.1;
    vec![
        Pos2::new(center.x - hw, bottom),   // bottom-left
        Pos2::new(center.x + hw, bottom),   // bottom-right
        Pos2::new(center.x + hw, wall_top), // top-right of wall
        Pos2::new(center.x, peak),          // roof peak
        Pos2::new(center.x - hw, wall_top), // top-left of wall
    ]
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
