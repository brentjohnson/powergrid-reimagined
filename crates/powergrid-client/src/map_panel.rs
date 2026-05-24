/// Renders the Germany map inside an egui panel using the registered Bevy texture,
/// with zoom/pan and interactive overlays (resource slots, city markers, build edges).
use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use powergrid_core::{
    actions::HintPayload,
    types::{Phase, PlayerColor, PlayerId},
    GameStateView,
};
use std::collections::HashMap;

use crate::{
    state::{player_color_to_egui, AppState},
    theme,
};

/// Fraction of displayed image width used as draw radius for city house indicators.
const CITY_R_FRAC: f32 = 0.011;
/// Half-width / half-height of the city rectangle in fractional image coords (zoom-invariant).
const CITY_RECT_HALF_W_FRAC: f32 = CITY_R_FRAC * 3.6;
const CITY_RECT_HALF_H_FRAC: f32 = CITY_R_FRAC * 2.0;
/// Zoom level beyond which city box/house/label size stops growing.
/// Aligns with the cost-label visibility threshold so zooming past this
/// spreads cities apart while boxes stay fixed, revealing the labels.
const CITY_SCALE_CAP: f32 = 1.2;

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

    let t = crate::effects::elapsed_seconds(ui.ctx());

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
        state.map_offset += egui::Vec2::new(delta.x, delta.y);
    }

    // Size zoom: boxes/houses/labels cap so zooming past the cap spreads
    // cities apart while box size stays fixed, revealing connection-cost labels.
    let render_zoom = state.map_zoom.min(CITY_SCALE_CAP);

    // ---- city click ----
    let is_my_build_turn = matches!(&game_state.phase,
        Phase::BuildCities { remaining } if remaining.first() == Some(&my_id));

    if response.clicked() && is_my_build_turn {
        if let Some(click) = response.interact_pointer_pos() {
            let (img_w, img_h, ox, oy) = image_layout(map_rect, map.width, map.height);
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
                    let dx = (xp - cx).abs();
                    let dy = (yp - cy).abs();
                    let hit_scale = render_zoom / state.map_zoom;
                    if dx <= CITY_RECT_HALF_W_FRAC * hit_scale
                        && dy <= CITY_RECT_HALF_H_FRAC * hit_scale
                    {
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
    let (img_w, img_h, ox, oy) = image_layout(map_rect, map.width, map.height);
    let city_r = CITY_R_FRAC * img_w * render_zoom;

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
        let conn_color = theme::map_conn_color();
        let conn_glow = theme::map_conn_glow();
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
                                Color32::from_black_alpha(180),
                            );
                        }
                        painter.text(
                            mid,
                            egui::Align2::CENTER_CENTER,
                            cost.to_string(),
                            egui::FontId::monospace(font_size),
                            theme::MAP_CONN_COST_LABEL,
                        );
                    }
                }
            }
        }
    }

    // Keep animating while cities are selected so the glow stays live.
    if !state.selected_build_cities.is_empty() {
        crate::effects::keep_animating(ui.ctx());
    }

    // Build preview edges — animated lightning bolts (local player)
    if !state.build_preview.edges.is_empty() {
        let stroke_w = (city_r * 0.6).max(2.0);
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
                crate::effects::draw_lightning_edge(
                    &painter,
                    fp,
                    tp,
                    t,
                    stroke_w,
                    crate::effects::edge_seed(from_id, to_id),
                );
            }
        }
        crate::effects::keep_animating(ui.ctx());
    }

    // Peer build route edges
    {
        let peer_edge_w = (city_r * 0.35).max(1.2);
        let mut any_peer_edges = false;
        for (pid, hint) in &state.peer_hints.hints {
            if let HintPayload::BuildSelection { edges, .. } = hint {
                if edges.is_empty() {
                    continue;
                }
                let peer_color = player_colors
                    .get(pid)
                    .copied()
                    .map(player_color_to_egui)
                    .unwrap_or(Color32::GRAY);
                let alpha = ((t * 2.5).sin() * 0.3 + 0.7) as f32;
                let edge_color = Color32::from_rgba_unmultiplied(
                    peer_color.r(),
                    peer_color.g(),
                    peer_color.b(),
                    (alpha * 180.0) as u8,
                );
                for (from_id, to_id) in edges {
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
                        painter.line_segment(
                            [fp, tp],
                            Stroke::new(peer_edge_w * 2.5, {
                                let c = peer_color;
                                Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 40)
                            }),
                        );
                        painter.line_segment([fp, tp], Stroke::new(peer_edge_w, edge_color));
                    }
                }
                any_peer_edges = true;
            }
        }
        if any_peer_edges {
            crate::effects::keep_animating(ui.ctx());
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
                    theme::map_city_inactive_house(),
                    Stroke::NONE,
                ));
                continue;
            }

            let is_selected = state.selected_build_cities.contains(city_id);
            let spacing = city_r * 2.3;
            let total_w = spacing * 2.0; // 3 slots → 2 gaps

            // Rectangle background — covers connection lines passing through the city.
            let hw = CITY_RECT_HALF_W_FRAC * img_w * render_zoom;
            let hh = CITY_RECT_HALF_H_FRAC * img_w * render_zoom;
            let rect = egui::Rect::from_center_size(center, egui::vec2(hw * 2.0, hh * 2.0));
            painter.rect_filled(rect, 4.0, theme::map_city_bg());
            if !is_selected {
                painter.rect_stroke(
                    rect,
                    4.0,
                    Stroke::new(1.0, theme::map_city_border()),
                    egui::StrokeKind::Inside,
                );
            }

            // Peer city selection rings — drawn outside the local rect, one ring per peer
            for (offset_idx, (pid, _)) in state
                .peer_hints
                .hints
                .iter()
                .filter(|(_, h)| matches!(h, HintPayload::BuildSelection { cities, .. } if cities.contains(city_id)))
                .enumerate()
            {
                let peer_color = player_colors
                    .get(pid)
                    .copied()
                    .map(player_color_to_egui)
                    .unwrap_or(Color32::GRAY);
                let expand = 3.0 + offset_idx as f32 * 3.5;
                let ring_rect = rect.expand(expand);
                let pulse = ((t * 2.8 + crate::effects::city_seed(city_id) as f64 / 1000.0).sin() * 0.5 + 0.5) as f32;
                let ring_a = (pulse * 120.0 + 100.0) as u8;
                painter.rect_stroke(
                    ring_rect,
                    5.0,
                    Stroke::new(1.5, Color32::from_rgba_unmultiplied(
                        peer_color.r(),
                        peer_color.g(),
                        peer_color.b(),
                        ring_a,
                    )),
                    egui::StrokeKind::Inside,
                );
                crate::effects::keep_animating(ui.ctx());
            }

            // Owners come from city_owners in the view (map.cities[].owners kept in sync too).
            let owners = game_state
                .city_owners
                .get(city_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            // Houses sit in the upper half of the rect.
            let house_y = center.y - city_r * 0.4;
            for slot in 0usize..3 {
                let x = center.x + (slot as f32 * spacing) - total_w / 2.0;
                let pos = Pos2::new(x, house_y);

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
                    // Available this step: electric fill+border when selected, dim outline otherwise.
                    if is_selected {
                        painter.add(egui::Shape::convex_polygon(
                            house_points(pos, city_r),
                            theme::map_slot_active_fill(),
                            Stroke::new(2.0, theme::MAP_SLOT_ACTIVE_BORDER),
                        ));
                    } else {
                        let stroke_color = if is_my_build_turn {
                            theme::TEXT_MID
                        } else {
                            theme::TEXT_DIM
                        };
                        painter.add(egui::Shape::convex_polygon(
                            house_points(pos, city_r),
                            Color32::TRANSPARENT,
                            Stroke::new(1.2, stroke_color),
                        ));
                    }
                } else {
                    // Locked: tiny faint house indicating the slot exists.
                    painter.add(egui::Shape::convex_polygon(
                        house_points(pos, city_r * 0.45),
                        theme::map_slot_locked(),
                        Stroke::NONE,
                    ));
                }
            }

            // City name label below the houses.
            let label_y = center.y + city_r * 1.1;
            let font_size = (city_r * 1.4).clamp(10.0, 22.0);
            let label_color = if is_my_build_turn {
                theme::MAP_CITY_LABEL_ACTIVE
            } else {
                theme::map_city_label_dim()
            };
            painter.text(
                Pos2::new(center.x, label_y),
                egui::Align2::CENTER_CENTER,
                &city.name,
                egui::FontId::proportional(font_size),
                label_color,
            );

            // Animated box highlight drawn last so particles appear on top of content.
            if is_selected {
                crate::effects::draw_selected_city_box_highlight(
                    &painter,
                    rect,
                    t,
                    crate::effects::city_seed(city_id),
                );
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
fn image_layout(rect: Rect, board_w: f32, board_h: f32) -> (f32, f32, f32, f32) {
    let img_ratio = board_w / board_h;
    let rect_ratio = rect.width() / rect.height();
    let (w, h) = if rect_ratio < img_ratio {
        let s = rect.width() / board_w;
        (rect.width(), board_h * s)
    } else {
        let s = rect.height() / board_h;
        (board_w * s, rect.height())
    };
    (w, h, (rect.width() - w) / 2.0, (rect.height() - h) / 2.0)
}
