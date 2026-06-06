use egui::{Color32, Id, Rect, RichText, Ui, UiBuilder};
use powergrid_core::{
    types::{Player, PlayerId, PlayerResources, Resource},
    GameStateView,
};

use crate::{card_painter, state::player_color_to_egui, theme};

use super::helpers::{dim_color, is_active_player, resource_color, resource_image};

pub(super) const PLANT_RES_GAP: f32 = 6.0;
pub(super) const ICON: f32 = 16.0;
const ICON_GAP: f32 = 2.0;
const MAX_PER_ROW: u8 = 3;

// Animation constants
const CARD_GAP: f32 = 4.0;
const SLIDE_DUR: f32 = 1.0; // reorder glide duration
const HEIGHT_DUR: f32 = 0.35; // card grow/shrink duration
const DEFAULT_H: f32 = 90.0; // first-frame height estimate before measurement

/// Width of the resource icon column for a player, based on the largest single-type count
/// capped at MAX_PER_ROW. Returns 0 when the player owns no resources.
pub(super) fn resource_col_width(res: &PlayerResources) -> f32 {
    let max_count = res.coal.max(res.oil).max(res.gas).max(res.uranium);
    let cols = max_count.min(MAX_PER_ROW) as f32;
    if cols == 0.0 {
        0.0
    } else {
        cols * ICON + (cols - 1.0) * ICON_GAP
    }
}

/// Linear interpolation between two `Color32` values.
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
        (a.a() as f32 + (b.a() as f32 - a.a() as f32) * t) as u8,
    )
}

/// Smooth cubic ease-in-out in [0, 1].
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The border is painted as a size-neutral overlay (StrokeKind::Inside) so that
/// active-vs-inactive border width does NOT affect the card's layout size.
fn draw_player_card(ui: &mut Ui, gs: &GameStateView, p: &Player, is_me: bool, active: bool) {
    let border_color = if active {
        player_color_to_egui(p.color)
    } else {
        dim_color(player_color_to_egui(p.color))
    };
    let border_width = if active { 2.0 } else { 1.0 };

    // No stroke on the Frame — stroke width would be counted in layout size.
    // The border is painted as an overlay below (size-neutral).
    let bg = lerp_color(theme::BG_PANEL, player_color_to_egui(p.color), 0.12);
    let frame_resp = egui::Frame::NONE
        .fill(bg)
        .inner_margin(egui::Margin::same(6))
        .corner_radius(egui::CornerRadius::same(3))
        .show(ui, |ui| {
            // Header row
            ui.horizontal(|ui| {
                // House icon — same shape as the map city markers.
                let (resp, painter) =
                    ui.allocate_painter(egui::vec2(ICON, ICON), egui::Sense::hover());
                let center = resp.rect.center();
                let r = ICON / 2.2;
                painter.add(egui::Shape::convex_polygon(
                    crate::map_panel::house_points(center, r),
                    player_color_to_egui(p.color),
                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                ));

                let name_color = player_color_to_egui(p.color);
                ui.colored_label(name_color, RichText::new(&p.name).monospace().strong());
                if is_me {
                    ui.label(RichText::new("(you)").color(theme::TEXT_DIM).small());
                    ui.label(
                        RichText::new(format!("${}", p.money))
                            .color(theme::NEON_GREEN)
                            .small()
                            .monospace(),
                    );
                }
                if active {
                    ui.label(
                        RichText::new("◀ ACTIVE")
                            .color(theme::NEON_AMBER)
                            .small()
                            .monospace(),
                    );
                }
            });

            // Cities + capacity row (single combined label — no width competition)
            let capacity: u32 = p.plants.iter().map(|pl| pl.cities as u32).sum();
            ui.label(
                RichText::new(format!(
                    "capacity {} / cities {}",
                    capacity,
                    gs.player_city_count(p.id)
                ))
                .color(theme::TEXT_MID)
                .small()
                .monospace(),
            );

            // Plants (always 3 slots; empty slots are grey outline placeholders) +
            // resources (right column, per-type rows of ≤3 icons).
            let res = &p.resources;
            let has_res = res.coal > 0 || res.oil > 0 || res.gas > 0 || res.uranium > 0;
            let res_w = resource_col_width(res);
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = PLANT_RES_GAP;
                let plant_w = if has_res {
                    (ui.available_width() - res_w - PLANT_RES_GAP).max(0.0)
                } else {
                    ui.available_width()
                };
                ui.scope(|ui| {
                    ui.set_max_width(plant_w);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                        for i in 0..3 {
                            match p.plants.get(i) {
                                Some(plant) => {
                                    card_painter::draw_plant_card(ui, plant);
                                }
                                None => {
                                    card_painter::draw_plant_placeholder(ui);
                                }
                            }
                        }
                    });
                });
                if has_res {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 1.0;
                        for (r, count) in [
                            (Resource::Coal, res.coal),
                            (Resource::Gas, res.gas),
                            (Resource::Oil, res.oil),
                            (Resource::Uranium, res.uranium),
                        ] {
                            if count == 0 {
                                continue;
                            }
                            let mut remaining = count;
                            while remaining > 0 {
                                let n = remaining.min(MAX_PER_ROW);
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = ICON_GAP;
                                    for _ in 0..n {
                                        ui.add(
                                            egui::Image::new(resource_image(r))
                                                .tint(resource_color(r))
                                                .fit_to_exact_size(egui::vec2(ICON, ICON)),
                                        );
                                    }
                                });
                                remaining -= n;
                            }
                        }
                    });
                }
            });
        });

    // Overlay the border inside the frame rect — size-neutral, does not shift layout.
    ui.painter().rect_stroke(
        frame_resp.response.rect,
        egui::CornerRadius::same(3),
        egui::Stroke::new(border_width, border_color),
        egui::StrokeKind::Inside,
    );
}

pub(super) fn left_panel_contents(ui: &mut Ui, gs: &GameStateView, my_id: PlayerId) {
    let ctx = ui.ctx().clone();
    let now = ctx.input(|i| i.time) as f32;
    let order = &gs.player_order;
    let origin = ui.cursor().min;
    let full_w = ui.available_width();

    // Previous order — used to detect position changes.
    let prev_order: Option<Vec<PlayerId>> = ctx.data(|d| d.get_temp(Id::new("pcard_prev_order")));

    let mut target_y = 0.0_f32;
    let mut any_animating = false;

    for (i_new, pid) in order.iter().enumerate() {
        let p = match gs.player(*pid) {
            Some(p) => p,
            None => continue,
        };

        // ── Animated height ───────────────────────────────────────────────────
        // Snap on first appearance (no cached measurement yet), then ease on change.
        let h_nat_opt: Option<f32> = ctx.data(|d| d.get_temp(Id::new(("pcard_h_nat", *pid))));
        let h_nat = h_nat_opt.unwrap_or(DEFAULT_H);
        let h_dur = if h_nat_opt.is_some() { HEIGHT_DUR } else { 0.0 };
        let h_anim = ctx.animate_value_with_time(Id::new(("pcard_h", *pid)), h_nat, h_dur);
        if (h_anim - h_nat).abs() > 0.5 {
            any_animating = true;
        }

        // ── Slot position (exact sum of animated heights — no overlap possible) ─
        let slot_top = target_y;
        target_y += h_anim + CARD_GAP;

        // ── Detect reorder and seed displacement ──────────────────────────────
        let i_old = prev_order
            .as_ref()
            .and_then(|po| po.iter().position(|x| x == pid));
        if i_old.map(|o| o != i_new).unwrap_or(false) {
            // Card changed position: displacement = distance from old rendered Y to new slot.
            let last_y: f32 = ctx
                .data(|d| d.get_temp(Id::new(("pcard_last_y", *pid))))
                .unwrap_or(slot_top);
            ctx.data_mut(|d| d.insert_temp(Id::new(("pcard_reord_d", *pid)), last_y - slot_top));
            ctx.data_mut(|d| d.insert_temp(Id::new(("pcard_reord_t0", *pid)), now));
        }

        // ── Reorder glide (decays to zero over SLIDE_DUR) ─────────────────────
        let reord_d: f32 = ctx
            .data(|d| d.get_temp(Id::new(("pcard_reord_d", *pid))))
            .unwrap_or(0.0);
        let reord_t0: f32 = ctx
            .data(|d| d.get_temp(Id::new(("pcard_reord_t0", *pid))))
            .unwrap_or(now - SLIDE_DUR - 1.0); // default: fully expired
        let t_frac = ((now - reord_t0) / SLIDE_DUR).clamp(0.0, 1.0);
        let glide = reord_d * (1.0 - smoothstep(t_frac));
        let rendered_y = slot_top + glide;
        if glide.abs() > 0.5 {
            any_animating = true;
        }

        // Remember rendered Y so the next reorder can compute the right displacement.
        ctx.data_mut(|d| d.insert_temp(Id::new(("pcard_last_y", *pid)), rendered_y));

        // ── Draw card at animated position, clipped to h_anim ────────────────
        // Clipping ensures growing content does not visually bleed into the next
        // card's slot. Setting clip_rect is purely visual — it does not affect
        // min_rect, so we still measure the true natural height below.
        let pos = origin + egui::vec2(0.0, rendered_y);
        let card_max_rect = Rect::from_min_size(pos, egui::vec2(full_w, 1000.0));
        let resp = ui.scope_builder(UiBuilder::new().max_rect(card_max_rect), |ui| {
            let clip = Rect::from_min_size(pos, egui::vec2(full_w, h_anim));
            ui.set_clip_rect(ui.clip_rect().intersect(clip));
            let is_me = p.id == my_id;
            let active = is_active_player(gs, p.id);
            draw_player_card(ui, gs, p, is_me, active);
        });

        // Cache the natural (unclipped) height for the next frame's layout.
        ctx.data_mut(|d| {
            d.insert_temp(Id::new(("pcard_h_nat", *pid)), resp.response.rect.height())
        });
    }

    // Persist order for next-frame change detection.
    ctx.data_mut(|d| d.insert_temp(Id::new("pcard_prev_order"), order.clone()));

    // Ensure the ScrollArea always reserves the full stacked content height.
    let needed_bottom = origin.y + target_y;
    let cursor_y = ui.cursor().min.y;
    if needed_bottom > cursor_y {
        ui.add_space(needed_bottom - cursor_y);
    }

    // animate_value_with_time self-schedules repaints for height easing.
    // Manually request repaints for glow and reorder glide.
    if any_animating {
        ctx.request_repaint();
    }
}
