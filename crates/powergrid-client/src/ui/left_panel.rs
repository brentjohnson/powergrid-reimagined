use egui::{RichText, Ui};
use powergrid_core::{
    types::{Phase, PlayerId, PlayerResources, Resource},
    GameStateView,
};

use crate::{card_painter, state::player_color_to_egui, theme};

use super::helpers::{dim_color, is_active_player, resource_color, resource_image};

pub(super) const PLANT_RES_GAP: f32 = 6.0;
const ICON: f32 = 16.0;
const ICON_GAP: f32 = 2.0;
const MAX_PER_ROW: u8 = 3;

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

pub(super) fn left_panel_contents(ui: &mut Ui, gs: &GameStateView, my_id: PlayerId) {
    for pid in &gs.player_order {
        if let Some(p) = gs.player(*pid) {
            let is_me = p.id == my_id;
            let active = is_active_player(gs, p.id);
            let border_color = if active {
                player_color_to_egui(p.color)
            } else {
                dim_color(player_color_to_egui(p.color))
            };

            egui::Frame::NONE
                .fill(theme::BG_PANEL)
                .stroke(egui::Stroke::new(
                    if active { 2.0 } else { 1.0 },
                    border_color,
                ))
                .inner_margin(egui::Margin::same(6))
                .corner_radius(egui::CornerRadius::same(3))
                .show(ui, |ui| {
                    // Header row
                    ui.horizontal(|ui| {
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

                    // Auction status row
                    if let Phase::Auction {
                        bought,
                        passed,
                        active_bid,
                        ..
                    } = &gs.phase
                    {
                        let status: Option<(String, egui::Color32)> = if bought.contains(&p.id) {
                            Some(("PURCHASED".into(), theme::NEON_GREEN))
                        } else if passed.contains(&p.id) {
                            Some(("PASSED".into(), theme::TEXT_DIM))
                        } else if let Some(bid) = active_bid {
                            if bid.highest_bidder == p.id {
                                Some((format!("BID: ${}", bid.amount), theme::NEON_AMBER))
                            } else if !bid.remaining_bidders.contains(&p.id) {
                                Some(("passed bid".into(), theme::TEXT_DIM))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some((text, color)) = status {
                            ui.label(RichText::new(text).color(color).small().monospace());
                        }
                    }

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

                    // Plants (left, wrapping) + resources (right, per-type rows of ≤3 icons)
                    let res = &p.resources;
                    let has_plants = !p.plants.is_empty();
                    let has_res = res.coal > 0 || res.oil > 0 || res.gas > 0 || res.uranium > 0;
                    if has_plants || has_res {
                        let res_w = resource_col_width(res);
                        ui.horizontal_top(|ui| {
                            ui.spacing_mut().item_spacing.x = PLANT_RES_GAP;
                            if has_plants {
                                let plant_w = if has_res {
                                    (ui.available_width() - res_w - PLANT_RES_GAP).max(0.0)
                                } else {
                                    ui.available_width()
                                };
                                ui.scope(|ui| {
                                    ui.set_max_width(plant_w);
                                    ui.horizontal_wrapped(|ui| {
                                        ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                                        for plant in &p.plants {
                                            card_painter::draw_plant_card(ui, plant);
                                        }
                                    });
                                });
                            }
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
                                                            .fit_to_exact_size(egui::vec2(
                                                                ICON, ICON,
                                                            )),
                                                    );
                                                }
                                            });
                                            remaining -= n;
                                        }
                                    }
                                });
                            }
                        });
                    }
                });
            ui.add_space(4.0);
        }
    }
}
