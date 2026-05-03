use bevy_egui::egui;
use egui::{RichText, Ui};
use powergrid_core::{
    types::{Phase, PlayerId},
    GameStateView,
};

use crate::{card_painter, state::player_color_to_egui, theme};

use super::helpers::{dim_color, is_active_player};

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
                        }
                        if active {
                            ui.label(
                                RichText::new("◀ ACTIVE")
                                    .color(theme::NEON_AMBER)
                                    .small()
                                    .monospace(),
                            );
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!("${}", p.money))
                                    .color(theme::NEON_GREEN)
                                    .monospace(),
                            );
                        });
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

                    // Resources + cities row
                    ui.horizontal(|ui| {
                        let res = &p.resources;
                        let mut parts = Vec::new();
                        if res.coal > 0 {
                            parts.push(format!("C:{}", res.coal));
                        }
                        if res.oil > 0 {
                            parts.push(format!("O:{}", res.oil));
                        }
                        if res.garbage > 0 {
                            parts.push(format!("G:{}", res.garbage));
                        }
                        if res.uranium > 0 {
                            parts.push(format!("U:{}", res.uranium));
                        }
                        let res_str = if parts.is_empty() {
                            "No resources".to_string()
                        } else {
                            parts.join("  ")
                        };
                        ui.label(
                            RichText::new(res_str)
                                .color(theme::TEXT_MID)
                                .small()
                                .monospace(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!("{} cities", p.cities.len()))
                                    .color(theme::TEXT_MID)
                                    .small()
                                    .monospace(),
                            );
                        });
                    });

                    // Plants row
                    if !p.plants.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                            for plant in &p.plants {
                                card_painter::draw_plant_card(ui, plant);
                            }
                        });
                    }
                });
            ui.add_space(4.0);
        }
    }
}
