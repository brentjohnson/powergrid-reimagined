use egui::{RichText, Sense};
use powergrid_core::GameStateView;

use crate::state::player_color_to_egui;
use crate::theme;
use crate::ui::helpers::{dim_color, is_active_player};

/// DETERMINE ORDER overlay — floating top-center, always visible during the game.
/// Shows the current round and step number, and the turn-order indicator.
pub(super) fn determine_order_overlay(ctx: &egui::Context, gs: &GameStateView) {
    egui::Area::new(egui::Id::new("determine_order_panel"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 8.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            theme::neon_frame_bright().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("ROUND {}", gs.round))
                            .color(theme::NEON_CYAN)
                            .monospace(),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("STEP {}", gs.step))
                            .color(theme::NEON_CYAN)
                            .monospace(),
                    );
                });
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for pid in &gs.player_order {
                        if let Some(p) = gs.player(*pid) {
                            let base = player_color_to_egui(p.color);
                            let active = is_active_player(gs, *pid);
                            let size = egui::Vec2::splat(12.0);
                            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                            if ui.is_rect_visible(rect) {
                                let painter = ui.painter();
                                if active {
                                    painter.rect_filled(rect, 2.0, dim_color(base));
                                    painter.rect_stroke(
                                        rect,
                                        2.0,
                                        egui::Stroke::new(2.0, base),
                                        egui::StrokeKind::Outside,
                                    );
                                } else {
                                    painter.rect_filled(rect, 2.0, base);
                                }
                            }
                        }
                    }
                });
            });
        });
}
