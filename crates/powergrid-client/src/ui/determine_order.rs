use egui::RichText;
use powergrid_core::GameStateView;

use crate::theme;

/// DETERMINE ORDER overlay — floating top-center, always visible during the game.
/// Shows the current round and step number.
pub(super) fn determine_order_overlay(ctx: &egui::Context, gs: &GameStateView) {
    egui::Area::new(egui::Id::new("determine_order_panel"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 8.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.label(
                RichText::new("DETERMINE ORDER")
                    .color(theme::TEXT_DIM)
                    .small()
                    .monospace()
                    .strong(),
            );
            ui.add_space(2.0);
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
}
