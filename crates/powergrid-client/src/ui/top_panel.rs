use bevy_egui::egui;
use egui::{Color32, RichText, Ui};
use powergrid_core::GameState;

use crate::theme;

use super::helpers::{phase_name, resource_badge, section_header};
use super::phase_tracker::phase_tracker;

pub(super) fn top_panel_contents(ui: &mut Ui, gs: GameState) {
    ui.horizontal(|ui| {
        // Phase / round header
        ui.vertical(|ui| {
            theme::neon_frame_bright().show(ui, |ui| {
                ui.label(
                    RichText::new(format!("ROUND {}", gs.round))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new(phase_name(&gs.phase))
                        .color(theme::NEON_AMBER)
                        .monospace(),
                );
            });
        });

        ui.add_space(8.0);

        // Phase tracker
        phase_tracker(ui, &gs);

        ui.add_space(8.0);

        // Resource market
        ui.vertical(|ui| {
            section_header(ui, "RESOURCE MARKET");
            theme::neon_frame().show(ui, |ui| {
                let r = &gs.resources;
                ui.horizontal(|ui| {
                    resource_badge(ui, "COAL", r.coal, Color32::from_rgb(107, 68, 35));
                    resource_badge(ui, "OIL", r.oil, Color32::from_rgb(60, 60, 60));
                    resource_badge(ui, "GARB", r.garbage, Color32::from_rgb(200, 170, 20));
                    resource_badge(ui, "URAN", r.uranium, Color32::from_rgb(200, 30, 30));
                });
            });
        });
    });
}
