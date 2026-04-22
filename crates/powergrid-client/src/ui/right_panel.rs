use bevy::prelude::Res;
use egui::{RichText, Ui};
use powergrid_core::{types::PlayerId, GameState};

use crate::{state::AppState, theme, ws::WsChannels};

use super::action_panel::action_panel;
use super::helpers::section_header;

pub(super) fn right_panel_contents(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
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
