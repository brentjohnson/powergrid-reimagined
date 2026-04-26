mod action_panel;
mod connect;
mod helpers;
mod left_panel;
mod lobby;
mod phase_tracker;
mod right_panel;
mod top_panel;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Color32, RichText};
use powergrid_core::types::Phase;

use crate::{
    state::{AppState, Screen},
    theme,
    ws::WsChannels,
};

// ---------------------------------------------------------------------------
// Main UI system (runs every frame via EguiContextPass)
// ---------------------------------------------------------------------------

pub fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<AppState>,
    channels: Option<Res<WsChannels>>,
    mut commands: Commands,
    mut exit_writer: MessageWriter<AppExit>,
) -> bevy::prelude::Result {
    let ctx = contexts.ctx_mut()?;

    // Re-apply theme every frame so settings survive window resize etc.
    // (cheap — just copies a struct)
    theme::apply(ctx);

    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.menu_open = !state.menu_open;
    }

    match state.screen {
        Screen::Connect => {
            connect::connect_screen(ctx, &mut state, &mut commands);
        }
        Screen::Game => {
            game_screen(ctx, &mut state, &channels);
        }
    }

    if state.menu_open {
        egui::Window::new("MENU")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                if ui
                    .add(helpers::neon_button("[ EXIT ]", theme::NEON_RED))
                    .clicked()
                {
                    exit_writer.write(AppExit::Success);
                }
                ui.add_space(4.0);
            });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Game screen
// ---------------------------------------------------------------------------

fn game_screen(ctx: &egui::Context, state: &mut AppState, channels: &Option<Res<WsChannels>>) {
    let Some(gs) = state.game_state.clone() else {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("● AWAITING UPLINK…")
                        .color(theme::NEON_AMBER)
                        .heading(),
                );
            });
        });
        return;
    };

    let my_id = state.my_id.unwrap_or_default();

    if matches!(gs.phase, Phase::Lobby) {
        lobby::lobby_screen(ctx, state, channels, &gs, my_id);
        return;
    }

    // Top panel — phase info and resource market
    egui::TopBottomPanel::top("top_panel")
        .exact_height(180.0)
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(6)),
        )
        .show(ctx, |ui| {
            top_panel::top_panel_contents(ui, gs.clone(), state, channels, my_id);
        });

    // Left panel — player info
    egui::SidePanel::left("player_panel")
        .resizable(false)
        .exact_width(220.0)
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                left_panel::left_panel_contents(ui, &gs, my_id);
            });
        });

    // Right panel — plant market, actions, event log
    egui::SidePanel::right("info_panel")
        .resizable(false)
        .exact_width(400.0)
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            let half_height = ui.available_height() / 2.0;

            // Top half: action console, scrollable, pinned to half height
            egui::ScrollArea::vertical()
                .max_height(half_height)
                .show(ui, |ui| {
                    ui.set_min_height(half_height);
                    ui.add_space(6.0);
                    right_panel::action_console_contents(ui, state, channels, &gs, my_id);
                });

            // Bottom half: event log at fixed midpoint
            right_panel::event_log_contents(ui, &gs);
        });

    // Central map
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(Color32::from_rgb(2, 4, 8))
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            crate::map_panel::draw(ui, state, &gs, my_id);
        });
}
