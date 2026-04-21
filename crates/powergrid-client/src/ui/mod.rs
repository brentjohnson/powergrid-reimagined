mod action_panel;
mod connect;
mod helpers;
mod left_panel;
mod lobby;
mod phase_tracker;
mod right_panel;
mod top_panel;

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
// One-time theme setup
// ---------------------------------------------------------------------------

pub fn setup_egui_theme(mut contexts: EguiContexts) {
    theme::apply(contexts.ctx_mut());
}

// ---------------------------------------------------------------------------
// Main UI system (runs every frame)
// ---------------------------------------------------------------------------

pub fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<AppState>,
    channels: Option<Res<WsChannels>>,
    mut commands: Commands,
) {
    let ctx = contexts.ctx_mut();

    // Re-apply theme every frame so settings survive window resize etc.
    // (cheap — just copies a struct)
    theme::apply(ctx);

    match state.screen {
        Screen::Connect => {
            connect::connect_screen(ctx, &mut state, &mut commands);
        }
        Screen::Game => {
            game_screen(ctx, &mut state, &channels);
        }
    }
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
        .exact_height(130.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(6.0)),
        )
        .show(ctx, |ui| {
            let players_info: Vec<_> = gs.players.iter().map(|p| (p.id, p.color)).collect();
            top_panel::top_panel_contents(ui, gs.clone(), &state.city_history, &players_info);
        });

    // Left panel — player info
    egui::SidePanel::left("player_panel")
        .resizable(false)
        .exact_width(220.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0.0)),
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
        .exact_width(340.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                right_panel::right_panel_contents(ui, state, channels, &gs, my_id);
            });
        });

    // Central map
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(Color32::from_rgb(2, 4, 8))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            crate::map_panel::draw(ui, state, &gs, my_id);
        });
}
