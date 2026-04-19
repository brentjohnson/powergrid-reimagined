use bevy::prelude::Commands;
use bevy_egui::egui;
use egui::{Color32, RichText};
use powergrid_core::types::PlayerColor;

use crate::{
    state::{player_color_to_egui, AppState, Screen},
    theme, ws,
};

use super::helpers::color_label;

pub(super) fn connect_screen(ctx: &egui::Context, state: &mut AppState, commands: &mut Commands) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                // Title
                ui.label(
                    RichText::new("POWER GRID")
                        .size(42.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("REIMAGINED")
                        .size(20.0)
                        .color(theme::NEON_CYAN_DIM)
                        .monospace(),
                );

                ui.add_space(40.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(360.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    // Server URL
                    ui.label(RichText::new("SERVER URL").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.connect_url);

                    // Player name
                    ui.label(RichText::new("CALLSIGN").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.player_name);

                    // Color selector
                    ui.label(
                        RichText::new("FACTION COLOR")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.horizontal_wrapped(|ui| {
                        for color in [
                            PlayerColor::Red,
                            PlayerColor::Blue,
                            PlayerColor::Green,
                            PlayerColor::Yellow,
                            PlayerColor::Purple,
                            PlayerColor::White,
                        ] {
                            let egui_color = player_color_to_egui(color);
                            let selected = state.selected_color == color;
                            let label = color_label(color);

                            let btn = egui::Button::new(RichText::new(label).color(if selected {
                                Color32::BLACK
                            } else {
                                egui_color
                            }))
                            .fill(if selected {
                                egui_color
                            } else {
                                theme::BG_WIDGET
                            })
                            .stroke(egui::Stroke::new(
                                if selected { 2.0 } else { 1.0 },
                                egui_color,
                            ));

                            if ui.add(btn).clicked() {
                                state.selected_color = color;
                            }
                        }
                    });

                    ui.add_space(8.0);

                    let can_connect = !state.player_name.trim().is_empty();
                    let connect_btn = egui::Button::new(
                        RichText::new("[ CONNECT ]")
                            .color(if can_connect {
                                theme::BG_DEEP
                            } else {
                                theme::TEXT_DIM
                            })
                            .monospace(),
                    )
                    .fill(if can_connect {
                        theme::NEON_CYAN
                    } else {
                        theme::BG_WIDGET
                    })
                    .stroke(egui::Stroke::new(
                        1.5,
                        if can_connect {
                            theme::NEON_CYAN
                        } else {
                            theme::NEON_CYAN_DARK
                        },
                    ));

                    if ui.add_enabled(can_connect, connect_btn).clicked() {
                        let url = state.connect_url.clone();
                        let name = state.player_name.trim().to_string();
                        let color = state.selected_color;
                        state.pending_join = Some((name, color));
                        let channels = ws::spawn_ws(url);
                        commands.insert_resource(channels);
                    }
                });

                if !state.connected
                    && state.pending_join.is_none()
                    && state.game_state.is_none()
                    && state.screen == Screen::Connect
                {
                    // No error to show yet
                } else if !state.connected && state.pending_join.is_some() {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("● CONNECTING…")
                            .color(theme::NEON_AMBER)
                            .monospace(),
                    );
                }
            });
        });
}
