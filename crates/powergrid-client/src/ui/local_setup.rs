use bevy::prelude::Commands;
use bevy_egui::egui;
use egui::RichText;
use powergrid_core::types::PlayerColor;

use crate::{
    local::LocalConfig,
    state::{player_color_to_egui, AppState, Screen},
    theme,
};

use super::helpers::{color_label, neon_button};

pub(super) fn local_setup_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    commands: &mut Commands,
) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                ui.label(
                    RichText::new("LOCAL PLAY")
                        .size(32.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("OFFLINE · BOT OPPONENTS")
                        .size(14.0)
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );

                ui.add_space(40.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(420.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    // Name
                    ui.label(RichText::new("YOUR NAME").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.local_name);

                    ui.add_space(4.0);

                    // Color picker
                    ui.label(RichText::new("YOUR COLOR").color(theme::TEXT_DIM).small());
                    ui.horizontal(|ui| {
                        for color in [
                            PlayerColor::Red,
                            PlayerColor::Blue,
                            PlayerColor::Green,
                            PlayerColor::Yellow,
                            PlayerColor::Purple,
                            PlayerColor::White,
                        ] {
                            let egui_color = player_color_to_egui(color);
                            let selected = state.local_color == color;
                            let btn = egui::Button::new(RichText::new(color_label(color)).color(
                                if selected {
                                    egui::Color32::BLACK
                                } else {
                                    egui_color
                                },
                            ))
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
                                state.local_color = color;
                            }
                        }
                    });

                    ui.add_space(4.0);

                    // Bot count
                    ui.label(
                        RichText::new("BOT OPPONENTS")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.horizontal(|ui| {
                        let count = state.local_bot_count;
                        ui.label(
                            RichText::new(format!("{count}"))
                                .color(theme::TEXT_BRIGHT)
                                .monospace()
                                .size(16.0),
                        );
                        ui.add_space(8.0);
                        if ui
                            .add_enabled(count > 1, neon_button("[-]", theme::NEON_AMBER))
                            .clicked()
                        {
                            state.local_bot_count -= 1;
                        }
                        if ui
                            .add_enabled(count < 5, neon_button("[+]", theme::NEON_GREEN))
                            .clicked()
                        {
                            state.local_bot_count += 1;
                        }
                        ui.label(
                            RichText::new(format!("({} total players)", count + 1))
                                .color(theme::TEXT_DIM)
                                .small(),
                        );
                    });

                    ui.add_space(12.0);

                    ui.horizontal(|ui| {
                        if ui.add(neon_button("[ BACK ]", theme::TEXT_DIM)).clicked() {
                            state.screen = Screen::MainMenu;
                        }

                        ui.add_space(8.0);

                        let can_start = !state.local_name.trim().is_empty();
                        let start_btn = egui::Button::new(
                            RichText::new("[ START LOCAL GAME ]")
                                .color(if can_start {
                                    theme::BG_DEEP
                                } else {
                                    theme::TEXT_DIM
                                })
                                .monospace(),
                        )
                        .fill(if can_start {
                            theme::NEON_GREEN
                        } else {
                            theme::BG_WIDGET
                        })
                        .stroke(egui::Stroke::new(
                            1.5,
                            if can_start {
                                theme::NEON_GREEN
                            } else {
                                theme::NEON_CYAN_DARK
                            },
                        ));

                        if ui.add_enabled(can_start, start_btn).clicked() {
                            let cfg = LocalConfig {
                                human_name: state.local_name.trim().to_string(),
                                human_color: state.local_color,
                                bot_count: state.local_bot_count,
                            };
                            state.pending_connect = true;
                            state.local_expected_players = cfg.bot_count + 1;
                            let (channels, handle) = crate::local::start_local_session(cfg);
                            commands.insert_resource(channels);
                            commands.insert_resource(handle);
                        }
                    });

                    if state.pending_connect {
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("● INITIALIZING LOCAL GRID…")
                                .color(theme::NEON_AMBER)
                                .monospace(),
                        );
                    }
                });
            });
        });
}
