use bevy::prelude::Res;
use bevy_egui::egui;
use egui::RichText;
use powergrid_core::actions::LobbyAction;

use crate::{state::AppState, theme, ws::WsChannels};

use super::helpers::send_lobby;

pub(super) fn room_browser_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(theme::BG_DEEP))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.label(
                    RichText::new("POWER GRID")
                        .size(36.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("REIMAGINED")
                        .size(18.0)
                        .color(theme::NEON_CYAN_DIM)
                        .monospace(),
                );
                ui.add_space(30.0);

                // Create / join room form
                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(440.0);
                    ui.spacing_mut().item_spacing.y = 8.0;

                    ui.label(
                        RichText::new("CREATE OR JOIN ROOM")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Room name:").color(theme::TEXT_DIM));
                        ui.text_edit_singleline(&mut state.room_name_input);
                    });

                    let room_name = state.room_name_input.trim().to_string();
                    let valid = !room_name.is_empty();

                    ui.horizontal(|ui| {
                        let create_btn = egui::Button::new(
                            RichText::new("[ CREATE ]")
                                .color(if valid {
                                    theme::BG_DEEP
                                } else {
                                    theme::TEXT_DIM
                                })
                                .monospace(),
                        )
                        .fill(if valid {
                            theme::NEON_CYAN
                        } else {
                            theme::BG_WIDGET
                        })
                        .stroke(egui::Stroke::new(
                            1.5,
                            if valid {
                                theme::NEON_CYAN
                            } else {
                                theme::NEON_CYAN_DARK
                            },
                        ));

                        if ui.add_enabled(valid, create_btn).clicked() {
                            send_lobby(
                                LobbyAction::CreateRoom {
                                    name: room_name.clone(),
                                },
                                channels,
                            );
                        }

                        let join_btn = egui::Button::new(
                            RichText::new("[ JOIN ]")
                                .color(if valid {
                                    theme::BG_DEEP
                                } else {
                                    theme::TEXT_DIM
                                })
                                .monospace(),
                        )
                        .fill(if valid {
                            theme::NEON_AMBER
                        } else {
                            theme::BG_WIDGET
                        })
                        .stroke(egui::Stroke::new(
                            1.5,
                            if valid {
                                theme::NEON_AMBER
                            } else {
                                theme::NEON_CYAN_DARK
                            },
                        ));

                        if ui.add_enabled(valid, join_btn).clicked() {
                            send_lobby(LobbyAction::JoinRoom { name: room_name }, channels);
                        }
                    });
                });

                ui.add_space(20.0);

                // Room list
                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(440.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("ACTIVE ROOMS").color(theme::TEXT_DIM).small());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("[ REFRESH ]")
                                            .color(theme::NEON_CYAN)
                                            .monospace(),
                                    )
                                    .fill(theme::BG_WIDGET)
                                    .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN)),
                                )
                                .clicked()
                            {
                                send_lobby(LobbyAction::ListRooms, channels);
                            }
                        });
                    });
                    ui.add_space(8.0);

                    if state.room_list.is_empty() {
                        ui.label(
                            RichText::new("No active rooms — create one above.")
                                .color(theme::TEXT_DIM)
                                .small(),
                        );
                    } else {
                        for room in &state.room_list.clone() {
                            ui.horizontal(|ui| {
                                let status_color = if room.has_started {
                                    theme::NEON_AMBER
                                } else {
                                    theme::NEON_GREEN
                                };
                                let status_label =
                                    if room.has_started { "IN GAME" } else { "LOBBY" };
                                ui.label(
                                    RichText::new(format!(
                                        "■  {}  ({}/{})  [{}]",
                                        room.name,
                                        room.player_count,
                                        room.max_players,
                                        status_label,
                                    ))
                                    .color(status_color)
                                    .monospace(),
                                );
                                if !room.has_started {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let room_name = room.name.clone();
                                            if ui
                                                .add(
                                                    egui::Button::new(
                                                        RichText::new("JOIN")
                                                            .color(theme::BG_DEEP)
                                                            .monospace(),
                                                    )
                                                    .fill(theme::NEON_CYAN),
                                                )
                                                .clicked()
                                            {
                                                state.room_name_input = room_name.clone();
                                                send_lobby(
                                                    LobbyAction::JoinRoom { name: room_name },
                                                    channels,
                                                );
                                            }
                                        },
                                    );
                                }
                            });
                        }
                    }
                });

                if let Some(err) = &state.error_message.clone() {
                    ui.add_space(12.0);
                    ui.label(RichText::new(format!("⚠ {err}")).color(theme::NEON_RED));
                    if ui
                        .add(
                            egui::Button::new(RichText::new("[ DISMISS ]").color(theme::TEXT_DIM))
                                .fill(theme::BG_WIDGET),
                        )
                        .clicked()
                    {
                        state.error_message = None;
                    }
                }
            });
        });
}
