use bevy::prelude::Res;
use bevy_egui::egui;
use egui::RichText;
use powergrid_core::{
    actions::{Action, LobbyAction},
    types::{PlayerColor, PlayerId},
    GameStateView,
};

use crate::{
    state::{player_color_to_egui, AppState},
    theme,
    ws::WsChannels,
};

use super::helpers::{color_label, send, send_lobby};

pub(super) fn lobby_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    let room = state.current_room.as_deref();
    let already_joined = gs.players.iter().any(|p| p.id == my_id);
    let is_host = gs.host_id() == Some(my_id);

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(theme::BG_DEEP))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.label(
                    RichText::new("GRID LOBBY")
                        .size(32.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                if let Some(r) = room {
                    ui.label(
                        RichText::new(format!("ROOM: {}", r))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                }
                ui.add_space(20.0);

                // Connected operators
                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(400.0);
                    ui.label(
                        RichText::new("CONNECTED OPERATORS")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.add_space(8.0);

                    if gs.players.is_empty() {
                        ui.label(
                            RichText::new("No operators have joined yet.")
                                .color(theme::TEXT_DIM)
                                .small(),
                        );
                    }

                    for player in &gs.players {
                        ui.horizontal(|ui| {
                            let c = player_color_to_egui(player.color);
                            ui.colored_label(c, format!("■  {}", player.name));
                            if player.id == my_id {
                                ui.label(RichText::new("(you)").color(theme::TEXT_DIM).small());
                            }
                        });
                    }
                });

                ui.add_space(20.0);

                // If not yet joined as a player, show color picker + join button.
                if !already_joined {
                    theme::neon_frame().show(ui, |ui| {
                        ui.set_width(400.0);
                        ui.label(
                            RichText::new("PICK YOUR FACTION COLOR")
                                .color(theme::TEXT_DIM)
                                .small(),
                        );
                        ui.add_space(8.0);

                        let taken: std::collections::HashSet<PlayerColor> =
                            gs.players.iter().map(|p| p.color).collect();

                        ui.horizontal(|ui| {
                            for color in [
                                PlayerColor::Red,
                                PlayerColor::Blue,
                                PlayerColor::Green,
                                PlayerColor::Yellow,
                                PlayerColor::Purple,
                                PlayerColor::White,
                            ] {
                                let available = !taken.contains(&color);
                                let egui_color = if available {
                                    player_color_to_egui(color)
                                } else {
                                    // Desaturated for taken colors
                                    let c = player_color_to_egui(color);
                                    egui::Color32::from_rgba_unmultiplied(
                                        (c.r() as f32 * 0.25) as u8,
                                        (c.g() as f32 * 0.25) as u8,
                                        (c.b() as f32 * 0.25) as u8,
                                        120,
                                    )
                                };
                                let selected = state.selected_color == color;
                                let btn = egui::Button::new(
                                    RichText::new(color_label(color)).color(if selected {
                                        egui::Color32::BLACK
                                    } else {
                                        egui_color
                                    }),
                                )
                                .fill(if selected {
                                    egui_color
                                } else {
                                    theme::BG_WIDGET
                                })
                                .stroke(egui::Stroke::new(
                                    if selected { 2.0 } else { 1.0 },
                                    egui_color,
                                ));
                                if ui.add_enabled(available, btn).clicked() {
                                    state.selected_color = color;
                                }
                            }
                        });

                        ui.add_space(8.0);

                        let chosen_free = !taken.contains(&state.selected_color);
                        let can_join = chosen_free && gs.players.len() < 6;
                        if ui
                            .add_enabled(
                                can_join,
                                egui::Button::new(
                                    RichText::new("[ JOIN GAME ]")
                                        .color(if can_join {
                                            theme::BG_DEEP
                                        } else {
                                            theme::TEXT_DIM
                                        })
                                        .monospace(),
                                )
                                .fill(if can_join {
                                    theme::NEON_CYAN
                                } else {
                                    theme::BG_WIDGET
                                })
                                .stroke(egui::Stroke::new(
                                    1.5,
                                    if can_join {
                                        theme::NEON_CYAN
                                    } else {
                                        theme::NEON_CYAN_DARK
                                    },
                                )),
                            )
                            .clicked()
                        {
                            let name = state
                                .auth_username
                                .clone()
                                .unwrap_or_else(|| "Operator".to_string());
                            let color = state.selected_color;
                            send(Action::JoinGame { name, color }, room, channels);
                        }
                    });

                    ui.add_space(20.0);
                }

                // Add Bot section (host only, while in lobby phase)
                if is_host {
                    theme::neon_frame().show(ui, |ui| {
                        ui.set_width(400.0);
                        ui.label(RichText::new("ADD BOT").color(theme::TEXT_DIM).small());
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Name:").color(theme::TEXT_DIM).small());
                            ui.text_edit_singleline(&mut state.bot_name_input);
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Color:").color(theme::TEXT_DIM).small());
                            let taken: std::collections::HashSet<PlayerColor> =
                                gs.players.iter().map(|p| p.color).collect();
                            for color in [
                                PlayerColor::Red,
                                PlayerColor::Blue,
                                PlayerColor::Green,
                                PlayerColor::Yellow,
                                PlayerColor::Purple,
                                PlayerColor::White,
                            ] {
                                let available = !taken.contains(&color);
                                let egui_color = player_color_to_egui(color);
                                let selected = state.bot_color_input == color;
                                let btn = egui::Button::new(
                                    RichText::new(color_label(color)).color(if selected {
                                        egui::Color32::BLACK
                                    } else {
                                        egui_color
                                    }),
                                )
                                .fill(if selected {
                                    egui_color
                                } else {
                                    theme::BG_WIDGET
                                })
                                .stroke(egui::Stroke::new(
                                    if selected { 2.0 } else { 1.0 },
                                    egui_color,
                                ));
                                if ui.add_enabled(available, btn).clicked() {
                                    state.bot_color_input = color;
                                }
                            }
                        });
                        ui.add_space(4.0);
                        let can_add_bot =
                            !state.bot_name_input.trim().is_empty() && gs.players.len() < 6;
                        if ui
                            .add_enabled(
                                can_add_bot,
                                egui::Button::new(
                                    RichText::new("[ ADD BOT ]")
                                        .color(if can_add_bot {
                                            theme::BG_DEEP
                                        } else {
                                            theme::TEXT_DIM
                                        })
                                        .monospace(),
                                )
                                .fill(if can_add_bot {
                                    theme::NEON_AMBER
                                } else {
                                    theme::BG_WIDGET
                                }),
                            )
                            .clicked()
                        {
                            let bot_name = state.bot_name_input.trim().to_string();
                            let bot_color = state.bot_color_input;
                            send_lobby(
                                LobbyAction::AddBot {
                                    bot_name,
                                    color: bot_color,
                                },
                                channels,
                            );
                            state.bot_name_input.clear();
                        }
                    });

                    ui.add_space(20.0);
                }

                // Start game button (host only)
                if is_host {
                    let enough = gs.players.len() >= 2;
                    let btn_text = if enough {
                        "[ INITIALIZE GRID ]"
                    } else {
                        "[ WAITING FOR OPERATORS ]"
                    };
                    let btn = egui::Button::new(
                        RichText::new(btn_text)
                            .color(if enough {
                                theme::BG_DEEP
                            } else {
                                theme::TEXT_DIM
                            })
                            .monospace(),
                    )
                    .fill(if enough {
                        theme::NEON_GREEN
                    } else {
                        theme::BG_WIDGET
                    })
                    .stroke(egui::Stroke::new(
                        1.5,
                        if enough {
                            theme::NEON_GREEN
                        } else {
                            theme::NEON_CYAN_DARK
                        },
                    ));

                    if ui.add_enabled(enough, btn).clicked() {
                        send(Action::StartGame, room, channels);
                    }
                } else if already_joined {
                    ui.label(
                        RichText::new("● AWAITING HOST INITIALIZATION…")
                            .color(theme::NEON_AMBER)
                            .monospace(),
                    );
                }

                if let Some(err) = &state.error_message {
                    ui.add_space(12.0);
                    ui.label(RichText::new(format!("⚠ {err}")).color(theme::NEON_RED));
                }
            });
        });
}
