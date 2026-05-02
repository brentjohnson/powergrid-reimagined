use bevy::prelude::Commands;
use bevy_egui::egui;
use egui::RichText;

use crate::{
    auth::{do_logout, AuthEvent},
    state::AppState,
    theme, ws,
};

pub(super) fn connect_screen(ctx: &egui::Context, state: &mut AppState, commands: &mut Commands) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0)),
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
                    ui.set_width(420.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    // Logged-in identity
                    if let Some(ref username) = state.auth_username.clone() {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("● {}", username))
                                    .color(theme::NEON_CYAN)
                                    .monospace(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("[ LOG OUT ]")
                                                    .color(theme::NEON_RED)
                                                    .small()
                                                    .monospace(),
                                            )
                                            .fill(egui::Color32::TRANSPARENT)
                                            .stroke(egui::Stroke::new(1.0, theme::NEON_RED)),
                                        )
                                        .clicked()
                                    {
                                        handle_logout(state);
                                    }
                                },
                            );
                        });
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                    }

                    // Server name
                    ui.label(RichText::new("SERVER NAME").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.server_name);

                    ui.add_space(8.0);

                    let can_connect = state.auth_token.is_some() && !state.pending_connect;
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
                        state.pending_connect = true;
                        let url = state.ws_url();
                        let channels = ws::spawn_ws(url);
                        commands.insert_resource(channels);
                    }
                });

                if state.pending_connect {
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

fn handle_logout(state: &mut AppState) {
    // Fire-and-forget logout request in background
    if let (Some(token), server, port) = (
        state.auth_token.clone(),
        state.server_name.clone(),
        state.port,
    ) {
        let slot = state.auth_pending.0.clone();
        std::thread::spawn(move || {
            do_logout(&server, port, &token);
            *slot.lock().unwrap() = Some(AuthEvent::LoggedOut);
        });
        state.auth_in_flight = true;
    } else {
        state.logout();
    }
}
