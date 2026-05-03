use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_egui::egui;
use egui::RichText;

use crate::{
    state::{AppState, Screen},
    theme,
};

use super::helpers::neon_button;

pub(super) fn main_menu_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    exit_writer: &mut MessageWriter<AppExit>,
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

                ui.add_space(60.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(300.0);
                    ui.spacing_mut().item_spacing.y = 12.0;

                    if let Some(ref username) = state.auth_username.clone() {
                        ui.label(
                            RichText::new(format!("● {username}"))
                                .color(theme::NEON_CYAN)
                                .monospace()
                                .small(),
                        );
                    }

                    if ui
                        .add(neon_button("[ LOCAL PLAY ]", theme::NEON_GREEN))
                        .clicked()
                    {
                        state.screen = Screen::LocalSetup;
                    }

                    if ui
                        .add(neon_button("[ ONLINE PLAY ]", theme::NEON_CYAN))
                        .clicked()
                    {
                        if state.auth_token.is_some() {
                            state.pending_connect = true;
                            state.screen = Screen::RoomBrowser;
                        } else {
                            state.screen = Screen::Login;
                        }
                    }

                    if state.auth_token.is_some()
                        && ui
                            .add(neon_button("[ LOG OUT ]", theme::NEON_RED))
                            .clicked()
                    {
                        state.trigger_logout();
                    }

                    if ui
                        .add(neon_button("[ QUIT GAME ]", theme::NEON_RED))
                        .clicked()
                    {
                        exit_writer.write(AppExit::Success);
                    }
                });
            });
        });
}
