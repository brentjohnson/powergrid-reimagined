use bevy::prelude::Res;
use bevy_egui::egui;
use egui::RichText;
use powergrid_core::{actions::Action, types::PlayerId, GameState};

use crate::{
    state::{player_color_to_egui, AppState},
    theme,
    ws::WsChannels,
};

use super::helpers::send;

pub(super) fn lobby_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(theme::BG_DEEP))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.label(
                    RichText::new("GRID LOBBY")
                        .size(32.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.add_space(30.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(400.0);
                    ui.label(
                        RichText::new("CONNECTED OPERATORS")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.add_space(8.0);

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

                let is_host = gs.host_id() == Some(my_id);
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
                        send(Action::StartGame, channels);
                    }
                } else {
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
