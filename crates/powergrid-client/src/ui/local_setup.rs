use egui::RichText;
use powergrid_core::{
    limits::MAX_PLAYER_NAME,
    types::{BotDifficulty, PlayerColor},
};

use crate::{
    local::LocalConfig,
    state::{player_color_to_egui, AppState, Screen},
    theme,
};

use super::helpers::{color_label, neon_button};
use super::UiAction;

pub(super) fn local_setup_screen(ctx: &egui::Context, state: &mut AppState, action: &mut UiAction) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            theme::draw_backdrop(ui);
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                ui.label(
                    RichText::new("LOCAL PLAY")
                        .size(theme::HEADING_L)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("OFFLINE · BOT OPPONENTS")
                        .size(theme::LABEL_S)
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );

                ui.add_space(40.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(480.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    // Name
                    ui.label(RichText::new("YOUR NAME").color(theme::TEXT_DIM).small());
                    ui.add(
                        egui::TextEdit::singleline(&mut state.local_name)
                            .char_limit(MAX_PLAYER_NAME),
                    );

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

                    // Per-bot difficulty rows
                    ui.label(
                        RichText::new("BOT OPPONENTS")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );

                    let mut remove_idx: Option<usize> = None;
                    for (i, difficulty) in state.local_bots.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("Bot {}", i + 1))
                                    .color(theme::TEXT_BRIGHT)
                                    .monospace()
                                    .small(),
                            );
                            ui.add_space(8.0);
                            egui::ComboBox::from_id_salt(format!("bot_diff_{i}"))
                                .selected_text(difficulty_label(*difficulty))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(difficulty, BotDifficulty::Easy, "Easy");
                                    ui.selectable_value(
                                        difficulty,
                                        BotDifficulty::Normal,
                                        "Normal",
                                    );
                                    ui.selectable_value(difficulty, BotDifficulty::Hard, "Hard");
                                });
                            if ui.add(neon_button("[×]", theme::NEON_AMBER)).clicked() {
                                remove_idx = Some(i);
                            }
                        });
                    }
                    if let Some(idx) = remove_idx {
                        state.local_bots.remove(idx);
                    }

                    if state.local_bots.len() < 5
                        && ui
                            .add(neon_button("[+ ADD BOT]", theme::NEON_GREEN))
                            .clicked()
                    {
                        state.local_bots.push(BotDifficulty::Normal);
                    }

                    ui.label(
                        RichText::new(format!("({} total players)", state.local_bots.len() + 1))
                            .color(theme::TEXT_DIM)
                            .small(),
                    );

                    ui.add_space(12.0);

                    ui.horizontal(|ui| {
                        if ui.add(neon_button("[ BACK ]", theme::TEXT_DIM)).clicked() {
                            state.screen = Screen::MainMenu;
                        }

                        ui.add_space(8.0);

                        let can_start =
                            !state.local_name.trim().is_empty() && !state.local_bots.is_empty();
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
                            *action = UiAction::StartLocal(LocalConfig {
                                human_name: state.local_name.trim().to_string(),
                                human_color: state.local_color,
                                bots: state.local_bots.clone(),
                            });
                            state.pending_connect = true;
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

fn difficulty_label(d: BotDifficulty) -> &'static str {
    match d {
        BotDifficulty::Easy => "Easy",
        BotDifficulty::Normal => "Normal",
        BotDifficulty::Hard => "Hard",
    }
}
