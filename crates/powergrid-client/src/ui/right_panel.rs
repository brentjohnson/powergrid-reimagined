use bevy::prelude::Res;
use egui::{text::LayoutJob, Color32, FontId, RichText, TextFormat, Ui};
use powergrid_core::{types::PlayerId, GameStateView};

use crate::{
    state::{player_color_to_egui, AppState},
    theme,
    ws::WsChannels,
};

use super::action_panel::action_panel;
use super::helpers::section_header;

pub(super) fn action_console_contents(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameStateView,
    my_id: PlayerId,
) {
    if gs.player(my_id).is_some() {
        section_header(ui, "ACTION CONSOLE");
        theme::neon_frame_bright().show(ui, |ui| {
            if let Some(err) = &state.error_message.clone() {
                ui.label(
                    RichText::new(format!("⚠ {err}"))
                        .color(theme::NEON_RED)
                        .small()
                        .monospace(),
                );
                ui.add_space(4.0);
            }
            action_panel(ui, state, channels, gs, my_id);
        });
    }
}

pub(super) fn event_log_contents(ui: &mut Ui, gs: &GameStateView) {
    section_header(ui, "EVENT LOG");

    // Build (name, color) list sorted longest-first so longer names match before prefixes.
    let mut players: Vec<(&str, Color32)> = gs
        .players
        .iter()
        .map(|p| (p.name.as_str(), player_color_to_egui(p.color)))
        .collect();
    players.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let font = FontId::monospace(egui::TextStyle::Small.resolve(ui.style()).size);

    theme::neon_frame().show(ui, |ui| {
        for entry in gs.event_log.iter().rev().take(16) {
            let mut job = LayoutJob::default();
            let mut remaining = entry.as_str();

            while !remaining.is_empty() {
                // Find the earliest player-name match.
                let best = players
                    .iter()
                    .filter_map(|(name, color)| {
                        remaining.find(name).map(|pos| (pos, *name, *color))
                    })
                    .min_by_key(|(pos, _, _)| *pos);

                if let Some((pos, name, color)) = best {
                    // Text before the match.
                    if pos > 0 {
                        job.append(
                            &remaining[..pos],
                            0.0,
                            TextFormat {
                                font_id: font.clone(),
                                color: theme::TEXT_DIM,
                                ..Default::default()
                            },
                        );
                    }
                    // The player name in their color.
                    job.append(
                        name,
                        0.0,
                        TextFormat {
                            font_id: font.clone(),
                            color,
                            ..Default::default()
                        },
                    );
                    remaining = &remaining[pos + name.len()..];
                } else {
                    // No more names — append the rest dimly.
                    job.append(
                        remaining,
                        0.0,
                        TextFormat {
                            font_id: font.clone(),
                            color: theme::TEXT_DIM,
                            ..Default::default()
                        },
                    );
                    break;
                }
            }

            ui.label(job);
        }
    });
}
