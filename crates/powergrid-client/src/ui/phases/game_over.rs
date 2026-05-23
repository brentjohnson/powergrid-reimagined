use egui::{Align2, Grid, RichText};
use powergrid_core::{types::PlayerId, GameStateView};

use crate::{state::player_color_to_egui, theme};

pub(in crate::ui) fn game_over_overlay(ctx: &egui::Context, gs: &GameStateView, winner: PlayerId) {
    // Build ranking: sort by (cities_powered desc, money desc, cities_owned desc)
    let mut ranked: Vec<_> = gs.players.iter().collect();
    ranked.sort_by(|a, b| {
        let key_a = (a.last_cities_powered, a.money, gs.player_city_count(a.id));
        let key_b = (b.last_cities_powered, b.money, gs.player_city_count(b.id));
        key_b.cmp(&key_a)
    });

    egui::Window::new("GAME OVER")
        .collapsible(false)
        .resizable(false)
        .movable(false)
        .order(egui::Order::Foreground)
        .anchor(Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            theme::neon_frame().show(ui, |ui| {
                ui.add_space(6.0);

                // Column headers
                Grid::new("game_over_header")
                    .num_columns(5)
                    .spacing([16.0, 2.0])
                    .show(ui, |ui| {
                        for label in ["", "MONEY", "POWERED", "CITIES", "CAP"] {
                            ui.label(
                                RichText::new(label)
                                    .size(11.0)
                                    .color(theme::TEXT_DIM)
                                    .monospace(),
                            );
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);

                // Per-player rows
                for (rank, p) in ranked.iter().enumerate() {
                    let is_winner = p.id == winner;
                    let player_egui_color = player_color_to_egui(p.color);
                    let stroke_width = if is_winner { 2.0 } else { 1.0 };
                    let stroke_color = if is_winner {
                        theme::NEON_GREEN
                    } else {
                        player_egui_color
                    };

                    let cities_owned = gs.player_city_count(p.id) as u32;
                    let capacity: u32 = p.plants.iter().map(|pl| pl.cities as u32).sum();

                    egui::Frame::NONE
                        .fill(theme::BG_PANEL)
                        .stroke(egui::Stroke::new(stroke_width, stroke_color))
                        .inner_margin(egui::Margin::same(6))
                        .corner_radius(egui::CornerRadius::same(3))
                        .show(ui, |ui| {
                            Grid::new(format!("game_over_row_{rank}"))
                                .num_columns(5)
                                .spacing([16.0, 0.0])
                                .show(ui, |ui| {
                                    // Position + name
                                    let ordinal = ordinal(rank + 1);
                                    ui.label(
                                        RichText::new(format!("{ordinal} {}", p.name))
                                            .monospace()
                                            .color(player_egui_color),
                                    );
                                    // Money
                                    ui.label(
                                        RichText::new(format!("${}", p.money))
                                            .monospace()
                                            .color(theme::NEON_GREEN),
                                    );
                                    // Cities powered (final bureaucracy result)
                                    ui.label(
                                        RichText::new(format!("{}", p.last_cities_powered))
                                            .monospace()
                                            .color(theme::TEXT_DIM),
                                    );
                                    // Cities owned
                                    ui.label(
                                        RichText::new(format!("{cities_owned}"))
                                            .monospace()
                                            .color(theme::TEXT_DIM),
                                    );
                                    // Plant capacity
                                    ui.label(
                                        RichText::new(format!("{capacity}"))
                                            .monospace()
                                            .color(theme::TEXT_DIM),
                                    );
                                    ui.end_row();
                                });
                        });

                    ui.add_space(4.0);
                }
            });
        });
}

fn ordinal(n: usize) -> &'static str {
    match n {
        1 => "1ST",
        2 => "2ND",
        3 => "3RD",
        4 => "4TH",
        5 => "5TH",
        _ => "6TH",
    }
}
