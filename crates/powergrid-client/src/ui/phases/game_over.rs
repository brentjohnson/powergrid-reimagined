use egui::{Align, Align2, Label, Layout, RichText};
use powergrid_core::{types::PlayerId, GameStateView};

use crate::{state::player_color_to_egui, theme};

const ROW_H: f32 = 16.0;
// name, MONEY, POWERED, CITIES, CAP
const COL_WIDTHS: [f32; 5] = [150.0, 70.0, 80.0, 70.0, 60.0];

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

                // Column headers — using the same fixed widths as the data rows
                let headers = ["", "MONEY", "POWERED", "CITIES", "CAP"];
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    for (i, (&width, &label)) in COL_WIDTHS.iter().zip(headers.iter()).enumerate() {
                        let text = RichText::new(label)
                            .size(11.0)
                            .color(theme::TEXT_BRIGHT)
                            .monospace();
                        if i == 0 {
                            ui.add_sized([width, ROW_H], Label::new(text));
                        } else {
                            ui.allocate_ui_with_layout(
                                egui::vec2(width, ROW_H),
                                Layout::right_to_left(Align::Center),
                                |ui| {
                                    ui.add(Label::new(text));
                                },
                            );
                        }
                    }
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

                    let cells: [(&str, egui::Color32, String); 5] = [
                        (
                            "name",
                            player_egui_color,
                            format!("{} {}", ordinal(rank + 1), p.name),
                        ),
                        ("money", theme::NEON_GREEN, format!("${}", p.money)),
                        (
                            "powered",
                            theme::TEXT_BRIGHT,
                            format!("{}", p.last_cities_powered),
                        ),
                        ("cities", theme::TEXT_BRIGHT, format!("{cities_owned}")),
                        ("cap", theme::TEXT_BRIGHT, format!("{capacity}")),
                    ];

                    egui::Frame::NONE
                        .fill(theme::BG_PANEL)
                        .stroke(egui::Stroke::new(stroke_width, stroke_color))
                        .inner_margin(egui::Margin::same(6))
                        .corner_radius(egui::CornerRadius::same(3))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 0.0;
                                for (i, (&width, (_, color, text))) in
                                    COL_WIDTHS.iter().zip(cells.iter()).enumerate()
                                {
                                    let rich = RichText::new(text).monospace().color(*color);
                                    if i == 0 {
                                        ui.add_sized([width, ROW_H], Label::new(rich));
                                    } else {
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(width, ROW_H),
                                            Layout::right_to_left(Align::Center),
                                            |ui| {
                                                ui.add(Label::new(rich));
                                            },
                                        );
                                    }
                                }
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
