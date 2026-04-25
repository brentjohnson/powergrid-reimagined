use bevy_egui::egui;
use egui::{RichText, Ui};
use powergrid_core::{
    types::{Phase, PlayerId},
    GameState,
};

use crate::{state::player_color_to_egui, theme};

use super::helpers::dim_color;

pub(super) fn phase_tracker(ui: &mut Ui, gs: &GameState) {
    #[derive(Clone, Copy, PartialEq)]
    enum Dp {
        Auction,
        Resource,
        Build,
        Bureaucracy,
    }

    let current = match &gs.phase {
        Phase::Auction { .. } | Phase::DiscardPlant { .. } => Some(Dp::Auction),
        Phase::BuyResources { .. } => Some(Dp::Resource),
        Phase::BuildCities { .. } => Some(Dp::Build),
        Phase::Bureaucracy { .. } => Some(Dp::Bureaucracy),
        _ => None,
    };

    let phases = [
        (Dp::Auction, "AUCTION"),
        (Dp::Resource, "RESOURCES"),
        (Dp::Build, "BUILD"),
        (Dp::Bureaucracy, "BUREAUCRACY"),
    ];

    // Measure the widest label to use as a fixed column width
    let col_width = phases
        .iter()
        .map(|(_, label)| {
            ui.fonts(|f| {
                f.layout_no_wrap(
                    label.to_string(),
                    egui::FontId::monospace(ui.style().text_styles[&egui::TextStyle::Small].size),
                    egui::Color32::WHITE,
                )
                .size()
                .x
            })
        })
        .fold(0.0_f32, f32::max)
        + ui.spacing().item_spacing.x * 2.0
        + 4.0;

    let render_phase = |ui: &mut Ui, dp: Dp, label: &str| {
        let is_current = current == Some(dp);

        let player_ids: Vec<PlayerId> = if dp == Dp::Auction {
            gs.player_order.clone()
        } else {
            gs.player_order.iter().rev().cloned().collect()
        };

        let phase_active: Option<PlayerId> = if !is_current {
            None
        } else {
            match &gs.phase {
                Phase::Auction {
                    current_bidder_idx, ..
                } => gs.player_order.get(*current_bidder_idx).copied(),
                Phase::BuyResources { remaining } | Phase::BuildCities { remaining } => {
                    remaining.first().copied()
                }
                Phase::Bureaucracy { .. } => None,
                _ => None,
            }
        };

        ui.vertical(|ui| {
            ui.set_width(col_width);

            let label_color = if is_current {
                theme::NEON_AMBER
            } else {
                theme::TEXT_DIM
            };
            ui.label(RichText::new(label).color(label_color).small().monospace());

            ui.horizontal(|ui| {
                for pid in &player_ids {
                    let is_active = phase_active == Some(*pid);
                    let is_completed = if !is_current {
                        false
                    } else {
                        match &gs.phase {
                            Phase::Auction { bought, passed, .. } => {
                                bought.contains(pid) || passed.contains(pid)
                            }
                            Phase::BuyResources { remaining }
                            | Phase::BuildCities { remaining }
                            | Phase::Bureaucracy { remaining } => !remaining.contains(pid),
                            _ => false,
                        }
                    };

                    if let Some(p) = gs.player(*pid) {
                        let base = player_color_to_egui(p.color);
                        let color = if !is_current || is_completed {
                            dim_color(base)
                        } else {
                            base
                        };
                        let square_text = if is_active { "▣" } else { "■" };
                        ui.label(
                            RichText::new(square_text)
                                .color(color)
                                .monospace()
                                .size(20.0),
                        );
                    }
                }
            });
        });
    };

    ui.vertical(|ui| {
        super::helpers::section_header(ui, "PHASE TRACKING");
        theme::neon_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                render_phase(ui, Dp::Auction, "AUCTION");
                render_phase(ui, Dp::Resource, "RESOURCES");
            });
            ui.horizontal(|ui| {
                render_phase(ui, Dp::Build, "BUILD");
                render_phase(ui, Dp::Bureaucracy, "BUREAUCRACY");
            });
        });
    });
}
