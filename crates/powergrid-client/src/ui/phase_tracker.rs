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
        Phase::Auction { .. } => Some(Dp::Auction),
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

    theme::neon_frame().show(ui, |ui| {
        for (dp, label) in &phases {
            let is_current = current == Some(*dp);

            let player_ids: Vec<PlayerId> = if *dp == Dp::Auction {
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
                    Phase::BuyResources { remaining }
                    | Phase::BuildCities { remaining }
                    | Phase::Bureaucracy { remaining } => remaining.first().copied(),
                    _ => None,
                }
            };

            ui.horizontal(|ui| {
                let label_color = if is_current {
                    theme::NEON_AMBER
                } else {
                    theme::TEXT_DIM
                };
                let prefix = if is_current { "▶ " } else { "  " };
                ui.label(
                    RichText::new(format!("{prefix}{label}"))
                        .color(label_color)
                        .small()
                        .monospace(),
                );

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
                        let dot_text = if is_active { "◉" } else { "●" };
                        ui.label(RichText::new(dot_text).color(color).small().monospace());
                    }
                }
            });
        }
    });
}
