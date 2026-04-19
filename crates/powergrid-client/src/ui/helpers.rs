use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{Color32, RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerColor, PlayerId, Resource},
    GameState,
};

use crate::{theme, ws::WsChannels};

pub(super) fn section_header(ui: &mut Ui, label: &str) {
    ui.label(
        RichText::new(label)
            .color(theme::NEON_CYAN_DIM)
            .small()
            .monospace(),
    );
}

pub(super) fn neon_button(label: &str, color: Color32) -> egui::Button<'static> {
    egui::Button::new(RichText::new(label).color(color).monospace())
        .fill(theme::BG_WIDGET)
        .stroke(egui::Stroke::new(1.0, color))
}

pub(super) fn resource_badge(ui: &mut Ui, label: &str, count: u8, color: Color32) {
    egui::Frame::none()
        .fill(theme::BG_WIDGET)
        .stroke(egui::Stroke::new(1.0, color))
        .inner_margin(egui::Margin::symmetric(6.0, 3.0))
        .rounding(egui::Rounding::same(2.0))
        .show(ui, |ui| {
            ui.label(
                RichText::new(format!("{label}\n{count:>2}"))
                    .color(color)
                    .small()
                    .monospace(),
            );
        });
}

pub(super) fn dim_color(c: Color32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (c.r() as f32 * 0.3) as u8,
        (c.g() as f32 * 0.3) as u8,
        (c.b() as f32 * 0.3) as u8,
        180,
    )
}

pub(super) fn phase_name(phase: &Phase) -> &'static str {
    match phase {
        Phase::Lobby => "LOBBY",
        Phase::PlayerOrder => "PLAYER ORDER",
        Phase::Auction { .. } => "AUCTION",
        Phase::BuyResources { .. } => "BUY RESOURCES",
        Phase::BuildCities { .. } => "BUILD",
        Phase::Bureaucracy { .. } => "BUREAUCRACY",
        Phase::GameOver { .. } => "GAME OVER",
    }
}

pub(super) fn is_active_player(gs: &GameState, pid: PlayerId) -> bool {
    match &gs.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            ..
        } => {
            if let Some(bid) = active_bid {
                bid.remaining_bidders.first() == Some(&pid)
            } else {
                gs.player_order.get(*current_bidder_idx) == Some(&pid)
            }
        }
        Phase::BuyResources { remaining }
        | Phase::BuildCities { remaining }
        | Phase::Bureaucracy { remaining } => remaining.first() == Some(&pid),
        _ => false,
    }
}

pub(super) fn resource_name(r: Resource) -> &'static str {
    match r {
        Resource::Coal => "COAL",
        Resource::Oil => "OIL",
        Resource::Garbage => "GARBAGE",
        Resource::Uranium => "URANIUM",
    }
}

pub(super) fn color_label(c: PlayerColor) -> &'static str {
    match c {
        PlayerColor::Red => "RED",
        PlayerColor::Blue => "BLUE",
        PlayerColor::Green => "GREEN",
        PlayerColor::Yellow => "YELLOW",
        PlayerColor::Purple => "PURPLE",
        PlayerColor::White => "WHITE",
    }
}

pub(super) fn send(action: Action, channels: &Option<Res<WsChannels>>) {
    if let Some(ch) = channels {
        ch.action_tx.send(action).ok();
    }
}
