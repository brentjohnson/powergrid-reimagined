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

pub(super) fn dim_color(c: Color32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (c.r() as f32 * 0.3) as u8,
        (c.g() as f32 * 0.3) as u8,
        (c.b() as f32 * 0.3) as u8,
        180,
    )
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
        Phase::BuyResources { remaining } | Phase::BuildCities { remaining } => {
            remaining.first() == Some(&pid)
        }
        Phase::Bureaucracy { remaining } => remaining.contains(&pid),
        Phase::DiscardPlant { player, .. }
        | Phase::DiscardResource { player, .. }
        | Phase::PowerCitiesFuel { player, .. } => *player == pid,
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

/// Renders one `[label: value] [-] [+] [trailing]` row used by the three batch-resource
/// prompts (BuyResources, DiscardResource, PowerCitiesFuel).  Returns -1 if the minus
/// button was clicked, +1 if plus was clicked, 0 otherwise.  Buttons are auto-disabled
/// at the `min`/`max` bounds.
pub(super) fn resource_counter_row(
    ui: &mut Ui,
    label: &str,
    value: u8,
    min: u8,
    max: u8,
    trailing: &str,
) -> i8 {
    let mut delta = 0i8;
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{label}: {value:>2}"))
                .color(theme::TEXT_BRIGHT)
                .monospace(),
        );
        if ui
            .add_enabled(value > min, neon_button("[-]", theme::NEON_AMBER))
            .clicked()
        {
            delta = -1;
        }
        if ui
            .add_enabled(value < max, neon_button("[+]", theme::NEON_GREEN))
            .clicked()
        {
            delta = 1;
        }
        ui.label(RichText::new(trailing).color(theme::TEXT_DIM).monospace());
    });
    delta
}
