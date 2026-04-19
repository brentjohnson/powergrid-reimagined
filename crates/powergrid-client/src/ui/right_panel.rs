use bevy::prelude::Res;
use bevy_egui::egui;
use egui::{RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerId},
    GameState,
};

use crate::{card_painter, state::AppState, theme, ws::WsChannels};

use super::action_panel::action_panel;
use super::helpers::{section_header, send};

pub(super) fn right_panel_contents(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    // ---- Power plant market ----
    section_header(ui, "PLANT MARKET");
    theme::neon_frame().show(ui, |ui| {
        ui.label(
            RichText::new("ACTUAL")
                .color(theme::TEXT_DIM)
                .small()
                .monospace(),
        );
        plant_row(
            ui,
            &gs.market.actual,
            channels,
            &gs.phase,
            my_id,
            &gs.player_order,
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new("FUTURE")
                .color(theme::TEXT_DIM)
                .small()
                .monospace(),
        );
        plant_row(
            ui,
            &gs.market.future,
            channels,
            &gs.phase,
            my_id,
            &gs.player_order,
        );
    });

    ui.add_space(4.0);

    // ---- My action panel ----
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

        ui.add_space(4.0);
    }

    // ---- Event log ----
    section_header(ui, "EVENT LOG");
    theme::neon_frame().show(ui, |ui| {
        for entry in gs.event_log.iter().rev().take(8) {
            ui.label(
                RichText::new(entry)
                    .color(theme::TEXT_DIM)
                    .small()
                    .monospace(),
            );
        }
    });

    ui.add_space(8.0);
}

fn plant_row(
    ui: &mut Ui,
    plants: &[powergrid_core::types::PowerPlant],
    channels: &Option<Res<WsChannels>>,
    phase: &Phase,
    my_id: PlayerId,
    player_order: &[PlayerId],
) {
    let is_my_auction_turn = matches!(phase, Phase::Auction { current_bidder_idx, active_bid, .. }
        if active_bid.is_none() && player_order.get(*current_bidder_idx) == Some(&my_id));

    ui.horizontal_wrapped(|ui| {
        for plant in plants {
            let resp = card_painter::draw_plant_card(ui, plant, 70.0);
            if is_my_auction_turn && resp.clicked() {
                send(
                    Action::SelectPlant {
                        plant_number: plant.number,
                    },
                    channels,
                );
            }
            if resp.hovered() {
                egui::show_tooltip_at_pointer(
                    ui.ctx(),
                    ui.layer_id(),
                    egui::Id::new(plant.number),
                    |ui| {
                        plant_tooltip(ui, plant);
                    },
                );
            }
        }
    });
}

fn plant_tooltip(ui: &mut Ui, plant: &powergrid_core::types::PowerPlant) {
    ui.label(
        RichText::new(format!(
            "#{} {:?}\nCost: {}  Cities: {}",
            plant.number, plant.kind, plant.cost, plant.cities
        ))
        .monospace()
        .color(theme::TEXT_BRIGHT),
    );
}
