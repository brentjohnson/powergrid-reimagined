use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui::{Color32, RichText, Ui};
use powergrid_core::{
    actions::Action,
    types::{Phase, PlayerColor, PlayerId, Resource},
    GameState,
};

use crate::{
    card_painter,
    state::{player_color_to_egui, AppState, Screen},
    theme,
    ws::WsChannels,
};

// ---------------------------------------------------------------------------
// One-time theme setup
// ---------------------------------------------------------------------------

pub fn setup_egui_theme(mut contexts: EguiContexts) {
    theme::apply(contexts.ctx_mut());
}

// ---------------------------------------------------------------------------
// Main UI system (runs every frame)
// ---------------------------------------------------------------------------

pub fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<AppState>,
    channels: Option<Res<WsChannels>>,
    mut commands: Commands,
) {
    let ctx = contexts.ctx_mut();

    // Re-apply theme every frame so settings survive window resize etc.
    // (cheap — just copies a struct)
    theme::apply(ctx);

    match state.screen {
        Screen::Connect => {
            connect_screen(ctx, &mut state, &mut commands);
        }
        Screen::Game => {
            game_screen(ctx, &mut state, &channels);
        }
    }
}

// ---------------------------------------------------------------------------
// Connect screen
// ---------------------------------------------------------------------------

fn connect_screen(ctx: &egui::Context, state: &mut AppState, commands: &mut Commands) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                // Title
                ui.label(
                    RichText::new("POWER GRID")
                        .size(42.0)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("REIMAGINED")
                        .size(20.0)
                        .color(theme::NEON_CYAN_DIM)
                        .monospace(),
                );

                ui.add_space(40.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(360.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    // Server URL
                    ui.label(RichText::new("SERVER URL").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.connect_url);

                    // Player name
                    ui.label(RichText::new("CALLSIGN").color(theme::TEXT_DIM).small());
                    ui.text_edit_singleline(&mut state.player_name);

                    // Color selector
                    ui.label(
                        RichText::new("FACTION COLOR")
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.horizontal_wrapped(|ui| {
                        for color in [
                            PlayerColor::Red,
                            PlayerColor::Blue,
                            PlayerColor::Green,
                            PlayerColor::Yellow,
                            PlayerColor::Purple,
                            PlayerColor::White,
                        ] {
                            let egui_color = player_color_to_egui(color);
                            let selected = state.selected_color == color;
                            let label = color_label(color);

                            let btn = egui::Button::new(RichText::new(label).color(if selected {
                                Color32::BLACK
                            } else {
                                egui_color
                            }))
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
                                state.selected_color = color;
                            }
                        }
                    });

                    ui.add_space(8.0);

                    let can_connect = !state.player_name.trim().is_empty();
                    let connect_btn = egui::Button::new(
                        RichText::new("[ CONNECT ]")
                            .color(if can_connect {
                                theme::BG_DEEP
                            } else {
                                theme::TEXT_DIM
                            })
                            .monospace(),
                    )
                    .fill(if can_connect {
                        theme::NEON_CYAN
                    } else {
                        theme::BG_WIDGET
                    })
                    .stroke(egui::Stroke::new(
                        1.5,
                        if can_connect {
                            theme::NEON_CYAN
                        } else {
                            theme::NEON_CYAN_DARK
                        },
                    ));

                    if ui.add_enabled(can_connect, connect_btn).clicked() {
                        let url = state.connect_url.clone();
                        let name = state.player_name.trim().to_string();
                        let color = state.selected_color;
                        state.pending_join = Some((name, color));
                        let channels = crate::ws::spawn_ws(url);
                        commands.insert_resource(channels);
                    }
                });

                if !state.connected
                    && state.pending_join.is_none()
                    && state.game_state.is_none()
                    && state.screen == Screen::Connect
                {
                    // No error to show yet
                } else if !state.connected && state.pending_join.is_some() {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("● CONNECTING…")
                            .color(theme::NEON_AMBER)
                            .monospace(),
                    );
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Game screen
// ---------------------------------------------------------------------------

fn game_screen(ctx: &egui::Context, state: &mut AppState, channels: &Option<Res<WsChannels>>) {
    let Some(gs) = state.game_state.clone() else {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("● AWAITING UPLINK…")
                        .color(theme::NEON_AMBER)
                        .heading(),
                );
            });
        });
        return;
    };

    let my_id = state.my_id.unwrap_or_default();

    if matches!(gs.phase, Phase::Lobby) {
        lobby_screen(ctx, state, channels, &gs, my_id);
        return;
    }

    // Top panel — phase info and resource market
    egui::TopBottomPanel::top("top_panel")
        .exact_height(110.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(6.0)),
        )
        .show(ctx, |ui| {
            top_panel_contents(ui, gs.clone());
        });

    // Left panel — player info
    egui::SidePanel::left("player_panel")
        .resizable(false)
        .exact_width(220.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                left_panel_contents(ui, &gs, my_id);
            });
        });

    // Right panel — plant market, actions, event log
    egui::SidePanel::right("info_panel")
        .resizable(false)
        .exact_width(340.0)
        .frame(
            egui::Frame::none()
                .fill(theme::BG_DEEP)
                .stroke(egui::Stroke::new(1.0, theme::NEON_CYAN_DARK))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                right_panel_contents(ui, state, channels, &gs, my_id);
            });
        });

    // Central map
    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(Color32::from_rgb(2, 4, 8))
                .inner_margin(egui::Margin::same(0.0)),
        )
        .show(ctx, |ui| {
            crate::map_panel::draw(ui, state, &gs, my_id);
        });
}

// ---------------------------------------------------------------------------
// Lobby screen
// ---------------------------------------------------------------------------

fn lobby_screen(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(theme::BG_DEEP))
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

// ---------------------------------------------------------------------------
// Top panel — phase/round header, phase tracker, resource market
// ---------------------------------------------------------------------------

fn top_panel_contents(ui: &mut Ui, gs: GameState) {
    ui.horizontal(|ui| {
        // Phase / round header
        ui.vertical(|ui| {
            theme::neon_frame_bright().show(ui, |ui| {
                ui.label(
                    RichText::new(format!("ROUND {}", gs.round))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new(phase_name(&gs.phase))
                        .color(theme::NEON_AMBER)
                        .monospace(),
                );
            });
        });

        ui.add_space(8.0);

        // Phase tracker
        phase_tracker(ui, &gs);

        ui.add_space(8.0);

        // Resource market
        ui.vertical(|ui| {
            section_header(ui, "RESOURCE MARKET");
            theme::neon_frame().show(ui, |ui| {
                let r = &gs.resources;
                ui.horizontal(|ui| {
                    resource_badge(ui, "COAL", r.coal, Color32::from_rgb(107, 68, 35));
                    resource_badge(ui, "OIL", r.oil, Color32::from_rgb(60, 60, 60));
                    resource_badge(ui, "GARB", r.garbage, Color32::from_rgb(200, 170, 20));
                    resource_badge(ui, "URAN", r.uranium, Color32::from_rgb(200, 30, 30));
                });
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Left panel — player info
// ---------------------------------------------------------------------------

fn left_panel_contents(ui: &mut Ui, gs: &GameState, my_id: PlayerId) {
    for pid in &gs.player_order {
        if let Some(p) = gs.player(*pid) {
            let is_me = p.id == my_id;
            let active = is_active_player(gs, p.id);
            let border_color = if active {
                player_color_to_egui(p.color)
            } else {
                dim_color(player_color_to_egui(p.color))
            };

            egui::Frame::none()
                .fill(theme::BG_PANEL)
                .stroke(egui::Stroke::new(
                    if active { 2.0 } else { 1.0 },
                    border_color,
                ))
                .inner_margin(egui::Margin::same(6.0))
                .rounding(egui::Rounding::same(3.0))
                .show(ui, |ui| {
                    // Header row
                    ui.horizontal(|ui| {
                        let name_color = player_color_to_egui(p.color);
                        ui.colored_label(name_color, RichText::new(&p.name).monospace().strong());
                        if is_me {
                            ui.label(RichText::new("(you)").color(theme::TEXT_DIM).small());
                        }
                        if active {
                            ui.label(
                                RichText::new("◀ ACTIVE")
                                    .color(theme::NEON_AMBER)
                                    .small()
                                    .monospace(),
                            );
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!("${}", p.money))
                                    .color(theme::NEON_GREEN)
                                    .monospace(),
                            );
                        });
                    });

                    // Resources + cities row
                    ui.horizontal(|ui| {
                        let res = &p.resources;
                        let mut parts = Vec::new();
                        if res.coal > 0 {
                            parts.push(format!("C:{}", res.coal));
                        }
                        if res.oil > 0 {
                            parts.push(format!("O:{}", res.oil));
                        }
                        if res.garbage > 0 {
                            parts.push(format!("G:{}", res.garbage));
                        }
                        if res.uranium > 0 {
                            parts.push(format!("U:{}", res.uranium));
                        }
                        let res_str = if parts.is_empty() {
                            "No resources".to_string()
                        } else {
                            parts.join("  ")
                        };
                        ui.label(
                            RichText::new(res_str)
                                .color(theme::TEXT_MID)
                                .small()
                                .monospace(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!("{} cities", p.cities.len()))
                                    .color(theme::TEXT_MID)
                                    .small()
                                    .monospace(),
                            );
                        });
                    });

                    // Plants row
                    if !p.plants.is_empty() {
                        ui.horizontal(|ui| {
                            for plant in &p.plants {
                                card_painter::draw_plant_card(ui, plant, 44.0);
                            }
                        });
                    }
                });
            ui.add_space(4.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Right panel — plant market, action console, your plants, event log
// ---------------------------------------------------------------------------

fn right_panel_contents(
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
    if let Some(me) = gs.player(my_id) {
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
        section_header(ui, "YOUR PLANTS");
        theme::neon_frame().show(ui, |ui| {
            if me.plants.is_empty() {
                ui.label(RichText::new("No plants").color(theme::TEXT_DIM).small());
            } else {
                ui.horizontal_wrapped(|ui| {
                    for plant in &me.plants {
                        card_painter::draw_plant_card(ui, plant, 70.0);
                    }
                });
            }
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

// ---------------------------------------------------------------------------
// Phase tracker
// ---------------------------------------------------------------------------

fn phase_tracker(ui: &mut Ui, gs: &GameState) {
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

// ---------------------------------------------------------------------------
// Action panel (phase-specific controls)
// ---------------------------------------------------------------------------

fn action_panel(
    ui: &mut Ui,
    state: &mut AppState,
    channels: &Option<Res<WsChannels>>,
    gs: &GameState,
    my_id: PlayerId,
) {
    match &gs.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            ..
        } => {
            let my_nominate_turn = gs.player_order.get(*current_bidder_idx) == Some(&my_id);

            if let Some(bid) = active_bid {
                let is_my_bid_turn = bid.remaining_bidders.first() == Some(&my_id);
                if is_my_bid_turn {
                    ui.label(
                        RichText::new(format!(
                            "Bid on plant #{} — current: ${}",
                            bid.plant_number, bid.amount
                        ))
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                    );
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut state.bid_amount)
                                .desired_width(80.0)
                                .hint_text("amount"),
                        );
                        let bid_valid = state.bid_amount.trim().parse::<u32>().is_ok();
                        if ui
                            .add_enabled(bid_valid, neon_button("[ BID ]", theme::NEON_CYAN))
                            .clicked()
                        {
                            if let Ok(amount) = state.bid_amount.trim().parse::<u32>() {
                                send(Action::PlaceBid { amount }, channels);
                                state.bid_amount.clear();
                            }
                        }
                        if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
                            send(Action::PassAuction, channels);
                        }
                    });
                } else {
                    ui.label(
                        RichText::new(format!("● Bidding on #{} — waiting…", bid.plant_number))
                            .color(theme::TEXT_DIM)
                            .monospace(),
                    );
                }
            } else if my_nominate_turn {
                ui.label(
                    RichText::new("Your turn — select a plant from the market, or pass.")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );
                if ui.add(neon_button("[ PASS ]", theme::NEON_AMBER)).clicked() {
                    send(Action::PassAuction, channels);
                }
            } else {
                ui.label(
                    RichText::new("● Waiting for other operators…")
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            }
        }

        Phase::BuyResources { remaining } => {
            if remaining.first() == Some(&my_id) {
                let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);

                for resource in [
                    Resource::Coal,
                    Resource::Oil,
                    Resource::Garbage,
                    Resource::Uranium,
                ] {
                    let count = state.resource_cart.get(&resource).copied().unwrap_or(0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{:>8}: {:>2}", resource_name(resource), count))
                                .color(theme::TEXT_BRIGHT)
                                .monospace(),
                        );
                        if ui.add(neon_button("[+]", theme::NEON_GREEN)).clicked() {
                            state.add_to_cart(resource);
                        }
                        if ui.add(neon_button("[-]", theme::NEON_AMBER)).clicked() {
                            state.remove_from_cart(resource);
                        }
                    });
                }

                if let Some(cost) = state.resource_cart_cost {
                    let cost_color = if cost > my_money {
                        theme::NEON_RED
                    } else {
                        theme::NEON_GREEN
                    };
                    ui.label(
                        RichText::new(format!("TOTAL: ${cost}  BALANCE: ${my_money}"))
                            .color(cost_color)
                            .monospace(),
                    );
                }

                let unaffordable = state.resource_cart_cost.is_some_and(|c| c > my_money);
                ui.horizontal(|ui| {
                    if ui
                        .add(neon_button("[ CLEAR ]", theme::NEON_AMBER))
                        .clicked()
                    {
                        state.clear_cart();
                    }
                    if ui
                        .add_enabled(
                            !unaffordable,
                            neon_button("[ DONE BUYING ]", theme::NEON_CYAN),
                        )
                        .clicked()
                    {
                        let purchases = state.cart_purchases();
                        if purchases.is_empty() {
                            send(Action::DoneBuying, channels);
                        } else {
                            send(Action::BuyResourceBatch { purchases }, channels);
                        }
                    }
                });
            } else {
                ui.label(
                    RichText::new("● Waiting for other operators to buy…")
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            }
        }

        Phase::BuildCities { remaining } => {
            if remaining.first() == Some(&my_id) {
                let my_money = gs.player(my_id).map(|p| p.money).unwrap_or(0);
                ui.label(
                    RichText::new("Click cities on map to select build targets.")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );

                if !state.selected_build_cities.is_empty() {
                    let bp = &state.build_preview;
                    let cost_color = if bp.total_cost > my_money {
                        theme::NEON_RED
                    } else {
                        theme::NEON_GREEN
                    };
                    ui.label(
                        RichText::new(format!(
                            "Selected: {}  Route: ${}  Slots: ${}  Total: ${}",
                            state.selected_build_cities.len(),
                            bp.total_route_cost,
                            bp.total_slot_cost,
                            bp.total_cost,
                        ))
                        .color(cost_color)
                        .monospace(),
                    );
                }

                ui.horizontal(|ui| {
                    if ui
                        .add(neon_button("[ CLEAR ]", theme::NEON_AMBER))
                        .clicked()
                    {
                        state.clear_build_selection();
                    }
                    if ui
                        .add(neon_button("[ DONE BUILDING ]", theme::NEON_CYAN))
                        .clicked()
                    {
                        if state.selected_build_cities.is_empty() {
                            send(Action::DoneBuilding, channels);
                        } else {
                            let city_ids = state.build_preview.ordered.clone();
                            send(Action::BuildCities { city_ids }, channels);
                        }
                    }
                });
            } else {
                ui.label(
                    RichText::new("● Waiting for other operators to build…")
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            }
        }

        Phase::Bureaucracy { remaining } => {
            if remaining.first() == Some(&my_id) {
                ui.label(
                    RichText::new("Fire all plants you can to power cities.")
                        .color(theme::TEXT_BRIGHT)
                        .monospace(),
                );
                if ui
                    .add(neon_button("[ POWER CITIES ]", theme::NEON_GREEN))
                    .clicked()
                {
                    if let Some(player) = gs.player(my_id) {
                        let plant_numbers: Vec<u8> =
                            player.plants.iter().map(|p| p.number).collect();
                        send(Action::PowerCities { plant_numbers }, channels);
                    }
                }
            } else {
                ui.label(
                    RichText::new("● Waiting for other operators…")
                        .color(theme::TEXT_DIM)
                        .monospace(),
                );
            }
        }

        Phase::GameOver { winner } => {
            let name = gs
                .player(*winner)
                .map(|p| p.name.as_str())
                .unwrap_or("UNKNOWN");
            ui.label(
                RichText::new(format!("GRID CONTROLLED BY: {name}"))
                    .size(20.0)
                    .color(theme::NEON_GREEN)
                    .monospace(),
            );
        }

        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Plant row (market)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn section_header(ui: &mut Ui, label: &str) {
    ui.label(
        RichText::new(label)
            .color(theme::NEON_CYAN_DIM)
            .small()
            .monospace(),
    );
}

fn neon_button(label: &str, color: Color32) -> egui::Button<'static> {
    egui::Button::new(RichText::new(label).color(color).monospace())
        .fill(theme::BG_WIDGET)
        .stroke(egui::Stroke::new(1.0, color))
}

fn resource_badge(ui: &mut Ui, label: &str, count: u8, color: Color32) {
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

fn dim_color(c: Color32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (c.r() as f32 * 0.3) as u8,
        (c.g() as f32 * 0.3) as u8,
        (c.b() as f32 * 0.3) as u8,
        180,
    )
}

fn phase_name(phase: &Phase) -> &'static str {
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

fn is_active_player(gs: &GameState, pid: PlayerId) -> bool {
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

fn resource_name(r: Resource) -> &'static str {
    match r {
        Resource::Coal => "COAL",
        Resource::Oil => "OIL",
        Resource::Garbage => "GARBAGE",
        Resource::Uranium => "URANIUM",
    }
}

fn color_label(c: PlayerColor) -> &'static str {
    match c {
        PlayerColor::Red => "RED",
        PlayerColor::Blue => "BLUE",
        PlayerColor::Green => "GREEN",
        PlayerColor::Yellow => "YELLOW",
        PlayerColor::Purple => "PURPLE",
        PlayerColor::White => "WHITE",
    }
}

fn send(action: Action, channels: &Option<Res<WsChannels>>) {
    if let Some(ch) = channels {
        ch.action_tx.send(action).ok();
    }
}
