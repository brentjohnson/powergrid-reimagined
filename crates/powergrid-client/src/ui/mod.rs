mod event_log;
mod helpers;
pub(crate) use helpers::is_active_player;
mod left_panel;
mod lobby;
mod local_setup;
mod login;
mod main_menu;
mod phases;
mod player_summary;
mod register;
mod room_browser;
mod top_panel;

use egui::{Color32, RichText};
use phases::{
    auction_panel, build_cities_panel, bureaucracy_panel, buy_resources_panel, discard_plant_panel,
    discard_resource_panel, power_cities_fuel_panel,
};
use powergrid_bot_strategy::{
    default_registry,
    features::{evaluate_plant, PlantValuation},
    BotProfile,
};
use powergrid_core::types::{Phase, PlayerColor, PlayerId};

use crate::{
    local::LocalConfig,
    state::{AppState, BottomTab, Screen},
    theme,
    ws::WsChannels,
};

/// Side-effects requested by the UI for the app to apply after the egui pass.
pub enum UiAction {
    None,
    StartLocal(LocalConfig),
    ExitToMenu,
    Exit,
    ToggleFullscreen,
}

// ---------------------------------------------------------------------------
// Main UI function (called from eframe App::update each frame)
// ---------------------------------------------------------------------------

pub fn ui_system(
    ctx: &egui::Context,
    state: &mut AppState,
    channels: Option<&WsChannels>,
) -> UiAction {
    theme::apply(ctx);

    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.menu_open = !state.menu_open;
    }

    if matches!(state.screen, Screen::Game)
        && !ctx.wants_keyboard_input()
        && ctx.input(|i| i.key_pressed(egui::Key::Space))
    {
        state.bottom_panel_open = !state.bottom_panel_open;
    }

    // Bot valuation popup — local play only ("local" is the fixed room name
    // start_local_session uses; see local.rs).
    if matches!(state.screen, Screen::Game)
        && state.current_room.as_deref() == Some("local")
        && !ctx.wants_keyboard_input()
        && ctx.input(|i| i.key_pressed(egui::Key::B))
    {
        state.valuation_open = !state.valuation_open;
    }

    let mut action = UiAction::None;

    match state.screen {
        Screen::MainMenu => {
            main_menu::main_menu_screen(ctx, state, &mut action);
        }
        Screen::LocalSetup => {
            local_setup::local_setup_screen(ctx, state, &mut action);
        }
        Screen::Login => {
            login::login_screen(ctx, state);
        }
        Screen::Register => {
            register::register_screen(ctx, state);
        }
        Screen::RoomBrowser => {
            room_browser::room_browser_screen(ctx, state, channels);
        }
        Screen::Game => {
            game_screen(ctx, state, channels);
        }
    }

    if state.menu_open {
        egui::Window::new("MENU")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                if ui
                    .add(helpers::neon_button(
                        "[ BACK TO MAIN MENU ]",
                        theme::NEON_AMBER,
                    ))
                    .clicked()
                {
                    state.connected = false;
                    state.pending_connect = false;
                    state.my_id = None;
                    state.current_room = None;
                    state.game_state = None;
                    state.map = None;
                    state.error_message = None;
                    state.screen = Screen::MainMenu;
                    state.menu_open = false;
                    action = UiAction::ExitToMenu;
                }
                ui.add_space(4.0);
                let fs_label = if state.fullscreen {
                    "[ WINDOWED MODE ]"
                } else {
                    "[ FULLSCREEN ]"
                };
                if ui
                    .add(helpers::neon_button(fs_label, theme::NEON_CYAN))
                    .clicked()
                {
                    state.fullscreen = !state.fullscreen;
                    state.menu_open = false;
                    action = UiAction::ToggleFullscreen;
                }
                ui.add_space(4.0);
                if ui
                    .add(helpers::neon_button("[ EXIT ]", theme::NEON_RED))
                    .clicked()
                {
                    action = UiAction::Exit;
                }
                ui.add_space(4.0);
            });
    }

    action
}

// ---------------------------------------------------------------------------
// Game screen
// ---------------------------------------------------------------------------

fn game_screen(ctx: &egui::Context, state: &mut AppState, channels: Option<&WsChannels>) {
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
        lobby::lobby_screen(ctx, state, channels, &gs, my_id);
        return;
    }

    // GameOver overlay — rendered last so it floats above everything
    if let Phase::GameOver { .. } = gs.phase {
        phases::game_over_overlay(ctx, &gs);
    }

    let top_resp = egui::TopBottomPanel::top("top_panel")
        .min_height(180.0)
        .frame(theme::panel_frame(6))
        .show(ctx, |ui| {
            top_panel::top_panel_contents(ui, gs.clone(), state, channels, my_id);
        });
    state.top_panel_bottom = top_resp.response.rect.bottom();

    // Left panel is added before CentralPanel so it extends the full remaining height.
    state.left_panel_width = compute_left_panel_width(ctx, &gs);
    egui::SidePanel::left("player_panel")
        .resizable(false)
        .exact_width(state.left_panel_width)
        .frame(theme::panel_frame(0))
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                left_panel::left_panel_contents(ui, &gs, my_id);
            });
        });

    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_MAP)
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            crate::map_panel::draw(ui, state, &gs, my_id);
        });

    floating_action_panel(ctx, state, channels, &gs, my_id);

    // ── Bot valuation popup ("b" toggles, local play only) ────────────────────
    if state.current_room.as_deref() == Some("local") {
        valuation_window(ctx, state, &gs, my_id);
    }

    // ── Resource market (fixed size, pinned bottom-right) ─────────────────────
    // Renders before the cart below so `resource_market_width` is fresh this
    // frame — `buy_cart_panel` anchors against the market's measured rect.
    top_panel::resource_market_overlay(ctx, state, &gs, my_id);

    // ── Buy-resources cart (anchored directly left of the market) ─────────────
    buy_cart_panel(ctx, state, channels, &gs, my_id);

    // ── Info panel toggle + panel (Space or button, drops down from the
    // ── top panel) ─────────────────────────────────────────────────────────
    if state.bottom_panel_open {
        bottom_info_panel(ctx, state, &gs);
    } else {
        info_panel_toggle(ctx, state);
    }
}

/// `[ ▼ INFO ]` toggle for `bottom_info_panel`, drawn directly under the top
/// panel on the right (mirrors the placement math `floating_action_panel` uses
/// for `state.top_panel_bottom`/`FLOAT_GAP`). Opens the panel, which then
/// drops down from this same corner.
fn info_panel_toggle(ctx: &egui::Context, state: &mut AppState) {
    #[allow(deprecated)]
    let x = ctx.screen_rect().right() - CORNER_MARGIN;
    let y = state.top_panel_bottom + FLOAT_GAP;

    egui::Area::new(egui::Id::new("info_panel_toggle"))
        .fixed_pos(egui::pos2(x, y))
        .pivot(egui::Align2::RIGHT_TOP)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            if ui
                .add(helpers::neon_button("[ ▼ INFO ]", theme::NEON_CYAN))
                .clicked()
            {
                state.bottom_panel_open = true;
            }
        });
}

// ---------------------------------------------------------------------------
// Bot valuation popup ("b" key, local play only)
// ---------------------------------------------------------------------------

/// Live Elektro valuation table — one row per market plant, one column per
/// bot — showing exactly what `evaluate_plant` (LOGIC.md's
/// `MaximumBid = PlantValue` model) thinks each plant is worth to each bot
/// right now. **Local play only**: the wire protocol never reveals which seats
/// are bots or what difficulty they run, so bot identity has to be derived
/// from the deterministic color→difficulty mapping `start_local_session`
/// builds (see local.rs: `all_colors` minus the human's color, in order,
/// zipped with `local_bots`) — a mapping only the local client can reproduce.
fn valuation_window(
    ctx: &egui::Context,
    state: &mut AppState,
    gs: &powergrid_core::GameStateView,
    my_id: PlayerId,
) {
    if !state.valuation_open {
        // Small affordance tab visible when the popup is closed.
        egui::Area::new(egui::Id::new("valuation_toggle_area"))
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(8.0, -8.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                if ui
                    .add(helpers::neon_button("[ b: VALUATIONS ]", theme::NEON_GREEN))
                    .clicked()
                {
                    state.valuation_open = true;
                }
            });
        return;
    }

    let Some(map) = state.map.clone() else {
        return;
    };

    // Reproduce start_local_session's color assignment: all six colors, minus
    // whichever one the human took, in order — index i is "Bot {i+1}", whose
    // configured difficulty lives at `local_bots[i]`.
    let human_color = gs.player(my_id).map(|p| p.color);
    const ALL_COLORS: [PlayerColor; 6] = [
        PlayerColor::Red,
        PlayerColor::Blue,
        PlayerColor::Green,
        PlayerColor::Yellow,
        PlayerColor::Purple,
        PlayerColor::White,
    ];
    let bot_colors: Vec<PlayerColor> = ALL_COLORS
        .iter()
        .copied()
        .filter(|&c| Some(c) != human_color)
        .collect();

    // Reconstruct the full GameState — evaluate_plant needs map/graph access
    // that the wire-safe GameStateView doesn't carry directly.
    let gstate = gs.clone().into_game_state(&map);
    let registry = default_registry();

    let bots: Vec<(&powergrid_core::types::Player, &BotProfile)> = gstate
        .players
        .iter()
        .filter(|p| p.id != my_id)
        .filter_map(|p| {
            let idx = bot_colors.iter().position(|&c| c == p.color)?;
            let difficulty = *state.local_bots.get(idx)?;
            Some((p, registry.profile_for(difficulty)))
        })
        .collect();

    if bots.is_empty() {
        return;
    }

    const COL_W: f32 = 150.0;
    let panel_w = 110.0 + bots.len() as f32 * COL_W;

    egui::Window::new("BOT VALUATIONS")
        .resizable(false)
        .collapsible(false)
        .order(egui::Order::Foreground)
        .frame(theme::panel_frame(6))
        .default_width(panel_w)
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(8.0, -8.0))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("ELEKTRO VALUE  —  MaxBid = PlantValue (LOGIC.md)")
                        .color(theme::NEON_GREEN)
                        .monospace()
                        .small(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(helpers::neon_button("[ b: close ]", theme::NEON_GREEN))
                        .clicked()
                    {
                        state.valuation_open = false;
                    }
                });
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(360.0)
                .show(ui, |ui| {
                    // Header row: a blank plant-column gutter + one column per bot
                    // (color dot + bot name + difficulty).
                    ui.horizontal(|ui| {
                        ui.add_space(80.0);
                        for (player, profile) in &bots {
                            ui.scope(|ui| {
                                ui.set_width(COL_W);
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(8.0, 8.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().circle_filled(
                                        rect.center(),
                                        4.0,
                                        crate::state::player_color_to_egui(player.color),
                                    );
                                    ui.label(
                                        RichText::new(format!(
                                            "{} ({})",
                                            player.name, profile.display_name
                                        ))
                                        .color(theme::TEXT_BRIGHT)
                                        .monospace()
                                        .small(),
                                    );
                                });
                            });
                        }
                    });
                    ui.separator();

                    // One row per plant currently up for auction.
                    for plant in &gstate.market.actual {
                        ui.horizontal(|ui| {
                            ui.scope(|ui| {
                                ui.set_width(76.0);
                                ui.label(
                                    RichText::new(format!(
                                        "#{:<3}{:>2}c ${}",
                                        plant.number, plant.cities, plant.cost
                                    ))
                                    .color(theme::TEXT_MID)
                                    .monospace()
                                    .small(),
                                );
                            });

                            for (player, profile) in &bots {
                                let valuation =
                                    evaluate_plant(plant, player, &gstate, &profile.auction);
                                let color = if valuation.total >= profile.auction.min_open_score {
                                    theme::NEON_GREEN
                                } else {
                                    theme::TEXT_DIM
                                };
                                let resp = ui.add_sized(
                                    egui::vec2(COL_W, ui.spacing().interact_size.y),
                                    egui::Label::new(
                                        RichText::new(format!("{:>5.0}", valuation.total))
                                            .color(color)
                                            .monospace(),
                                    ),
                                );
                                resp.on_hover_ui(|ui| valuation_breakdown(ui, &valuation));
                            }
                        });
                    }
                });
        });
}

/// Component breakdown for a single plant valuation — shown on cell hover.
/// Mirrors the seven signed terms of `PlantValuation` plus the floored total
/// (LOGIC.md's `PlantValue ≈ IncomeGain + FuelSavings + EndgameBonus +
/// DenialBonus − OperatingCost − FuelRisk − ReplacementWaste`).
fn valuation_breakdown(ui: &mut egui::Ui, v: &PlantValuation) {
    ui.set_width(230.0);
    let row = |ui: &mut egui::Ui, label: &str, value: f32, color: Color32| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(label)
                    .color(theme::TEXT_DIM)
                    .monospace()
                    .small(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{value:>+7.1}"))
                        .color(color)
                        .monospace()
                        .small(),
                );
            });
        });
    };
    row(
        ui,
        "Incremental income",
        v.incremental_income,
        theme::NEON_CYAN,
    );
    row(ui, "Fuel savings", v.fuel_savings, theme::NEON_CYAN);
    row(ui, "Capacity premium", v.capacity_premium, theme::NEON_CYAN);
    row(ui, "Denial bonus", v.denial, theme::NEON_CYAN);
    row(ui, "Operating cost", -v.operating_cost, theme::NEON_RED);
    row(ui, "Fuel risk", -v.fuel_risk, theme::NEON_RED);
    row(
        ui,
        "Replacement waste",
        -v.replacement_waste,
        theme::NEON_RED,
    );
    ui.separator();
    row(ui, "TOTAL = Max Bid", v.total, theme::NEON_GREEN);
}

// ---------------------------------------------------------------------------
// Bottom-right tabbed info panel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Left-panel dynamic width
// ---------------------------------------------------------------------------

fn compute_left_panel_width(ctx: &egui::Context, gs: &powergrid_core::GameStateView) -> f32 {
    const MIN_W: f32 = 200.0;
    const MAX_W: f32 = 320.0;
    // Overhead: card inner margin (6*2=12) + frame stroke (~2) + scrollbar (~16) + extra padding (~14)
    const OVERHEAD: f32 = 44.0;

    let style = ctx.style();
    // Approximate monospace char width as 0.6 × font size (accurate for typical monospace fonts).
    let mono_char_w = style
        .text_styles
        .get(&egui::TextStyle::Monospace)
        .map(|f| f.size)
        .unwrap_or(12.0)
        * 0.6;
    let small_char_w = style
        .text_styles
        .get(&egui::TextStyle::Small)
        .map(|f| f.size)
        .unwrap_or(10.0)
        * 0.6;

    let measure = |text: &str, char_w: f32| -> f32 { text.chars().count() as f32 * char_w };

    let mut max_content = 0f32;

    for pid in &gs.player_order {
        if let Some(p) = gs.player(*pid) {
            let capacity: u32 = p.plants.iter().map(|pl| pl.cities as u32).sum();
            let header = format!("{} (you)", p.name);
            let capacity_line = format!(
                "capacity {} / cities {}",
                capacity,
                gs.player_city_count(p.id)
            );
            let has_plants = !p.plants.is_empty();
            let res_w = left_panel::resource_col_width(&p.resources);
            let has_res = res_w > 0.0;
            let row_w = match (has_plants, has_res) {
                (true, true) => crate::card_painter::CARD_W + left_panel::PLANT_RES_GAP + res_w,
                (true, false) => crate::card_painter::CARD_W,
                (false, true) => res_w,
                (false, false) => 0.0,
            };
            // House icon (ICON wide) + item_spacing.x (6) precedes the name in the header row.
            let icon_w = left_panel::ICON + 6.0;
            max_content = max_content
                .max(row_w)
                .max(icon_w + measure(&header, mono_char_w))
                .max(measure(&capacity_line, small_char_w));
        }
    }

    (max_content + OVERHEAD).clamp(MIN_W, MAX_W)
}

// ---------------------------------------------------------------------------
// Floating action panel — overlays the map beneath the active phase column
// ---------------------------------------------------------------------------

const FLOAT_GAP: f32 = 6.0;

fn floating_action_panel(
    ctx: &egui::Context,
    state: &mut crate::state::AppState,
    channels: Option<&crate::ws::WsChannels>,
    gs: &powergrid_core::GameStateView,
    my_id: PlayerId,
) {
    let (col_idx, show): (usize, bool) = match &gs.phase {
        Phase::Auction { .. } | Phase::DiscardPlant { .. } => (0, true),
        // BuyResources is handled by `buy_cart_panel`, anchored to the
        // bottom-right resource market overlay rather than this column.
        Phase::DiscardResource { .. } => (1, true),
        Phase::BuildCities { .. } => (2, true),
        Phase::Bureaucracy { .. } | Phase::PowerCitiesFuel { .. } => (3, true),
        _ => (0, false),
    };

    if !show {
        return;
    }

    let Some(col_rect) = state.phase_column_rects[col_idx] else {
        return; // first frame — rects not captured yet
    };

    #[allow(deprecated)]
    let screen_right = ctx.screen_rect().right() - 8.0;
    // Clamp x so the floating panel always has at least 280px of room to the right.
    let x = col_rect
        .min
        .x
        .max(state.left_panel_width + FLOAT_GAP)
        .min(screen_right - 280.0);
    let y = state.top_panel_bottom + FLOAT_GAP;
    let pos = egui::pos2(x, y);

    let max_width = (col_rect.width().max(280.0)).min(screen_right - x);

    egui::Area::new(egui::Id::new("floating_action_panel"))
        .fixed_pos(pos)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            theme::neon_frame().show(ui, |ui| {
                ui.set_max_width(max_width);
                match &gs.phase {
                    Phase::Auction { .. } => {
                        auction_panel(ui, state, channels, gs, my_id);
                    }
                    Phase::DiscardPlant { .. } => {
                        discard_plant_panel(ui, state, channels, gs, my_id);
                    }
                    Phase::DiscardResource { .. } => {
                        discard_resource_panel(ui, state, channels, gs, my_id);
                    }
                    Phase::BuildCities { .. } => {
                        build_cities_panel(ui, state, channels, gs, my_id);
                    }
                    Phase::Bureaucracy { .. } => {
                        bureaucracy_panel(ui, state, channels, gs, my_id);
                    }
                    Phase::PowerCitiesFuel { .. } => {
                        power_cities_fuel_panel(ui, state, channels, gs, my_id);
                    }
                    _ => {}
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Buy-resources cart (anchored directly left of the resource market overlay)
// ---------------------------------------------------------------------------

/// Cart UI for the Buy Resources phase — resource counts, TOTAL/BALANCE,
/// 1 SET / 2 SETS shortcuts, CLEAR / DONE BUYING. Anchored to sit directly to
/// the left of `top_panel::resource_market_overlay` (which you click to fill
/// the cart), bottom-aligned with it via `state.resource_market_width`.
fn buy_cart_panel(
    ctx: &egui::Context,
    state: &mut crate::state::AppState,
    channels: Option<&crate::ws::WsChannels>,
    gs: &powergrid_core::GameStateView,
    my_id: PlayerId,
) {
    if !matches!(&gs.phase, Phase::BuyResources { .. }) {
        return;
    }

    let x_offset = -(CORNER_MARGIN + state.resource_market_width + STACK_GAP);

    egui::Area::new(egui::Id::new("buy_cart_panel"))
        .anchor(
            egui::Align2::RIGHT_BOTTOM,
            egui::vec2(x_offset, -CORNER_MARGIN),
        )
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            theme::neon_frame().show(ui, |ui| {
                ui.set_max_width(240.0);
                buy_resources_panel(ui, state, channels, gs, my_id);
            });
        });
}

// ---------------------------------------------------------------------------
// Tabbed info panel — drops down from under the top panel (right side)
// ---------------------------------------------------------------------------

const PANEL_HEIGHT: f32 = 280.0;
// Corner margin shared with the resource market overlay's anchor offset
// (top_panel::resource_market_overlay) and the buy-cart panel below.
const CORNER_MARGIN: f32 = 8.0;
const STACK_GAP: f32 = 8.0;

fn bottom_info_panel(
    ctx: &egui::Context,
    state: &mut AppState,
    gs: &powergrid_core::GameStateView,
) {
    #[allow(deprecated)]
    let panel_w = (ctx.screen_rect().width() * 0.5).max(320.0);

    // Hangs directly under the top panel, right-aligned — the same corner the
    // `[ ▼ INFO ]` toggle (`info_panel_toggle`) occupies when this is closed.
    #[allow(deprecated)]
    let x = ctx.screen_rect().right() - CORNER_MARGIN;
    let y = state.top_panel_bottom + FLOAT_GAP;

    egui::Window::new("info_panel")
        .title_bar(false)
        .resizable(false)
        .movable(false)
        .collapsible(false)
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(egui::pos2(x, y))
        .fixed_size(egui::vec2(panel_w, PANEL_HEIGHT))
        .frame(theme::panel_frame(4))
        .show(ctx, |ui| {
            // Tab bar + collapse button
            ui.horizontal(|ui| {
                for tab in [
                    BottomTab::EventLog,
                    BottomTab::CityGraph,
                    BottomTab::Replenish,
                    BottomTab::Payout,
                ] {
                    let active = state.bottom_panel_tab == tab;
                    let color = if active {
                        theme::NEON_CYAN
                    } else {
                        theme::TEXT_DIM
                    };
                    let resp = ui.add(
                        egui::Button::new(
                            RichText::new(tab.label()).color(color).monospace().small(),
                        )
                        .fill(if active {
                            theme::BG_WIDGET
                        } else {
                            egui::Color32::TRANSPARENT
                        })
                        .stroke(egui::Stroke::new(
                            if active { 1.0 } else { 0.0 },
                            theme::NEON_CYAN,
                        )),
                    );
                    if resp.clicked() {
                        state.bottom_panel_tab = tab;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(helpers::neon_button("[ ▲ ]", theme::NEON_CYAN))
                        .clicked()
                    {
                        state.bottom_panel_open = false;
                    }
                });
            });

            ui.separator();

            let content_h = ui.available_height();

            // Tab content
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match state.bottom_panel_tab {
                    BottomTab::EventLog => {
                        event_log::event_log_contents(ui, gs);
                    }
                    BottomTab::CityGraph => {
                        if !state.city_history.is_empty() {
                            let players_info: Vec<(PlayerId, PlayerColor)> =
                                gs.players.iter().map(|p| (p.id, p.color)).collect();
                            theme::neon_frame().show(ui, |ui| {
                                top_panel::city_history_graph(
                                    ui,
                                    &state.city_history,
                                    &players_info,
                                    gs.end_game_cities,
                                    gs,
                                    content_h,
                                );
                            });
                        } else {
                            ui.label(
                                RichText::new("No city history yet.")
                                    .color(theme::TEXT_DIM)
                                    .monospace()
                                    .small(),
                            );
                        }
                    }
                    BottomTab::Replenish => {
                        theme::neon_frame().show(ui, |ui| {
                            top_panel::step_replenish_columns(ui, gs.step, gs.players.len());
                        });
                    }
                    BottomTab::Payout => {
                        theme::neon_frame().show(ui, |ui| {
                            top_panel::city_payout_table(ui, gs);
                        });
                    }
                });
        });
}
