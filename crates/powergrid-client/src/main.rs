mod audio;
mod auth;
mod card_painter;
mod effects;
mod local;
mod map_panel;
mod peer_hints;
mod state;
mod theme;
mod ui;
mod ws;

use local::LocalHandle;
use state::{AppState, CliArgs};
use std::time::Duration;
use ui::UiAction;
use ws::WsChannels;

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = CliArgs::parse();
    let app_state = AppState::new(cli);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Power Grid: Reimagined")
            .with_inner_size([1600.0, 900.0])
            .with_fullscreen(app_state.fullscreen),
        ..Default::default()
    };

    eframe::run_native(
        "Power Grid: Reimagined",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(PowerGridApp::new(app_state)))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct PowerGridApp {
    state: AppState,
    ws: Option<WsChannels>,
    local: Option<LocalHandle>,
    audio: Option<audio::Audio>,
}

impl PowerGridApp {
    fn new(state: AppState) -> Self {
        Self {
            state,
            ws: None,
            local: None,
            audio: audio::Audio::new(),
        }
    }
}

impl eframe::App for PowerGridApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Per-frame polling: drain background thread results before drawing.
        state::process_auth_events(&mut self.state);
        ws::process_ws_events(&mut self.state, self.ws.as_ref());

        // Tear down the WS worker after a logout or auth rejection so it stops
        // reconnecting (and stops clobbering the error we just showed). A fresh
        // connection is only spawned via the pending_connect path below.
        if std::mem::take(&mut self.state.disconnect_requested) {
            self.ws = None;
            self.local = None;
        }

        // Play the turn-notification sound once per transition into "it's my turn",
        // but only in online (lobby) games — not local-vs-bots play.
        if std::mem::take(&mut self.state.play_turn_sound) && self.local.is_none() {
            if let Some(audio) = &self.audio {
                audio.play_turn_sound();
            }
        }

        // Auto-connect for online play: trigger once when pending and not yet connected.
        if self.state.pending_connect && self.ws.is_none() {
            let url = self.state.ws_url();
            self.ws = Some(ws::spawn_ws(url));
        }

        // Auto-refresh room list (every 10s when in RoomBrowser).
        if self.state.screen == state::Screen::RoomBrowser {
            let now = ctx.input(|i| i.time);
            if now - self.state.room_list_last_refresh >= 10.0 {
                if let Some(ch) = &self.ws {
                    ch.send_lobby(powergrid_core::actions::LobbyAction::ListRooms);
                }
                self.state.room_list_last_refresh = now;
            }
        }

        // Keep the app responsive while a session or auth request is in flight.
        if self.ws.is_some() || self.state.auth_in_flight {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        // Emit peer hints when local selection changes (debounced 150 ms).
        if let (Some(ws), Some(room)) = (self.ws.as_ref(), self.state.current_room.clone()) {
            let now = ctx.input(|i| i.time);
            let cart = self.state.resource_cart.clone();
            let cities = self.state.selected_build_cities.clone();
            let edges = self.state.build_preview.edges.clone();
            if let Some(hint) = self.state.hint_tracker.update(&cart, &cities, &edges, now) {
                ws.send_hint(room, hint);
            }
        }

        // Draw UI and collect deferred actions.
        let action = ui::ui_system(ctx, &mut self.state, self.ws.as_ref());

        // Apply side-effects after the egui pass.
        match action {
            UiAction::None => {}
            UiAction::StartLocal(cfg) => {
                let (channels, handle) = local::start_local_session(cfg);
                self.ws = Some(channels);
                self.local = Some(handle);
            }
            UiAction::ExitToMenu => {
                self.local = None;
                self.ws = None;
            }
            UiAction::ToggleFullscreen => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.state.fullscreen));
                if !self.state.no_preferences {
                    if let Err(e) = auth::save_preferences(&auth::UserPreferences {
                        fullscreen: self.state.fullscreen,
                    }) {
                        tracing::warn!("Failed to save preferences: {e}");
                    }
                }
            }
            UiAction::Exit => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }
}
