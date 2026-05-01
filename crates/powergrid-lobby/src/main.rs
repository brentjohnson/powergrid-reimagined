mod driver;
mod lobby_handler;
mod room_handler;
mod rooms;
mod ws;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use powergrid_core::{actions::RoomSummary, map::Map};
use rooms::RoomManager;
use std::{sync::Arc, time::Duration};
use tracing::info;

#[derive(Clone)]
struct AppState {
    manager: Arc<RoomManager>,
    bot_delay: Duration,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("powergrid_lobby=debug,info")
            }),
        )
        .init();

    const DEFAULT_MAP: &str = include_str!("../../powergrid-server/maps/germany.toml");

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let bot_delay_ms: u64 = std::env::var("BOT_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);

    let map_str = if let Ok(path) = std::env::var("MAP_FILE") {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read map file {path}: {e}"))
    } else {
        DEFAULT_MAP.to_string()
    };
    let map = Map::load(&map_str).unwrap_or_else(|e| panic!("Failed to parse map: {e}"));
    info!("Loaded map: {}", map.name);

    let state = AppState {
        manager: Arc::new(RoomManager::new(map)),
        bot_delay: Duration::from_millis(bot_delay_ms),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/rooms", get(list_rooms))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Lobby server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn list_rooms(State(state): State<AppState>) -> Json<Vec<RoomSummary>> {
    Json(state.manager.list().await)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws::handle_socket(socket, state.manager, state.bot_delay))
}
