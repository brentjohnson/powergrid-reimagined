mod ws;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use powergrid_core::{map::Map, GameState};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

pub type SharedState = Arc<Mutex<ServerState>>;

pub struct ServerState {
    pub game: GameState,
    /// Senders for all connected clients: (player_id, tx).
    pub clients: Vec<(uuid::Uuid, tokio::sync::mpsc::UnboundedSender<String>)>,
}

impl ServerState {
    pub fn new(map: Map) -> Self {
        Self {
            game: GameState::new(map, 6),
            clients: Vec::new(),
        }
    }
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: powergrid-server

Environment variables:
  PORT       Port to listen on (default: 3000)
  MAP_FILE   Path to a custom map TOML file (default: embedded Germany map)
  RUST_LOG   Log filter, e.g. debug or info (default: info)

Options:
  -h, --help   Show this help message"
        );
        std::process::exit(0);
    }

    tracing_subscriber::fmt::init();

    const DEFAULT_MAP: &str = include_str!("../maps/germany.toml");

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    let map_str = if let Ok(path) = std::env::var("MAP_FILE") {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read map file {path}: {e}"))
    } else {
        DEFAULT_MAP.to_string()
    };
    let map = Map::load(&map_str).unwrap_or_else(|e| panic!("Failed to parse map: {e}"));

    info!("Loaded map: {}", map.name);

    let state: SharedState = Arc::new(Mutex::new(ServerState::new(map)));

    let app = Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws::handle_socket(socket, state))
}
