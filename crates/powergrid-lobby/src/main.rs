mod auth;
mod db;
mod driver;
mod lobby_handler;
mod room_handler;
mod rooms;
mod ws;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use db::Db;
use powergrid_core::{actions::RoomSummary, map::Map};
use rooms::RoomManager;
use std::{sync::Arc, time::Duration};
use tower_http::cors::CorsLayer;
use tracing::info;

#[derive(Clone)]
pub struct AppState {
    pub manager: Arc<RoomManager>,
    pub bot_delay: Duration,
    pub db: Db,
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: powergrid-lobby

Environment variables:
  PORT          Port to listen on (default: 3000)
  DATABASE_URL  PostgreSQL connection URL (required)
  MAP_FILE      Path to a custom map TOML file (default: embedded Germany map)
  BOT_DELAY_MS  Delay between bot moves in milliseconds (default: 250)
  RUST_LOG      Log filter, e.g. debug or info (default: info)

Options:
  -h, --help   Show this help message"
        );
        std::process::exit(0);
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("powergrid_lobby=debug,info")
            }),
        )
        .init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let bot_delay_ms: u64 = std::env::var("BOT_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable must be set");

    let map = if let Ok(path) = std::env::var("MAP_FILE") {
        let s = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read map file {path}: {e}"));
        Map::load(&s).unwrap_or_else(|e| panic!("Failed to parse map: {e}"))
    } else {
        powergrid_core::default_map()
    };
    info!("Loaded map: {}", map.name);

    let db = Db::connect(&database_url)
        .await
        .unwrap_or_else(|e| panic!("Failed to connect to database: {e}"));
    db.migrate()
        .await
        .unwrap_or_else(|e| panic!("Failed to run migrations: {e}"));
    info!("Database ready");

    let db_for_shutdown = db.clone();
    let state = AppState {
        manager: Arc::new(RoomManager::new(map)),
        bot_delay: Duration::from_millis(bot_delay_ms),
        db,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/rooms", get(list_rooms))
        .route("/ws", get(ws_handler))
        .route("/auth/register", post(auth::register))
        .route("/auth/login", post(auth::login))
        .route("/auth/logout", post(auth::logout))
        .layer(CorsLayer::very_permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Lobby server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    info!("Closing database pool");
    db_for_shutdown.pool.close().await;
    info!("Lobby server stopped");
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn list_rooms(State(state): State<AppState>) -> Json<Vec<RoomSummary>> {
    Json(state.manager.list().await)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws::handle_socket(socket, state.manager, state.bot_delay, state.db))
}
