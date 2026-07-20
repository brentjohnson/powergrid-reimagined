//! Ingest endpoint for locally-hosted (in-process) games.
//!
//! The desktop client can run a game entirely in-process (no room on the
//! server). To still record those games in the metrics DB, the client POSTs a
//! `GameReport` here when the game ends. Attribution is decided server-side:
//! the human seat is credited to the account behind the bearer token, or to
//! `anonymous` (NULL user) if there's no token or it fails to validate — the
//! client never gets to name which user a result belongs to.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use powergrid_session::{GameReport, MAX_PLAYERS};

use crate::{db::GameRecord, AppState};

/// Name recorded for the human seat when a local game is submitted without a
/// valid login.
const ANONYMOUS_NAME: &str = "anonymous";

pub async fn submit_local_game(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(report): Json<GameReport>,
) -> StatusCode {
    // Guard against malformed / abusive payloads before touching the DB.
    if report.seats.is_empty() || report.seats.len() > MAX_PLAYERS as usize {
        return StatusCode::BAD_REQUEST;
    }

    // Resolve the submitter from the bearer token. Any failure (missing,
    // malformed, expired, or unknown token) falls through to anonymous.
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_owned);
    let resolved = match token {
        Some(t) => app.db.validate_token(&t).await.ok(),
        None => None,
    };

    // Re-attribute seats: bots never carry a user; the (single) human seat is
    // credited to the resolved account, or anonymized. The client-supplied
    // user_id is never trusted — it's always overwritten here.
    let mut seats = report.seats;
    for seat in &mut seats {
        if seat.is_bot {
            seat.user_id = None;
        } else if let Some((uid, _)) = &resolved {
            seat.user_id = Some(*uid);
        } else {
            seat.user_id = None;
            seat.player_name = ANONYMOUS_NAME.to_string();
        }
    }

    let record = GameRecord {
        room_name: "local".to_string(),
        map_name: report.map_name,
        // The client doesn't track a reliable start time for local play; the
        // games row's finished_at (now) is the durable timestamp.
        started_at: None,
        rounds: report.rounds,
        seats,
    };

    match app.db.record_game(&record).await {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(e) => {
            tracing::error!("Failed to record local game: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
