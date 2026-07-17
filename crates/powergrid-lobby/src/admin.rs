//! Admin web interface: a token-gated JSON API under `/admin/api/*` plus a
//! self-contained static UI served at `/admin`. Mounted only when `ADMIN_TOKEN`
//! is set (see `main.rs`).

use std::sync::Arc;

use axum::{
    extract::{Path, Query, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::AppState;

#[derive(Serialize)]
struct ErrResp {
    error: String,
}

fn err(status: StatusCode, msg: impl ToString) -> (StatusCode, Json<ErrResp>) {
    (
        status,
        Json(ErrResp {
            error: msg.to_string(),
        }),
    )
}

/// Constant-time byte comparison for the admin token.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Build the admin router (merged into the main app under `/admin`). `token`
/// gates the `/admin/api/*` routes; the static UI is served without a token (the
/// page itself prompts for and stores it). Routes are spelled out with the full
/// `/admin` prefix rather than nested, so both `/admin` and `/admin/` resolve.
pub fn router(token: Arc<str>) -> Router<AppState> {
    let api = Router::new()
        .route("/admin/api/players", get(list_players))
        .route("/admin/api/players/:id", get(player_detail))
        .route(
            "/admin/api/players/:id/reset-password",
            post(reset_password),
        )
        .route("/admin/api/metrics", get(metrics))
        .route("/admin/api/games", get(recent_games))
        .layer(middleware::from_fn_with_state(token, check_token));

    Router::new()
        .route("/admin", get(index))
        .route("/admin/", get(index))
        .route("/admin/admin.css", get(css))
        .route("/admin/admin.js", get(js))
        .merge(api)
}

async fn check_token(State(token): State<Arc<str>>, req: Request, next: Next) -> Response {
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if ct_eq(provided.as_bytes(), token.as_bytes()) {
        next.run(req).await
    } else {
        err(StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

// ---- Static UI ----

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/admin.html"))
}

async fn css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../static/admin.css"),
    )
}

async fn js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../static/admin.js"),
    )
}

// ---- API handlers ----

async fn list_players(
    State(app): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrResp>)> {
    let players = app
        .db
        .admin_list_players()
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({ "players": players })))
}

async fn player_detail(
    State(app): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrResp>)> {
    let player = app
        .db
        .admin_player(id)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let Some(player) = player else {
        return Err(err(StatusCode::NOT_FOUND, "player not found"));
    };
    let games = app
        .db
        .admin_player_games(id, 50)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let position_counts = app
        .db
        .admin_player_position_counts(id)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({
        "player": player,
        "games": games,
        "position_counts": position_counts,
    })))
}

async fn reset_password(
    State(app): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrResp>)> {
    match app.db.admin_reset_password(id).await {
        Ok(Some(temp)) => Ok(Json(json!({ "temp_password": temp }))),
        Ok(None) => Err(err(StatusCode::NOT_FOUND, "player not found")),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

async fn metrics(
    State(app): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrResp>)> {
    let metrics = app
        .db
        .admin_metrics()
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(metrics))
}

#[derive(Deserialize)]
struct GamesQuery {
    limit: Option<i64>,
}

async fn recent_games(
    State(app): State<AppState>,
    Query(q): Query<GamesQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrResp>)> {
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let games = app
        .db
        .admin_recent_games(limit)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({ "games": games })))
}
