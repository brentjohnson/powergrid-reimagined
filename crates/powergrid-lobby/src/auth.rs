use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{db::AuthError, AppState};
use powergrid_core::limits::{MAX_EMAIL, MAX_PASSWORD, MAX_USERNAME, MIN_PASSWORD, MIN_USERNAME};

#[derive(Deserialize)]
pub struct RegisterReq {
    pub email: String,
    pub username: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct LoginReq {
    pub identifier: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct AuthResp {
    pub token: String,
    pub user_id: Uuid,
    pub username: String,
}

#[derive(Serialize)]
pub struct ErrResp {
    pub error: String,
}

fn err(status: StatusCode, msg: impl ToString) -> (StatusCode, Json<ErrResp>) {
    (
        status,
        Json(ErrResp {
            error: msg.to_string(),
        }),
    )
}

pub async fn register(
    State(app): State<AppState>,
    Json(req): Json<RegisterReq>,
) -> Result<(StatusCode, Json<AuthResp>), (StatusCode, Json<ErrResp>)> {
    let elen = req.email.chars().count();
    if !req.email.contains('@') || elen > MAX_EMAIL {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!("email must contain @ and be at most {MAX_EMAIL} characters"),
        ));
    }
    let ulen = req.username.chars().count();
    if !(MIN_USERNAME..=MAX_USERNAME).contains(&ulen) {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!("username must be {MIN_USERNAME}–{MAX_USERNAME} characters"),
        ));
    }
    if !req
        .username
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "username may only contain letters, digits, hyphens, and underscores",
        ));
    }
    let plen = req.password.chars().count();
    if !(MIN_PASSWORD..=MAX_PASSWORD).contains(&plen) {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!("password must be {MIN_PASSWORD}–{MAX_PASSWORD} characters"),
        ));
    }

    match app
        .db
        .register(&req.email, &req.username, &req.password)
        .await
    {
        Ok(s) => Ok((
            StatusCode::CREATED,
            Json(AuthResp {
                token: s.token,
                user_id: s.user_id,
                username: s.username,
            }),
        )),
        Err(AuthError::EmailTaken) => Err(err(StatusCode::CONFLICT, "email already in use")),
        Err(AuthError::UsernameTaken) => Err(err(StatusCode::CONFLICT, "username already in use")),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

pub async fn login(
    State(app): State<AppState>,
    Json(req): Json<LoginReq>,
) -> Result<Json<AuthResp>, (StatusCode, Json<ErrResp>)> {
    if req.identifier.chars().count() > MAX_EMAIL {
        return Err(err(StatusCode::BAD_REQUEST, "invalid credentials"));
    }
    if req.password.chars().count() > MAX_PASSWORD {
        return Err(err(StatusCode::BAD_REQUEST, "invalid credentials"));
    }
    match app.db.login(&req.identifier, &req.password).await {
        Ok(s) => Ok(Json(AuthResp {
            token: s.token,
            user_id: s.user_id,
            username: s.username,
        })),
        Err(AuthError::InvalidCredentials) => {
            Err(err(StatusCode::UNAUTHORIZED, "invalid credentials"))
        }
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

pub async fn logout(State(app): State<AppState>, headers: HeaderMap) -> StatusCode {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    let _ = app.db.logout(token).await;
    StatusCode::NO_CONTENT
}
