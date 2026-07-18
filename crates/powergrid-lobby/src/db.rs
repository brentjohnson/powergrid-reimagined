use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{Duration, Utc};
use rand_core::RngCore;
use sqlx::PgPool;
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone)]
pub struct Db {
    pub pool: PgPool,
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("email already in use")]
    EmailTaken,
    #[error("username already in use")]
    UsernameTaken,
    #[error("invalid credentials")]
    InvalidCredentials,
    #[error("invalid or expired session")]
    InvalidSession,
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("password hash error")]
    Hash,
}

pub struct AuthSession {
    pub user_id: Uuid,
    pub username: String,
    pub token: String,
}

fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// A finished game and its final standings, ready to persist.
pub struct GameRecord {
    pub room_name: String,
    pub map_name: String,
    pub started_at: Option<chrono::DateTime<Utc>>,
    pub rounds: i32,
    pub seats: Vec<SeatRecord>,
}

/// One seat's final state in a finished game.
pub struct SeatRecord {
    pub user_id: Option<Uuid>,
    pub player_name: String,
    pub color: String,
    pub is_bot: bool,
    /// Difficulty for bot seats ("easy"|"normal"|"hard"|"expert"); None for humans.
    pub bot_difficulty: Option<String>,
    /// 1-based seat/turn order at game end.
    pub turn_order: i16,
    pub finish_position: i16,
    pub cities: i16,
    pub money: i32,
    pub powered: i16,
    pub plants: i16,
    // Cumulative economic activity over the whole game (from `Player::stats`).
    pub plants_bought: i32,
    pub spent_on_plants: i32,
    pub resources_bought: i32,
    pub spent_on_resources: i32,
    pub cities_bought: i32,
    pub spent_on_cities: i32,
    /// Every plant this seat still owned at game end.
    pub plant_details: Vec<PlantRecord>,
}

/// One power plant held by a seat at game end.
pub struct PlantRecord {
    pub number: i16,
    pub kind: String,
    pub capacity: i16,
    pub resource_cost: i16,
}

fn map_insert_error(e: sqlx::Error) -> AuthError {
    if let sqlx::Error::Database(ref db_err) = e {
        if db_err.code().as_deref() == Some("23505") {
            let constraint = db_err.constraint().unwrap_or("");
            if constraint.contains("email") {
                return AuthError::EmailTaken;
            }
            return AuthError::UsernameTaken;
        }
    }
    AuthError::Db(e)
}

impl Db {
    pub async fn connect(url: &str) -> Result<Self, sqlx::Error> {
        let pool = PgPool::connect(url).await?;
        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> Result<(), sqlx::migrate::MigrateError> {
        sqlx::migrate!("./migrations").run(&self.pool).await
    }

    pub async fn register(
        &self,
        email: &str,
        username: &str,
        password: &str,
    ) -> Result<AuthSession, AuthError> {
        let email = email.to_lowercase();
        let salt = SaltString::generate(&mut OsRng);
        let hash = Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map_err(|_| AuthError::Hash)?
            .to_string();

        let user_id: Uuid = sqlx::query_scalar(
            "INSERT INTO users (email, username, password_hash) VALUES ($1, $2, $3) RETURNING id",
        )
        .bind(&email)
        .bind(username)
        .bind(&hash)
        .fetch_one(&self.pool)
        .await
        .map_err(map_insert_error)?;

        let token = generate_token();
        let expires_at = Utc::now() + Duration::days(30);
        sqlx::query("INSERT INTO sessions (token, user_id, expires_at) VALUES ($1, $2, $3)")
            .bind(&token)
            .bind(user_id)
            .bind(expires_at)
            .execute(&self.pool)
            .await?;

        sqlx::query("UPDATE users SET last_login = now() WHERE id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok(AuthSession {
            user_id,
            username: username.to_string(),
            token,
        })
    }

    pub async fn login(&self, identifier: &str, password: &str) -> Result<AuthSession, AuthError> {
        let identifier_lower = identifier.to_lowercase();

        let row: Option<(Uuid, String, String)> = sqlx::query_as(
            "SELECT id, username, password_hash FROM users \
             WHERE email = $1 OR lower(username) = $1",
        )
        .bind(&identifier_lower)
        .fetch_optional(&self.pool)
        .await?;

        let (user_id, username, hash_str) = row.ok_or(AuthError::InvalidCredentials)?;

        let parsed = PasswordHash::new(&hash_str).map_err(|_| AuthError::Hash)?;
        Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .map_err(|_| AuthError::InvalidCredentials)?;

        let token = generate_token();
        let expires_at = Utc::now() + Duration::days(30);
        sqlx::query("INSERT INTO sessions (token, user_id, expires_at) VALUES ($1, $2, $3)")
            .bind(&token)
            .bind(user_id)
            .bind(expires_at)
            .execute(&self.pool)
            .await?;

        sqlx::query("UPDATE users SET last_login = now() WHERE id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok(AuthSession {
            user_id,
            username,
            token,
        })
    }

    pub async fn validate_token(&self, token: &str) -> Result<(Uuid, String), AuthError> {
        let row: Option<(Uuid, String)> = sqlx::query_as(
            "SELECT s.user_id, u.username \
             FROM sessions s JOIN users u ON u.id = s.user_id \
             WHERE s.token = $1 AND s.expires_at > now()",
        )
        .bind(token)
        .fetch_optional(&self.pool)
        .await?;

        let (user_id, username) = row.ok_or(AuthError::InvalidSession)?;

        let new_expires = Utc::now() + Duration::days(30);
        sqlx::query("UPDATE sessions SET expires_at = $1 WHERE token = $2")
            .bind(new_expires)
            .bind(token)
            .execute(&self.pool)
            .await?;

        // Refresh last_login so the admin console reflects token-based
        // reconnects (the WS client authenticates with a saved token and never
        // hits the REST login path that would otherwise update this).
        sqlx::query("UPDATE users SET last_login = now() WHERE id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok((user_id, username))
    }

    pub async fn logout(&self, token: &str) -> Result<(), AuthError> {
        sqlx::query("DELETE FROM sessions WHERE token = $1")
            .bind(token)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Persist a finished game and its per-seat standings in one transaction.
    pub async fn record_game(&self, rec: &GameRecord) -> Result<Uuid, sqlx::Error> {
        let mut tx = self.pool.begin().await?;
        let game_id: Uuid = sqlx::query_scalar(
            "INSERT INTO games (room_name, map_name, started_at, rounds, num_players) \
             VALUES ($1, $2, $3, $4, $5) RETURNING id",
        )
        .bind(&rec.room_name)
        .bind(&rec.map_name)
        .bind(rec.started_at)
        .bind(rec.rounds)
        .bind(rec.seats.len() as i16)
        .fetch_one(&mut *tx)
        .await?;

        for s in &rec.seats {
            sqlx::query(
                "INSERT INTO game_players \
                 (game_id, user_id, player_name, color, is_bot, bot_difficulty, turn_order, \
                  finish_position, cities, money, powered, plants, \
                  plants_bought, spent_on_plants, resources_bought, spent_on_resources, \
                  cities_bought, spent_on_cities) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, \
                         $13, $14, $15, $16, $17, $18)",
            )
            .bind(game_id)
            .bind(s.user_id)
            .bind(&s.player_name)
            .bind(&s.color)
            .bind(s.is_bot)
            .bind(&s.bot_difficulty)
            .bind(s.turn_order)
            .bind(s.finish_position)
            .bind(s.cities)
            .bind(s.money)
            .bind(s.powered)
            .bind(s.plants)
            .bind(s.plants_bought)
            .bind(s.spent_on_plants)
            .bind(s.resources_bought)
            .bind(s.spent_on_resources)
            .bind(s.cities_bought)
            .bind(s.spent_on_cities)
            .execute(&mut *tx)
            .await?;

            for p in &s.plant_details {
                sqlx::query(
                    "INSERT INTO game_player_plants \
                     (game_id, finish_position, plant_number, kind, capacity, resource_cost) \
                     VALUES ($1, $2, $3, $4, $5, $6)",
                )
                .bind(game_id)
                .bind(s.finish_position)
                .bind(p.number)
                .bind(&p.kind)
                .bind(p.capacity)
                .bind(p.resource_cost)
                .execute(&mut *tx)
                .await?;
            }
        }

        tx.commit().await?;
        Ok(game_id)
    }

    /// Reset a user's password to the admin-supplied value, revoke all of their
    /// sessions, and return `Ok(true)`. `Ok(false)` if no such user.
    pub async fn admin_reset_password(
        &self,
        user_id: Uuid,
        new_password: &str,
    ) -> Result<bool, AuthError> {
        let salt = SaltString::generate(&mut OsRng);
        let hash = Argon2::default()
            .hash_password(new_password.as_bytes(), &salt)
            .map_err(|_| AuthError::Hash)?
            .to_string();

        let mut tx = self.pool.begin().await?;
        let updated = sqlx::query("UPDATE users SET password_hash = $1 WHERE id = $2")
            .bind(&hash)
            .bind(user_id)
            .execute(&mut *tx)
            .await?;
        if updated.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        sqlx::query("DELETE FROM sessions WHERE user_id = $1")
            .bind(user_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(true)
    }
}
