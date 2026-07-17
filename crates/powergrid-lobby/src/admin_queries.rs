//! Read-only queries backing the admin API. A second `impl Db` block, kept
//! separate from `db.rs` so the auth-critical code stays focused.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::FromRow;
use uuid::Uuid;

use crate::db::Db;

#[derive(FromRow, Serialize)]
pub struct AdminPlayerRow {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
    pub games_played: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

#[derive(FromRow, Serialize)]
pub struct PlayerGameRow {
    pub game_id: Uuid,
    pub room_name: String,
    pub map_name: String,
    pub finished_at: DateTime<Utc>,
    pub rounds: i32,
    pub num_players: i16,
    pub finish_position: i16,
    pub cities: i16,
    pub money: i32,
    pub powered: i16,
    pub plants: i16,
}

#[derive(FromRow, Serialize)]
pub struct PositionCount {
    pub finish_position: i16,
    pub count: i64,
}

#[derive(FromRow, Serialize)]
pub struct RecentGameRow {
    pub id: Uuid,
    pub room_name: String,
    pub map_name: String,
    pub finished_at: DateTime<Utc>,
    pub rounds: i32,
    pub num_players: i16,
    pub winner_name: Option<String>,
    pub winner_is_bot: Option<bool>,
}

#[derive(FromRow, Serialize)]
pub struct GamesPerDay {
    pub day: DateTime<Utc>,
    pub count: i64,
}

#[derive(FromRow, Serialize)]
pub struct LeaderboardRow {
    pub username: String,
    pub games_played: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

#[derive(Serialize)]
pub struct Metrics {
    pub total_users: i64,
    pub total_games: i64,
    pub games_last_7d: i64,
    pub avg_rounds: Option<f64>,
    pub avg_players: Option<f64>,
    pub games_per_day: Vec<GamesPerDay>,
    pub human_wins: i64,
    pub bot_wins: i64,
    pub winner_avg_cities: Option<f64>,
    pub winner_avg_money: Option<f64>,
    pub winner_avg_plants: Option<f64>,
    pub winner_avg_powered: Option<f64>,
    pub leaderboard: Vec<LeaderboardRow>,
}

impl Db {
    pub async fn admin_list_players(&self) -> Result<Vec<AdminPlayerRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT u.id, u.username, u.email, u.created_at, u.last_login, \
                    COUNT(gp.game_id) AS games_played, \
                    COUNT(*) FILTER (WHERE gp.finish_position = 1) AS wins, \
                    AVG(gp.finish_position)::float8 AS avg_finish \
             FROM users u \
             LEFT JOIN game_players gp ON gp.user_id = u.id \
             GROUP BY u.id \
             ORDER BY u.created_at",
        )
        .fetch_all(&self.pool)
        .await
    }

    pub async fn admin_player(&self, user_id: Uuid) -> Result<Option<AdminPlayerRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT u.id, u.username, u.email, u.created_at, u.last_login, \
                    COUNT(gp.game_id) AS games_played, \
                    COUNT(*) FILTER (WHERE gp.finish_position = 1) AS wins, \
                    AVG(gp.finish_position)::float8 AS avg_finish \
             FROM users u \
             LEFT JOIN game_players gp ON gp.user_id = u.id \
             WHERE u.id = $1 \
             GROUP BY u.id",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
    }

    pub async fn admin_player_games(
        &self,
        user_id: Uuid,
        limit: i64,
    ) -> Result<Vec<PlayerGameRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT g.id AS game_id, g.room_name, g.map_name, g.finished_at, g.rounds, \
                    g.num_players, gp.finish_position, gp.cities, gp.money, gp.powered, gp.plants \
             FROM game_players gp \
             JOIN games g ON g.id = gp.game_id \
             WHERE gp.user_id = $1 \
             ORDER BY g.finished_at DESC \
             LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }

    pub async fn admin_player_position_counts(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<PositionCount>, sqlx::Error> {
        sqlx::query_as(
            "SELECT finish_position, COUNT(*) AS count \
             FROM game_players \
             WHERE user_id = $1 \
             GROUP BY finish_position \
             ORDER BY finish_position",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
    }

    pub async fn admin_recent_games(&self, limit: i64) -> Result<Vec<RecentGameRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT g.id, g.room_name, g.map_name, g.finished_at, g.rounds, g.num_players, \
                    wp.player_name AS winner_name, wp.is_bot AS winner_is_bot \
             FROM games g \
             LEFT JOIN game_players wp ON wp.game_id = g.id AND wp.finish_position = 1 \
             ORDER BY g.finished_at DESC \
             LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }

    pub async fn admin_metrics(&self) -> Result<Metrics, sqlx::Error> {
        let total_users: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM users")
            .fetch_one(&self.pool)
            .await?;
        let total_games: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM games")
            .fetch_one(&self.pool)
            .await?;
        let games_last_7d: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM games WHERE finished_at > now() - interval '7 days'",
        )
        .fetch_one(&self.pool)
        .await?;

        let (avg_rounds, avg_players): (Option<f64>, Option<f64>) =
            sqlx::query_as("SELECT AVG(rounds)::float8, AVG(num_players)::float8 FROM games")
                .fetch_one(&self.pool)
                .await?;

        let games_per_day: Vec<GamesPerDay> = sqlx::query_as(
            "SELECT date_trunc('day', finished_at) AS day, COUNT(*) AS count \
             FROM games \
             WHERE finished_at > now() - interval '30 days' \
             GROUP BY 1 ORDER BY 1",
        )
        .fetch_all(&self.pool)
        .await?;

        let win_rows: Vec<(bool, i64)> = sqlx::query_as(
            "SELECT is_bot, COUNT(*) FROM game_players WHERE finish_position = 1 GROUP BY is_bot",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut human_wins = 0;
        let mut bot_wins = 0;
        for (is_bot, count) in win_rows {
            if is_bot {
                bot_wins = count;
            } else {
                human_wins = count;
            }
        }

        let (winner_avg_cities, winner_avg_money, winner_avg_plants, winner_avg_powered): (
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<f64>,
        ) = sqlx::query_as(
            "SELECT AVG(cities)::float8, AVG(money)::float8, AVG(plants)::float8, \
                    AVG(powered)::float8 \
             FROM game_players WHERE finish_position = 1",
        )
        .fetch_one(&self.pool)
        .await?;

        let leaderboard: Vec<LeaderboardRow> = sqlx::query_as(
            "SELECT u.username, \
                    COUNT(gp.game_id) AS games_played, \
                    COUNT(*) FILTER (WHERE gp.finish_position = 1) AS wins, \
                    AVG(gp.finish_position)::float8 AS avg_finish \
             FROM users u \
             JOIN game_players gp ON gp.user_id = u.id \
             GROUP BY u.id \
             HAVING COUNT(gp.game_id) > 0 \
             ORDER BY wins DESC, avg_finish ASC \
             LIMIT 10",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(Metrics {
            total_users,
            total_games,
            games_last_7d,
            avg_rounds,
            avg_players,
            games_per_day,
            human_wins,
            bot_wins,
            winner_avg_cities,
            winner_avg_money,
            winner_avg_plants,
            winner_avg_powered,
            leaderboard,
        })
    }
}
