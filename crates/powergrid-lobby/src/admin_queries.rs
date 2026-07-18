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

/// A player's aggregate performance figures (single row).
#[derive(FromRow, Serialize)]
pub struct PlayerStats {
    pub best_finish: Option<i16>,
    pub avg_cities: Option<f64>,
    pub avg_money: Option<f64>,
    pub avg_powered: Option<f64>,
    pub avg_plants: Option<f64>,
    pub avg_plants_bought: Option<f64>,
    pub avg_spent_on_plants: Option<f64>,
    pub avg_resources_bought: Option<f64>,
    pub avg_spent_on_resources: Option<f64>,
    pub avg_cities_bought: Option<f64>,
    pub avg_spent_on_cities: Option<f64>,
}

/// Average end-of-game and economic figures for all seats that finished in a
/// given position (across every recorded game). Powers the "what does a typical
/// 1st/2nd/3rd/… place game look like" breakdown.
#[derive(FromRow, Serialize)]
pub struct FinishPositionAvg {
    pub finish_position: i16,
    pub seats: i64,
    pub avg_cities: Option<f64>,
    pub avg_money: Option<f64>,
    pub avg_powered: Option<f64>,
    pub avg_plants: Option<f64>,
    pub avg_plants_bought: Option<f64>,
    pub avg_spent_on_plants: Option<f64>,
    pub avg_resources_bought: Option<f64>,
    pub avg_spent_on_resources: Option<f64>,
    pub avg_cities_bought: Option<f64>,
    pub avg_spent_on_cities: Option<f64>,
}

/// One of a player's most-owned plants across their finished games.
#[derive(FromRow, Serialize)]
pub struct PlayerPlantPref {
    pub plant_number: i16,
    pub kind: String,
    pub capacity: i16,
    pub times_held: i64,
    pub avg_finish: Option<f64>,
}

#[derive(FromRow, Serialize)]
pub struct RecentGameRow {
    pub id: Uuid,
    pub room_name: String,
    pub map_name: String,
    pub finished_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
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

/// Win/finish stats grouped by the AI strength that occupied a seat
/// ('human' plus each bot difficulty).
#[derive(FromRow, Serialize)]
pub struct DifficultyStat {
    pub difficulty: String,
    pub seats: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

/// Per-plant usage and effectiveness across all finished games.
#[derive(FromRow, Serialize)]
pub struct PlantStat {
    pub plant_number: i16,
    pub kind: String,
    pub capacity: i16,
    pub times_held: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

/// Aggregated by fuel kind rather than individual plant.
#[derive(FromRow, Serialize)]
pub struct PlantKindStat {
    pub kind: String,
    pub times_held: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

/// Win/finish stats grouped by player color (a proxy for seat identity).
#[derive(FromRow, Serialize)]
pub struct ColorStat {
    pub color: String,
    pub seats: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

/// Win/finish stats grouped by seat/turn order (1 = first seat).
#[derive(FromRow, Serialize)]
pub struct TurnOrderStat {
    pub turn_order: i16,
    pub seats: i64,
    pub wins: i64,
    pub avg_finish: Option<f64>,
}

/// How many games ended after a given number of rounds.
#[derive(FromRow, Serialize)]
pub struct RoundsBucket {
    pub rounds: i32,
    pub count: i64,
}

/// Game count and average length by table size.
#[derive(FromRow, Serialize)]
pub struct PlayerCountStat {
    pub num_players: i16,
    pub count: i64,
    pub avg_rounds: Option<f64>,
}

#[derive(Serialize)]
pub struct Metrics {
    pub total_users: i64,
    pub total_games: i64,
    pub total_seats: i64,
    pub games_last_7d: i64,
    pub avg_rounds: Option<f64>,
    pub avg_players: Option<f64>,
    pub avg_game_minutes: Option<f64>,
    pub games_per_day: Vec<GamesPerDay>,
    pub human_wins: i64,
    pub bot_wins: i64,
    pub winner_avg_cities: Option<f64>,
    pub winner_avg_money: Option<f64>,
    pub winner_avg_plants: Option<f64>,
    pub winner_avg_powered: Option<f64>,
    pub difficulty_stats: Vec<DifficultyStat>,
    pub plant_stats: Vec<PlantStat>,
    pub plant_kind_stats: Vec<PlantKindStat>,
    pub color_stats: Vec<ColorStat>,
    pub turn_order_stats: Vec<TurnOrderStat>,
    pub rounds_histogram: Vec<RoundsBucket>,
    pub player_count_dist: Vec<PlayerCountStat>,
    pub finish_position_averages: Vec<FinishPositionAvg>,
    pub leaderboard: Vec<LeaderboardRow>,
}

/// Game-level metadata for the game-detail view.
#[derive(FromRow, Serialize)]
pub struct GameMetaRow {
    pub id: Uuid,
    pub room_name: String,
    pub map_name: String,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: DateTime<Utc>,
    pub rounds: i32,
    pub num_players: i16,
}

/// One seat's full final standing for the game-detail view.
#[derive(FromRow, Serialize)]
pub struct GameSeatRow {
    pub user_id: Option<Uuid>,
    pub player_name: String,
    pub color: String,
    pub is_bot: bool,
    pub bot_difficulty: Option<String>,
    pub turn_order: Option<i16>,
    pub finish_position: i16,
    pub cities: i16,
    pub money: i32,
    pub powered: i16,
    pub plants: i16,
    pub plants_bought: Option<i32>,
    pub spent_on_plants: Option<i32>,
    pub resources_bought: Option<i32>,
    pub spent_on_resources: Option<i32>,
    pub cities_bought: Option<i32>,
    pub spent_on_cities: Option<i32>,
}

/// One plant held by a seat, for the game-detail view.
#[derive(FromRow, Serialize)]
pub struct GamePlantRow {
    pub finish_position: i16,
    pub plant_number: i16,
    pub kind: String,
    pub capacity: i16,
    pub resource_cost: i16,
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

    /// A single row of a player's aggregate performance figures.
    pub async fn admin_player_stats(&self, user_id: Uuid) -> Result<PlayerStats, sqlx::Error> {
        sqlx::query_as(
            "SELECT MIN(finish_position) AS best_finish, \
                    AVG(cities)::float8 AS avg_cities, \
                    AVG(money)::float8 AS avg_money, \
                    AVG(powered)::float8 AS avg_powered, \
                    AVG(plants)::float8 AS avg_plants, \
                    AVG(plants_bought)::float8 AS avg_plants_bought, \
                    AVG(spent_on_plants)::float8 AS avg_spent_on_plants, \
                    AVG(resources_bought)::float8 AS avg_resources_bought, \
                    AVG(spent_on_resources)::float8 AS avg_spent_on_resources, \
                    AVG(cities_bought)::float8 AS avg_cities_bought, \
                    AVG(spent_on_cities)::float8 AS avg_spent_on_cities \
             FROM game_players WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
    }

    /// A player's most-frequently-owned plants (at game end), with the average
    /// finish position achieved while holding each.
    pub async fn admin_player_favorite_plants(
        &self,
        user_id: Uuid,
        limit: i64,
    ) -> Result<Vec<PlayerPlantPref>, sqlx::Error> {
        sqlx::query_as(
            "SELECT pp.plant_number, MAX(pp.kind) AS kind, MAX(pp.capacity) AS capacity, \
                    COUNT(*) AS times_held, AVG(gp.finish_position)::float8 AS avg_finish \
             FROM game_player_plants pp \
             JOIN game_players gp \
               ON gp.game_id = pp.game_id AND gp.finish_position = pp.finish_position \
             WHERE gp.user_id = $1 \
             GROUP BY pp.plant_number \
             ORDER BY times_held DESC, pp.plant_number \
             LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }

    pub async fn admin_recent_games(&self, limit: i64) -> Result<Vec<RecentGameRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT g.id, g.room_name, g.map_name, g.finished_at, g.started_at, g.rounds, \
                    g.num_players, wp.player_name AS winner_name, wp.is_bot AS winner_is_bot \
             FROM games g \
             LEFT JOIN game_players wp ON wp.game_id = g.id AND wp.finish_position = 1 \
             ORDER BY g.finished_at DESC \
             LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
    }

    /// Game metadata for the detail view.
    pub async fn admin_game(&self, game_id: Uuid) -> Result<Option<GameMetaRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT id, room_name, map_name, started_at, finished_at, rounds, num_players \
             FROM games WHERE id = $1",
        )
        .bind(game_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// All seats of a game, ordered by finish position.
    pub async fn admin_game_seats(&self, game_id: Uuid) -> Result<Vec<GameSeatRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT user_id, player_name, color, is_bot, bot_difficulty, turn_order, \
                    finish_position, cities, money, powered, plants, \
                    plants_bought, spent_on_plants, resources_bought, spent_on_resources, \
                    cities_bought, spent_on_cities \
             FROM game_players WHERE game_id = $1 \
             ORDER BY finish_position",
        )
        .bind(game_id)
        .fetch_all(&self.pool)
        .await
    }

    /// All plants held (at game end) by every seat of a game.
    pub async fn admin_game_plants(&self, game_id: Uuid) -> Result<Vec<GamePlantRow>, sqlx::Error> {
        sqlx::query_as(
            "SELECT finish_position, plant_number, kind, capacity, resource_cost \
             FROM game_player_plants WHERE game_id = $1 \
             ORDER BY finish_position, plant_number",
        )
        .bind(game_id)
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
        let total_seats: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM game_players")
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

        let avg_game_minutes: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(EXTRACT(EPOCH FROM (finished_at - started_at)) / 60.0)::float8 \
             FROM games WHERE started_at IS NOT NULL",
        )
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

        let difficulty_stats: Vec<DifficultyStat> = sqlx::query_as(
            "SELECT CASE WHEN is_bot THEN COALESCE(bot_difficulty, 'unknown') ELSE 'human' END \
                        AS difficulty, \
                    COUNT(*) AS seats, \
                    COUNT(*) FILTER (WHERE finish_position = 1) AS wins, \
                    AVG(finish_position)::float8 AS avg_finish \
             FROM game_players \
             GROUP BY 1 ORDER BY seats DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let plant_stats: Vec<PlantStat> = sqlx::query_as(
            "SELECT pp.plant_number, MAX(pp.kind) AS kind, MAX(pp.capacity) AS capacity, \
                    COUNT(*) AS times_held, \
                    COUNT(*) FILTER (WHERE gp.finish_position = 1) AS wins, \
                    AVG(gp.finish_position)::float8 AS avg_finish \
             FROM game_player_plants pp \
             JOIN game_players gp \
               ON gp.game_id = pp.game_id AND gp.finish_position = pp.finish_position \
             GROUP BY pp.plant_number \
             ORDER BY pp.plant_number",
        )
        .fetch_all(&self.pool)
        .await?;

        let plant_kind_stats: Vec<PlantKindStat> = sqlx::query_as(
            "SELECT pp.kind, COUNT(*) AS times_held, \
                    COUNT(*) FILTER (WHERE gp.finish_position = 1) AS wins, \
                    AVG(gp.finish_position)::float8 AS avg_finish \
             FROM game_player_plants pp \
             JOIN game_players gp \
               ON gp.game_id = pp.game_id AND gp.finish_position = pp.finish_position \
             GROUP BY pp.kind \
             ORDER BY times_held DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let color_stats: Vec<ColorStat> = sqlx::query_as(
            "SELECT color, COUNT(*) AS seats, \
                    COUNT(*) FILTER (WHERE finish_position = 1) AS wins, \
                    AVG(finish_position)::float8 AS avg_finish \
             FROM game_players \
             GROUP BY color ORDER BY seats DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let turn_order_stats: Vec<TurnOrderStat> = sqlx::query_as(
            "SELECT turn_order, COUNT(*) AS seats, \
                    COUNT(*) FILTER (WHERE finish_position = 1) AS wins, \
                    AVG(finish_position)::float8 AS avg_finish \
             FROM game_players \
             WHERE turn_order IS NOT NULL \
             GROUP BY turn_order ORDER BY turn_order",
        )
        .fetch_all(&self.pool)
        .await?;

        let rounds_histogram: Vec<RoundsBucket> = sqlx::query_as(
            "SELECT rounds, COUNT(*) AS count FROM games GROUP BY rounds ORDER BY rounds",
        )
        .fetch_all(&self.pool)
        .await?;

        let player_count_dist: Vec<PlayerCountStat> = sqlx::query_as(
            "SELECT num_players, COUNT(*) AS count, AVG(rounds)::float8 AS avg_rounds \
             FROM games GROUP BY num_players ORDER BY num_players",
        )
        .fetch_all(&self.pool)
        .await?;

        let finish_position_averages: Vec<FinishPositionAvg> = sqlx::query_as(
            "SELECT finish_position, COUNT(*) AS seats, \
                    AVG(cities)::float8 AS avg_cities, \
                    AVG(money)::float8 AS avg_money, \
                    AVG(powered)::float8 AS avg_powered, \
                    AVG(plants)::float8 AS avg_plants, \
                    AVG(plants_bought)::float8 AS avg_plants_bought, \
                    AVG(spent_on_plants)::float8 AS avg_spent_on_plants, \
                    AVG(resources_bought)::float8 AS avg_resources_bought, \
                    AVG(spent_on_resources)::float8 AS avg_spent_on_resources, \
                    AVG(cities_bought)::float8 AS avg_cities_bought, \
                    AVG(spent_on_cities)::float8 AS avg_spent_on_cities \
             FROM game_players \
             GROUP BY finish_position ORDER BY finish_position",
        )
        .fetch_all(&self.pool)
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
            total_seats,
            games_last_7d,
            avg_rounds,
            avg_players,
            avg_game_minutes,
            games_per_day,
            human_wins,
            bot_wins,
            winner_avg_cities,
            winner_avg_money,
            winner_avg_plants,
            winner_avg_powered,
            difficulty_stats,
            plant_stats,
            plant_kind_stats,
            color_stats,
            turn_order_stats,
            rounds_histogram,
            player_count_dist,
            finish_position_averages,
            leaderboard,
        })
    }
}
