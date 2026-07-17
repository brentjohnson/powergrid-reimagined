-- Track last login directly. sessions rows are deleted on logout, so
-- MAX(sessions.created_at) is an unreliable proxy; a dedicated column is set
-- on every login/register.
ALTER TABLE users ADD COLUMN last_login TIMESTAMPTZ;
UPDATE users SET last_login = s.max_created
FROM (SELECT user_id, max(created_at) AS max_created FROM sessions GROUP BY user_id) s
WHERE s.user_id = users.id;

-- Persisted results of finished games (rooms are otherwise dropped from memory
-- when a game ends, so this is the only durable record of play).
CREATE TABLE games (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_name   TEXT NOT NULL,
    map_name    TEXT NOT NULL,
    started_at  TIMESTAMPTZ,                         -- NULL if start wasn't observed
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    rounds      INTEGER NOT NULL,
    num_players SMALLINT NOT NULL
);
CREATE INDEX games_finished_at_idx ON games (finished_at);

-- One row per seat in a finished game. user_id is set for human seats
-- (PlayerId == users.id) and NULL for bots.
CREATE TABLE game_players (
    game_id         UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    player_name     TEXT NOT NULL,
    color           TEXT NOT NULL,
    is_bot          BOOLEAN NOT NULL,
    finish_position SMALLINT NOT NULL,               -- 1-based, 1 = winner
    cities          SMALLINT NOT NULL,
    money           INTEGER NOT NULL,
    powered         SMALLINT NOT NULL,               -- last_cities_powered
    plants          SMALLINT NOT NULL,               -- plant count at game end
    PRIMARY KEY (game_id, finish_position)
);
CREATE INDEX game_players_user_id_idx ON game_players (user_id);
