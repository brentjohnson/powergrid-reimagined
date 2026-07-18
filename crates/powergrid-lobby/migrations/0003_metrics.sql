-- Richer game-result metrics.

-- Which kind of bot occupied a seat ('easy' | 'normal' | 'hard' | 'expert').
-- NULL for human seats. Lets the admin console break wins/finishes down by the
-- AI strength that was played against (or as).
ALTER TABLE game_players ADD COLUMN bot_difficulty TEXT;

-- Turn order (1-based join/seat index) so we can measure whether going first
-- (or last) correlates with winning. NULL for pre-migration rows.
ALTER TABLE game_players ADD COLUMN turn_order SMALLINT;

-- Every power plant a seat still owned at game end. One row per plant, so we can
-- answer "which plants show up in wins vs losses" and per-plant effectiveness.
CREATE TABLE game_player_plants (
    game_id         UUID NOT NULL,
    finish_position SMALLINT NOT NULL,
    plant_number    SMALLINT NOT NULL,          -- the plant's market number / id
    kind            TEXT NOT NULL,              -- coal|oil|gasoroil|gas|uranium|wind
    capacity        SMALLINT NOT NULL,          -- cities this plant can power
    resource_cost   SMALLINT NOT NULL,          -- resources burned per firing
    FOREIGN KEY (game_id, finish_position)
        REFERENCES game_players (game_id, finish_position) ON DELETE CASCADE
);
CREATE INDEX game_player_plants_game_idx ON game_player_plants (game_id);
CREATE INDEX game_player_plants_number_idx ON game_player_plants (plant_number);
