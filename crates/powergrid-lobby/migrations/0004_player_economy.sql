-- Per-seat cumulative economic activity over a whole game (from
-- Player::stats). Nullable with no default: games finished before this
-- migration have NULL here and are excluded from AVG()s rather than counted
-- as zeros.
ALTER TABLE game_players ADD COLUMN plants_bought      INTEGER;
ALTER TABLE game_players ADD COLUMN spent_on_plants    INTEGER;
ALTER TABLE game_players ADD COLUMN resources_bought   INTEGER;
ALTER TABLE game_players ADD COLUMN spent_on_resources INTEGER;
ALTER TABLE game_players ADD COLUMN cities_bought      INTEGER;
ALTER TABLE game_players ADD COLUMN spent_on_cities    INTEGER;
