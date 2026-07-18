# Admin Console

The admin console is a web interface for administering the lobby server and its
database: browsing registered players, resetting passwords, and reviewing
game-result metrics. It is embedded in the `powergrid-lobby` binary and served at
`/admin` — there is no separate service to deploy.

## Enabling it

The console is **disabled by default**. It is mounted only when the `ADMIN_TOKEN`
environment variable is set to a non-empty value:

```bash
DATABASE_URL=postgres://powergrid:powergrid@localhost:5432/powergrid \
ADMIN_TOKEN=your-long-random-secret \
cargo run -p powergrid-lobby
```

When `ADMIN_TOKEN` is unset, every `/admin` route returns `404` and the server
logs `ADMIN_TOKEN not set; admin interface disabled`. With it set, the log reads
`Admin interface enabled at /admin`.

**Docker Compose:** uncomment and set the `ADMIN_TOKEN` line under the `lobby`
service in `docker-compose.yml`:

```yaml
    environment:
      - ADMIN_TOKEN=change-me
```

### Choosing a token

`ADMIN_TOKEN` is a single shared secret — there are no per-admin accounts or
roles. Anyone with the token has full admin access, so:

- Use a long, random value (e.g. `openssl rand -base64 32`).
- Keep it out of source control; supply it via the environment or a secrets
  manager.
- Rotate it by restarting the server with a new value. Rotation immediately
  invalidates the old token; any open admin tab will be prompted to re-enter.

## Accessing it

1. Point a browser at the lobby server's `/admin` path, e.g.
   `http://localhost:3000/admin` (both `/admin` and `/admin/` work).
2. The page loads and prompts for the admin token.
3. Enter the token and click **Unlock**. The token is verified with a probe
   request and then kept in the browser's `sessionStorage` — it is sent as an
   `Authorization: Bearer <token>` header on every API call and is **not**
   persisted to disk. Closing the tab, or clicking **Lock** in the top bar,
   forgets it.

### Authentication model

- The static UI (`/admin`, `/admin/admin.css`, `/admin/admin.js`) is served
  **without** a token — it is just the shell that prompts for one.
- The data endpoints under `/admin/api/*` are gated by a constant-time bearer
  check against `ADMIN_TOKEN`. A missing or wrong token returns
  `401 {"error":"unauthorized"}`.
- If the token is rejected mid-session (e.g. after rotation), the UI clears it
  and returns to the token prompt automatically.

The admin auth is entirely separate from player auth: it does **not** use the
`users`/`sessions` tables or the `/auth/*` login flow. Player session tokens
grant no admin access, and the admin token grants no player access.

## Features

### Players

A sortable table of every registered account: username, email, account-creation
date, last login, games played, wins, and average finish position. Click any
column header to sort; click a row to open that player's detail view.

### Player detail

Per-player summary tiles (games played, wins, win rate, average finish, best
finish, last login), a **career averages** panel — end-of-game figures
(cities/money/powered/plants) plus per-game economy totals (plants bought and
Elektro spent on them, resources bought and spent, cities built and spent) — a
**finish-position distribution** chart (how often they placed 1st, 2nd, …), a
**favorite plants** table (their most-owned plants at game end and the average
finish achieved while holding each), and a table of their most recent games.

### Metrics

Server-wide statistics:

- **Totals** — users, games, seats played, games in the last 7 days, average
  rounds/players per game, and average game length (wall-clock).
- **Games-per-day** chart (last 30 days) and **human-vs-bot win split**.
- **Winner averages** — mean end-of-game cities/money/plants/powered.
- **Averages by finish position** — a matrix of mean end-of-game *and* total-spend
  figures (cities/powered/money/plants held, plus plants/resources/cities bought
  and the Elektro spent on each) for every seat that finished 1st, 2nd, 3rd, …,
  so you can compare what a winning game looks like against a losing one.
- **Performance by opponent type** — seats, wins, win rate, and average finish
  broken out by human vs each bot difficulty (easy/normal/hard/expert).
- **Performance by color** and **win rate by seat / turn order** — whether seat
  identity or going first/last correlates with winning.
- **Table size** distribution and a **game-length (rounds)** histogram.
- **Fuel kinds** and a sortable **power-plant effectiveness** table — for plants
  still held at game end, how often each shows up and the average finish (and
  win rate) of seats holding it, so you can see which plants win and which lose.
- Most-wins **leaderboard** (human players only).

### Recent games

The most recently finished games with map, player count, round count, wall-clock
length, and the winner (flagged human or bot). Click any row to open the
**game detail** view: full final standings (place, color, human/bot difficulty,
cities/money/powered/plants/seat), each seat's economy for that game (plants
bought and spent, resources bought and spent, cities built and spent), and every
power plant each seat held at the end.

### Reset password

From the Players table, click **Reset** on a player's row. A modal opens with a
password input; type a new password (8–128 characters) or click **Generate** to
fill the field with a random one, then click **Save**. The server:

1. Replaces the player's Argon2 password hash with the supplied password.
2. Deletes all of that player's active sessions (logging them out everywhere).

Deliver the new password to the player through a secure channel. The player can
log in with it and change it as normal.

There is no email-based self-service reset flow; the server has no mail
infrastructure, so admin-initiated reset is the recovery path.

## Where the data comes from

Game metrics are populated by a persistence hook in the lobby: when a game
reaches `GameOver`, the final standings (`finish_ranks`) are written to the
`games` and `game_players` tables, and each seat's end-of-game plants to
`game_player_plants`. Games that were in progress before this feature was
deployed are not retroactively recorded — metrics accrue from the first game
finished after upgrade. The bot-difficulty, turn-order, and plant breakdowns
only populate for games finished after migration `0003_metrics.sql`, and the
per-seat economy figures (spend/activity totals) only after `0004_player_economy.sql`
(older rows have NULL for those columns and are excluded from the corresponding
charts and averages).
Player list, last login, and password reset operate on the pre-existing
`users`/`sessions` tables (with a new `last_login` column added by migration
`0002_admin.sql`, backfilled from existing sessions).

## HTTP API reference

All endpoints require the `Authorization: Bearer <ADMIN_TOKEN>` header and return
JSON.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/admin/api/players` | List all players with aggregate stats. |
| `GET`  | `/admin/api/players/:id` | One player: profile, recent games, finish-position counts, career averages, favorite plants. |
| `POST` | `/admin/api/players/:id/reset-password` | Set the player's password to the supplied `{ "password": "…" }` (8–128 chars) and revoke their sessions; returns `{ "ok": true }`. |
| `GET`  | `/admin/api/metrics` | Server-wide aggregate metrics (see the Metrics section for the full breakdown). |
| `GET`  | `/admin/api/games?limit=N` | Recent finished games (default 50, max 500). |
| `GET`  | `/admin/api/games/:id` | One game: metadata, full standings, and per-seat plants. |

Example:

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:3000/admin/api/players
```

## Security notes

- Serve the lobby behind TLS in production — the admin token is a bearer
  credential and must not travel over plaintext HTTP.
- The server sends permissive CORS headers, so the admin API is reachable
  cross-origin by anyone who has the token; treat the token as the only barrier
  and protect it accordingly.
- Consider restricting `/admin*` at your reverse proxy (IP allow-list, etc.) as
  defense in depth, since a single shared token is the whole access model.
