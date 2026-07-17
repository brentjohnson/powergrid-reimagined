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

Per-player summary tiles (games played, wins, win rate, average finish, last
login), a **finish-position distribution** chart (how often they placed 1st,
2nd, …), and a table of their most recent games with per-game cities, money,
powered cities, and plant count.

### Metrics

Server-wide statistics: total users and games, games in the last 7 days, average
rounds and players per game, a **games-per-day** chart (last 30 days), a
**human-vs-bot win split**, average end-of-game stats for winners, and a
most-wins leaderboard (human players only).

### Recent games

The most recently finished games with map, player count, round count, and the
winner (flagged human or bot).

### Reset password

From the Players table, click **Reset** on a player's row and confirm. The server:

1. Generates a new random temporary password.
2. Replaces the player's Argon2 password hash with it.
3. Deletes all of that player's active sessions (logging them out everywhere).

The temporary password is then shown **once** in a modal with a copy button —
it is not stored anywhere retrievable, so copy it and deliver it to the player
through a secure channel. The player can log in with it and change it as normal.

There is no email-based self-service reset flow; the server has no mail
infrastructure, so admin-initiated reset is the recovery path.

## Where the data comes from

Game metrics are populated by a persistence hook in the lobby: when a game
reaches `GameOver`, the final standings (`finish_ranks`) are written to the
`games` and `game_players` tables. Games that were in progress before this
feature was deployed are not retroactively recorded — metrics accrue from the
first game finished after upgrade. Player list, last login, and password reset
operate on the pre-existing `users`/`sessions` tables (with a new `last_login`
column added by migration `0002_admin.sql`, backfilled from existing sessions).

## HTTP API reference

All endpoints require the `Authorization: Bearer <ADMIN_TOKEN>` header and return
JSON.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/admin/api/players` | List all players with aggregate stats. |
| `GET`  | `/admin/api/players/:id` | One player: profile, recent games, finish-position counts. |
| `POST` | `/admin/api/players/:id/reset-password` | Reset password; returns `{ "temp_password": "…" }`. |
| `GET`  | `/admin/api/metrics` | Server-wide aggregate metrics. |
| `GET`  | `/admin/api/games?limit=N` | Recent finished games (default 50, max 500). |

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
