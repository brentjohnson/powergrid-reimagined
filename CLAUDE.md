# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build everything
cargo build

# Run tests (game logic lives here)
cargo test -p powergrid-core

# Check types/lints
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt

# Check code
cargo check

# Run a single test
cargo test -p powergrid-core test_join_and_start

# Run the legacy single-game server (from repo root)
cargo run -p powergrid-server

# Run the lobby server (requires DATABASE_URL)
DATABASE_URL=postgres://... cargo run -p powergrid-lobby

# Run the client
cargo run -p powergrid-client
cargo run -p powergrid-client --features dev   # fast incremental rebuilds

# Run standalone bots (against a running server)
cargo run -p powergrid-bot -- --name BotA --color red
cargo run -p powergrid-bot -- --name BotB --color blue

# Docker (lobby + postgres)
docker compose up --build
```

## Workflow

Before running a build, do "cargo fmt" "cargo check" and run clippy.  Then fix any issues before building.
When making architectural or structural changes, update CLAUDE.md accordingly.

## Architecture

Seven-crate Cargo workspace:

```
crates/
  powergrid-core/          # pure game logic, no I/O
  powergrid-session/       # shared Session abstraction: apply_action, broadcast, BotPump
  powergrid-bot-strategy/  # pure strategy lib — no I/O, no tokio
  powergrid-bot/           # re-exports strategy + WS runtime; standalone bot binary
  powergrid-server/        # legacy single-game axum WS server; also embeddable as a lib
  powergrid-lobby/         # production multi-game server: auth, rooms, in-process bots, PostgreSQL
  powergrid-client/        # Bevy/egui GUI — online (lobby) or local play (in-process session)
assets/
  maps/germany.toml        # canonical map asset, embedded at compile time via powergrid-core
```

Dependency graph: core ← bot-strategy ← {bot, session} ← {server, lobby, client}.

### powergrid-core

All game state and rules. The key entry point is `rules::apply_action(state, player_id, action) -> Result<(), ActionError>`. It's pure — no I/O — and fully unit-testable.

- `types.rs` — `Player`, `PowerPlant`, `ResourceMarket`, `PlantMarket`, `Phase`, `PlayerColor`, `PlayerId` (Uuid alias), etc.
- `state.rs` — `GameState` struct (all game data including the map)
- `actions.rs` — all wire types: `Action` (game moves), `ActionError`, `ServerMessage`, `ClientMessage` (lobby envelope), `LobbyAction`, `RoomSummary`. The file contains both pure-game types and lobby-protocol types.
- `map.rs` — `Map` (runtime graph) + `MapData` (TOML-deserializable). Dijkstra routing in `Map::connection_cost_to`.
- `rules.rs` — `apply_action` dispatcher + one `handle_*` function per phase. Also `build_plant_deck()`.

**Phase flow:** `Lobby → Auction → BuyResources → BuildCities → Bureaucracy → [next round or GameOver]`

### powergrid-session

Shared game session abstraction used by both server and lobby.

- `lib.rs` — `Session { game, subscribers, bots }`. Methods: `apply(actor, action)` (calls `apply_action`, broadcasts `StateUpdate`), `add_subscriber(Subscriber)`, `add_bot(name, color)`, `remove_bot(id)`, `broadcast(msg)`.
- `Subscriber` — two variants: `Mpsc(UnboundedSender<String>)` serializes to JSON (WS use); `Local(crossbeam::Sender<ServerMessage>)` sends typed messages (in-process use).
- `run_bot_pump(Arc<Mutex<Session>>, delay)` — drives all in-process bots until none has a move or 50-iteration cap is hit; releases lock between turns.
- `MAX_PLAYERS: u8 = 6` — single workspace-level constant.

### powergrid-bot-strategy

Pure strategy lib with no I/O. Depended on by session, lobby, and client.

- `strategy.rs` — `decide(state, me) -> Option<Action>` — pure function with one `decide_*` helper per phase.

### powergrid-server

Legacy single-game WS server. No auth, no room concept. Also runnable as a standalone binary for LAN play.

- `lib.rs` — `pub async fn serve_embedded(map, addr) -> (SocketAddr, impl Future)`. Wraps `Session` in `Arc<Mutex<Session>>`. Used only for LAN play via the standalone binary; no longer embedded in the client.
- `main.rs` — thin binary wrapper. Reads `PORT`/`MAP_FILE` env vars, uses `powergrid_core::default_map()`, calls `serve_embedded`.
- `ws.rs` — per-connection handler using `Session::apply`; prunes dead subscribers via `retain`.
- Configured via env vars: `PORT` (default 3000), `MAP_FILE` (optional override), `RUST_LOG`.

### powergrid-lobby

Production multi-game server. Handles auth, room lifecycle, and in-process bots. Requires PostgreSQL (`DATABASE_URL` env var).

- `main.rs` — axum router: `/health`, `/rooms` (REST), `/ws`, `/auth/{register,login,logout}`. `AppState { manager, bot_delay, db }`.
- `ws.rs` — `ConnState { user_id, username, current_room, tx }`. Pre-auth gate: expects `ClientMessage::Authenticate { token }` as the first message (10s timeout). On success dispatches `Lobby(LobbyAction)` and `Room { room, action }` messages.
- `rooms.rs` — `Room { name, game, humans, bots, creator_user_id }` with `broadcast`, `broadcast_msg`, `add_bot`, `remove_bot`, `summary`. `RoomManager` owns `RwLock<HashMap<String, Arc<Mutex<Room>>>>`.
- `lobby_handler.rs` — handles `LobbyAction` variants: `ListRooms`, `CreateRoom`, `JoinRoom`, `LeaveRoom`, `AddBot`, `RemoveBot`.
- `room_handler.rs` — handles in-game `Action`: lock room, call `apply_action`, broadcast `StateUpdate`, trigger `run_bot_pump`.
- `driver.rs` — `run_bot_pump(room_arc, delay)`: polls `strategy::decide` for each in-process bot (up to 50 iterations), applies moves via `apply_action`, broadcasts state. Bots never touch the network.
- `auth.rs` — REST handlers for register/login/logout. 32-byte URL-safe-base64 tokens, 30-day TTL.
- `db.rs` — `Db { pool: PgPool }`. Methods: `register`, `login`, `validate_token`, `logout`. Uses Argon2 for password hashing.
- Configured via env vars: `PORT` (3000), `DATABASE_URL` (required), `BOT_DELAY_MS` (250), `MAP_FILE`, `RUST_LOG`.

### powergrid-bot

Thin crate that re-exports `powergrid-bot-strategy::strategy` and adds a WS runtime. Standalone bot binary.

- `lib.rs` — `pub use powergrid_bot_strategy::strategy; pub mod runtime;`
- `runtime.rs` — `pub async fn run_bot(url, name, color)` — WS connect loop (legacy protocol), calls `strategy::decide` each turn. Used only by the standalone binary (and useful for testing remote servers). **Not used by the client or lobby.**
- `main.rs` — CLI arg parsing (`--name`, `--color`, `--server`, `--port`); calls `runtime::run_bot`.

Run with:
```bash
cargo run -p powergrid-bot -- --name BotA --color red
cargo run -p powergrid-bot -- --name BotB --color blue --server localhost --port 3000
```

> **Important:** Whenever `Action`, `ServerMessage`, or any phase/state type in `powergrid-core` changes, update `strategy.rs` in `powergrid-bot` to match. The bot mirrors the full client/server protocol and will silently produce wrong decisions or fail to compile if it falls out of sync.

### powergrid-client

Bevy + egui GUI client. Supports two modes: **online** (connects to `powergrid-lobby`) and **local** (in-process session, no TCP server, no network required).

- `main.rs` — Bevy app setup.
- `ws.rs` — `WsChannels` resource wraps crossbeam channels + oneshot shutdown. `spawn_ws(url)` creates online channels backed by a background WS worker thread. `process_ws_events` Bevy system drains incoming `WsEvent`s each frame. Only the lobby protocol is used (`ClientMessage` envelopes). Reconnects on disconnect; shutdown propagates via `WsChannels::drop`.
- `local.rs` — `start_local_session(LocalConfig) -> (WsChannels, LocalHandle)`. Creates a `Session` in-process (human + bots join, game auto-starts). Spawns a tokio runtime thread running `local_session_driver` which routes `ClientMessage::Room` actions to `Session::apply` and runs `BotPump` after each human action. Pre-queues `Connected + Authenticated + RoomJoined + StateUpdates` before the first Bevy frame. No loopback TCP. `LocalHandle` joins the runtime thread on drop.
- `state.rs` — `AppState` Bevy resource. Screen enum: `MainMenu → {LocalSetup | Login → Register → Connect → RoomBrowser} → Game`.
- `ui/` — egui UI systems:
  - `mod.rs` — `ui_system` dispatch, `setup_egui_theme`
  - `main_menu.rs` — main menu (online vs local fork)
  - `local_setup.rs` — local game config (bot count, color)
  - `connect.rs` — online connect/login form
  - `lobby.rs` — room browser + in-room lobby (player list, add/remove bots, start)
  - `top_panel.rs` — round/phase header + resource market
  - `phase_tracker.rs` — per-player phase progress dots
  - `left_panel.rs` — player info cards
  - `action_panel.rs` — phase-specific interactive controls
  - `right_panel.rs` — plant market, action console, event log
  - `helpers.rs` — shared widgets (`section_header`, `neon_button`, `send`, etc.)
- `map_panel.rs` — renders the map with city overlays.
- `card_painter.rs` — procedurally paints power plant cards using egui primitives.
- `theme.rs` — custom egui visual theme.

Run with `cargo run -p powergrid-client` or `cargo run -p powergrid-client --features dev` for fast incremental rebuilds.

### Protocol

**Online (lobby) protocol** — `ClientMessage` (tagged `"type"`) client→server:
- `Authenticate { token }` — must be first message; 10s timeout.
- `Lobby(LobbyAction)` — room management (`ListRooms`, `CreateRoom`, `JoinRoom`, `LeaveRoom`, `AddBot`, `RemoveBot`).
- `Room { room, action: Action }` — in-game move scoped to a named room.

`ServerMessage` (tagged `"type"`) server→client: `Authenticated`, `AuthError`, `Welcome`, `StateUpdate`, `ActionError`, `LobbyError`, `RoomList`, `RoomJoined`, `RoomLeft`, `Event`.

**Legacy (embedded) protocol** — bare `Action` (tagged `"type"`) client→server; same `ServerMessage` subset (`Welcome`, `StateUpdate`, `ActionError`) server→client. Used only for local play via the embedded `powergrid-server`.

Full `GameState` is broadcast to all clients after every valid action in both protocols.

### Map format

`assets/maps/*.toml` — list of `[[cities]]` (id, name, region) and `[[connections]]` (from, to, cost). The germany map is embedded at compile time via `powergrid_core::default_map()`, which all crates call. To use a custom map, set `MAP_FILE=/path/to/map.toml` at runtime.
