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

# Run the lobby server (requires DATABASE_URL)
DATABASE_URL=postgres://... cargo run -p powergrid-lobby

# Run the client
cargo run -p powergrid-client
cargo run -p powergrid-client --features dev   # fast incremental rebuilds

# Docker (lobby + postgres)
docker compose up --build

# RL environment (Python — run from python/ directory)
make develop                                     # build PyO3 crate + install Python package
pytest tests/                                    # run Python tests
python scripts/train_vs_bots.py                  # MaskablePPO vs Rust bots
python scripts/train_selfplay.py                 # self-play
python scripts/play_game.py --all-bots --render  # watch a rollout
```

## Workflow

Before running a build, do "cargo fmt" "cargo check" and run clippy.  Then fix any issues before building.  
When making architectural or structural changes, update CLAUDE.md accordingly.

When making visual changes, do not attempt to launch the game to verify changes.  Tell me what to verify and I will do it manually.

If adding or removing crates, update the stubs in the Dockerfile.

## Architecture

Seven-crate Cargo workspace:

```
crates/
  powergrid-core/          # pure game logic, no I/O
  powergrid-session/       # shared Session abstraction: apply_action, broadcast, BotPump
  powergrid-bot-strategy/  # bot AI: Bot struct, BotProfile/TOML profiles, weighted strategy
  powergrid-lobby/         # production multi-game server: auth, rooms, in-process bots, PostgreSQL
  powergrid-client/        # egui GUI — online (lobby) or local play (in-process session)
  powergrid-py/            # PyO3 extension module for the Python RL environment
  powergrid-maptool/       # egui desktop tool for creating/editing map TOML files
assets/
  maps/germany.toml        # canonical map asset, embedded at compile time via powergrid-core
  bots/default.toml        # default bot profiles (BotProfile weights), embedded at compile time
python/                    # PettingZoo RL environment (see docs/rl-environment.md)
  src/powergrid_env/       # Python package: AECEnv, encoding, policies
  scripts/                 # training and rollout scripts
  tests/                   # 21 Python tests
  pyproject.toml           # hatchling build; maturin builds the Rust extension separately
  Makefile                 # make develop = build Rust + install Python
```

Dependency graph (Rust): core ← bot-strategy ← {session, powergrid-py} ← {lobby, client}.

`powergrid-py` depends only on `powergrid-core` and `powergrid-bot-strategy` — no server, lobby, or async runtime.

### powergrid-core

All game state and rules. The key entry point is `rules::apply_action(state, player_id, action) -> Result<(), ActionError>`. It's pure — no I/O — and fully unit-testable.

- `types.rs` — `Player`, `PowerPlant`, `ResourceMarket`, `PlantMarket`, `Phase`, `PlayerColor`, `PlayerId` (Uuid alias), etc.
- `limits.rs` — validation constants: `MAX_PLAYER_NAME`, `MAX_ROOM_NAME`, `MIN_USERNAME`, `MAX_USERNAME`, `MAX_EMAIL`, `MIN_PASSWORD`, `MAX_PASSWORD`, etc.
- `state.rs` — `GameState` struct (all game data including the map); `GameStateView` is the wire-safe projection.
- `actions/` — all wire types, split across three files:
  - `game.rs` — `Action` (game moves), `ActionError`
  - `protocol.rs` — `ServerMessage`, `ClientMessage`, `LobbyAction`, `RoomSummary`, `AuthError`, `LobbyError`
  - `hints.rs` — `HintPayload`
  - `mod.rs` — re-exports + `PROTOCOL_VERSION` constant
- `map.rs` — `Map` (runtime graph) + `MapData` (TOML-deserializable). Dijkstra routing in `Map::connection_cost_to`.
- `rules.rs` — `apply_action` dispatcher + one `handle_*` function per phase. Also `build_plant_deck()`.

**Phase flow:** `Lobby → Auction → BuyResources → BuildCities → Bureaucracy → [next round or GameOver]`

### powergrid-session

Shared game session abstraction used by both lobby and client.

- `lib.rs` — `Session { game, subscribers, bots }`. Methods: `apply(actor, action)` (calls `apply_action`, broadcasts `StateUpdate`), `add_subscriber(Subscriber)`, `add_bot(name, color, difficulty) -> Result<PlayerId, ActionError>`, `remove_bot(id)`, `broadcast(msg)`.
- `Subscriber` — two variants: `Mpsc(UnboundedSender<String>)` serializes to JSON (WS use); `Local(crossbeam::Sender<ServerMessage>)` sends typed messages (in-process use).
- `run_bot_pump(Arc<Mutex<Session>>, delay)` — drives all in-process bots until none has a move or 500-iteration cap is hit; releases lock between turns.
- `MAX_PLAYERS: u8 = 6` — single workspace-level constant.

### powergrid-bot-strategy

Pure strategy + AI lib. No I/O, no tokio. Depended on by session, lobby, and client.

- `bot.rs` — `Bot { id, name, color, profile, rng }`: stateful bot with a seeded `SmallRng`. `decide(&mut self, state) -> Option<Action>` is the primary call site. Holds the RNG across calls so sampling is stable within a game.
- `profile.rs` — `BotProfile` (per-difficulty weight struct), `ProfileRegistry` (named profile map), `default_registry()`. Profiles are embedded from `assets/bots/default.toml` at compile time; a runtime override path is reserved via `BOT_PROFILES_FILE`.
- `features.rs` — feature extraction helpers (plant value scoring, resource cost estimation) shared by the auction and buy-resources decision functions.
- `strategy.rs` — `decide(state, me) -> Option<Action>` (stateless, used by the Python bridge) and `decide_with_bot(state, bot) -> Option<Action>` (profile-weighted + softmax sampling). One `decide_`* helper per phase.

### powergrid-lobby

Production multi-game server. Handles auth, room lifecycle, and in-process bots. Requires PostgreSQL (`DATABASE_URL` env var).

- `main.rs` — axum router: `/health`, `/rooms` (REST), `/ws`, `/auth/{register,login,logout}`. `AppState { manager, bot_delay, db }`.
- `ws.rs` — `ConnState { user_id, username, current_room, tx }`. Pre-auth gate: expects `ClientMessage::Authenticate { token, protocol_version }` as the first message (10s timeout); rejects mismatched `protocol_version` with `AuthError::VersionMismatch`. On success dispatches `Lobby { action }` and `Room { room, action }` messages.
- `rooms.rs` — `Room { name, game, humans, bots, creator_user_id }` with `broadcast`, `broadcast_msg`, `add_bot`, `remove_bot`, `summary`. `RoomManager` owns `RwLock<HashMap<String, Arc<Mutex<Room>>>>`.
- `lobby_handler.rs` — handles `LobbyAction` variants: `ListRooms`, `CreateRoom`, `JoinRoom`, `LeaveRoom`, `AddBot`, `RemoveBot`.
- `room_handler.rs` — handles in-game `Action`: lock room, call `apply_action`, broadcast `StateUpdate`, trigger `run_bot_pump`.
- `hint_handler.rs` — handles `ClientMessage::RoomHint`: forwards `HintPayload` to all other clients in the room via `PeerHint`.
- `driver.rs` — `run_bot_pump(room_arc, delay)`: polls `strategy::decide` for each in-process bot (up to 500 iterations), applies moves via `apply_action`, broadcasts state. Bots never touch the network.
- `auth.rs` — REST handlers for register/login/logout. 32-byte URL-safe-base64 tokens, 30-day TTL.
- `db.rs` — `Db { pool: PgPool }`. Methods: `register`, `login`, `validate_token`, `logout`. Uses Argon2 for password hashing.
- Configured via env vars: `PORT` (3000), `DATABASE_URL` (required), `BOT_DELAY_MS` (250), `MAP_FILE`, `RUST_LOG`.

### powergrid-client

egui GUI client. Supports two modes: **online** (connects to `powergrid-lobby`) and **local** (in-process session, no TCP server, no network required).

- `main.rs` — eframe app setup, Bevy-free.
- `ws.rs` — `WsChannels` resource wraps crossbeam channels + oneshot shutdown. `spawn_ws(url)` creates online channels backed by a background WS worker thread. `process_ws_events` Bevy system drains incoming `WsEvent`s each frame. Only the lobby protocol is used (`ClientMessage` envelopes). Reconnects on disconnect; shutdown propagates via `WsChannels::drop`.
- `local.rs` — `start_local_session(LocalConfig) -> (WsChannels, LocalHandle)`. Creates a `Session` in-process (human + bots join, game auto-starts). Spawns a tokio runtime thread running `local_session_driver` which routes `ClientMessage::Room` actions to `Session::apply` and runs `BotPump` after each human action. Pre-queues `Connected + Authenticated + RoomJoined + StateUpdates` before the first frame. No loopback TCP. `LocalHandle` joins the runtime thread on drop.
- `auth.rs` — `UserPreferences` (fullscreen flag, persisted via `directories`), async `do_login` / `do_register` helpers, `AuthEvent` channel type.
- `effects.rs` — animated city highlight effects (pulsing border, particles) drawn on the map during `BuildCities`.
- `peer_hints.rs` — `PeerHints` map: stores the latest `HintPayload` from each peer, consumed by the map panel to show peer cursors.
- `state.rs` — `AppState` resource. Screen enum: `MainMenu | LocalSetup | Login | Register | RoomBrowser | Game`.
- `ui/` — egui UI systems:
  - `mod.rs` — `ui_system` dispatch, `setup_egui_theme`
  - `main_menu.rs` — main menu (online vs local fork)
  - `local_setup.rs` — local game config (bot count, color)
  - `login.rs` — online login form
  - `register.rs` — account registration form
  - `room_browser.rs` — room list + create/join controls
  - `lobby.rs` — in-room lobby (player list, add/remove bots, start)
  - `top_panel.rs` — round/phase header + resource market
  - `left_panel.rs` — player info cards
  - `player_summary.rs` — end-of-round player summary widget
  - `event_log.rs` — scrollable game event log panel
  - `helpers.rs` — shared widgets (`section_header`, `neon_button`, `send`, etc.)
  - `phases/` — per-phase action UI, one file each: `auction`, `buy_resources`, `build_cities`, `bureaucracy`, `power_cities_fuel`, `discard_plant`, `discard_resource`, `game_over`
- `map_panel.rs` — renders the map with city overlays and peer hint cursors.
- `card_painter.rs` — procedurally paints power plant cards using egui primitives.
- `theme.rs` — custom egui visual theme.

Run with `cargo run -p powergrid-client` or `cargo run -p powergrid-client --features dev` for fast incremental rebuilds.

### powergrid-maptool

egui desktop tool for creating and editing map TOML files. Point it at a background image and click to place cities, set region colors, and draw connections; it writes `assets/maps/*.toml` directly.

- `main.rs` — self-contained eframe app. Usage: `cargo run -p powergrid-maptool -- <image_path> [output.toml]`. If `output.toml` already exists it is loaded automatically.

### powergrid-py

PyO3 extension module (Python 3.14, pyo3 0.28). Exposes the game engine to the Python RL environment without any network layer.

- `src/lib.rs` — `Game` pyclass with methods: `start(names, colors)`, `apply(actor, action_json)`, `state_json()`, `current_actor()`, `legal_move_info(actor)`, `bot_decide(actor, difficulty)`, `city_ids()`, `is_terminal()`, `winner()`.
- `legal_move_info` returns a JSON blob encoding every legal move for the given actor — used by the Python env to build `info["action_mask"]` without re-implementing game rules.
- Built with `maturin develop --release` from the `python/` directory (see `python/Makefile`).
- crate-type = `["cdylib"]` — produces a `.so` wheel, not a binary.

See [docs/rl-environment.md](docs/rl-environment.md) for the full Python API and training workflow.

### Protocol

Single protocol. Version negotiation is enforced at the handshake: the client must send `PROTOCOL_VERSION` (defined in `powergrid_core::actions::PROTOCOL_VERSION`); the server rejects mismatches with `AuthError::VersionMismatch`.

**Client→server** — `ClientMessage` (tagged `"type"`):

- `Authenticate { token, protocol_version }` — must be first message; 10s timeout.
- `Lobby { action: LobbyAction }` — room management (`ListRooms`, `CreateRoom`, `JoinRoom`, `LeaveRoom`, `AddBot`, `RemoveBot`).
- `Room { room, action: Action }` — in-game move scoped to a named room.
- `RoomHint { room, hint: HintPayload }` — ephemeral peer selection hint.

**Server→client** — `ServerMessage` (tagged `"type"`):

- `Authenticated`, `AuthError { error: AuthError }` — auth handshake result.
- `StateUpdate(GameStateView)` — full wire-safe state after every valid action.
- `ActionError { error: ActionError }`, `LobbyError { error: LobbyError }` — structured errors.
- `RoomList`, `RoomJoined` (includes full static map once), `RoomLeft`, `Event`, `PeerHint`.

`GameStateView` is broadcast to all clients after every valid action. It omits hidden info (deck, rng seed, map graph) and zeroes opponent money. The full `Map` is sent once in `RoomJoined`; subsequent `StateUpdate`s carry only `city_owners`.

### Map format

`assets/maps/*.toml` — list of `[[cities]]` (id, name, region) and `[[connections]]` (from, to, cost). The germany map is embedded at compile time via `powergrid_core::default_map()`, which all crates call. To use a custom map, set `MAP_FILE=/path/to/map.toml` at runtime.