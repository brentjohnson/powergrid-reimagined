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

# RL policy network inspector (load expert.bin, edit obs, watch the forward pass)
cargo run -p powergrid-netviz
cargo run -p powergrid-netviz -- path/to/policy.bin

# Docker (lobby + postgres)
docker compose up --build

# RL environment (Python — run from python/ directory)
make develop                                     # build PyO3 crate + install Python package
pytest tests/                                    # run Python tests
python scripts/train_vs_bots.py                  # MaskablePPO vs Rust bots
python scripts/train_selfplay.py                 # self-play (vs frozen snapshots of own policy)
python scripts/train_selfplay.py --curriculum-start 3   # end-game-cities curriculum (trigger 3 → rulebook, +2 per --curriculum-every steps)
python scripts/evaluate.py --model runs/vs_bots/best_model  # win-rate vs bots
python scripts/run_report.py runs/selfplay           # training-run status: checkpoints, eval history, health flags
python scripts/play_game.py --all-bots --render  # watch a rollout
python scripts/export_policy.py --model runs/vs_bots/best_model  # export weights for the Rust Expert bot
```

## Workflow

Before running a build, do "cargo fmt" "cargo check" and run clippy.  Then fix any issues before building.  
When making architectural or structural changes, update CLAUDE.md accordingly.

When making visual changes, do not attempt to launch the game to verify changes.  Tell me what to verify and I will do it manually.

If adding or removing crates, update the stubs in the Dockerfile.

## Architecture

Eight-crate Cargo workspace:

```
crates/
  powergrid-core/          # pure game logic, no I/O
  powergrid-session/       # shared Session abstraction: apply_action, broadcast, BotPump
  powergrid-bot-strategy/  # bot AI: Bot struct, BotProfile/TOML profiles, weighted strategy
  powergrid-lobby/         # production multi-game server: auth, rooms, in-process bots, PostgreSQL
  powergrid-client/        # egui GUI — online (lobby) or local play (in-process session)
  powergrid-py/            # PyO3 extension module for the Python RL environment
  powergrid-maptool/       # egui desktop tool for creating/editing map TOML files
  powergrid-netviz/        # egui desktop tool for inspecting the RL Expert policy network
assets/
  maps/usa.toml            # default map asset (49 cities), embedded at compile time via powergrid-core
  maps/germany.toml        # alternate map (42 cities), usable via MAP_FILE
  bots/default.toml        # default bot profiles (BotProfile weights), embedded at compile time
  policies/expert.bin      # RL policy weights for the Expert bot, embedded at compile time
  policies/expert.golden.json  # torch reference logits for the Rust↔torch parity test
python/                    # PettingZoo RL environment (see docs/rl-environment.md)
  src/powergrid_env/       # Python package: AECEnv, encoding, policies
  scripts/                 # training and rollout scripts
  tests/                   # Python tests (encoding, parity, reseeding, random play)
  TRAINING.md              # step-by-step training runbook (start/resume/monitor)
  pyproject.toml           # hatchling build; maturin builds the Rust extension separately
  Makefile                 # make develop = build Rust + install Python
```

Dependency graph (Rust): core ← bot-strategy ← {session, powergrid-py} ← {lobby, client}.
`powergrid-client` also depends on `powergrid-bot-strategy` directly (for the
bot valuation popup — see below), as does `powergrid-netviz` (for `policy`/`encoding`).

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

- `bot.rs` — `Bot { id, name, color, profile, rng, policy }`: stateful bot with a seeded `SmallRng`. `decide(&mut self, state) -> Option<Action>` is the primary call site. Holds the RNG across calls so sampling is stable within a game. When `policy` is set (Expert difficulty, via `with_policy`), `decide` plays the RL policy and only falls back to the heuristic if it's unusable.
- `profile.rs` — `BotProfile` (per-difficulty weight struct), `ProfileRegistry` (named profile map: easy/normal/hard/expert), `default_registry()`. Profiles are embedded from `assets/bots/default.toml` at compile time; a runtime override path is reserved via `BOT_PROFILES_FILE`. The `expert` profile (a hard clone) is only the fallback/valuation profile — Expert play is driven by the RL policy.
- `features.rs` — feature extraction helpers (plant value scoring, resource cost estimation) shared by the auction and buy-resources decision functions.
- `strategy.rs` — `decide(state, me) -> Option<Action>` (stateless, used by the Python bridge) and `decide_with_bot(state, bot) -> Option<Action>` (profile-weighted + softmax sampling). One `decide_`* helper per phase. Anti-stall guarantees (the game only ends when someone builds `end_game_cities`): `decide_build_cities` overbuilds — spends *surplus* cash (above fuel + city reserves) on cities beyond powering headroom, capped at `end_game_cities` — and the auction thresholds (`min_open_score`, `upgrade_margin`) are scaled by `late_game_urgency` so bots keep buying plants as the game closes (regression test: `tests/heuristic_termination.rs`). Also `decide_rl(state, bot) -> RlDecision`: the Expert path — strict `current_actor_id` turn gate, obs/mask encoding, MLP forward, stochastic masked-softmax sampling (never argmax: greedy play can stall).
- `encoding.rs` — the RL observation/action encoding (moved here from powergrid-py so the Expert bot can use it): `CITY_IDS`/`REGION_NAMES`/`OBS_SIZE` (454)/`N_ACTIONS` (143) constants, `build_observation`, `build_action_mask`, `action_id_to_action`, `compute_legal_move_info`, `current_actor_id`, `map_matches_default`. Compiled against the **default (USA) map**; mirrors `python/src/powergrid_env/constants.py` (parity tests in `python/tests/test_native_bridge.py` catch drift).
- `policy.rs` — native inference for the Expert policy: `MlpPolicy` (454 → H → tanh → H → tanh → 143 logits, where the hidden width H is read from the policy file header — the trainer's `--net-width` sets it for fresh runs (default 128; the embedded `expert.bin` is still 64-wide); the architecture is fixed at two equal-width hidden layers — plain-loop forward pass, no ML deps), `sample_masked`, `default_policy()` (parses `assets/policies/expert.bin`, embedded at compile time, once via `OnceLock`; `RL_POLICY_FILE` env var overrides). Weights are produced by `python/scripts/export_policy.py` from an sb3 MaskablePPO checkpoint; a golden-logits test pins Rust output to torch. On a non-default map or missing/corrupt weights, Expert bots degrade to the hard-style heuristic (warn log, game still starts).

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
  - `mod.rs` — `ui_system` dispatch, `setup_egui_theme`. Also hosts the bot
    valuation popup (`valuation_window`, local play only): press **`b`** to
    show a live table — rows are market plants, columns are bots — of each
    bot's `evaluate_plant(...).total` (Elektro `PlantValue`, per LOGIC.md;
    `MaxBid = PlantValue`), with the six-term `PlantValuation` breakdown on
    cell hover. Bot identity (color → `BotDifficulty`) is reconstructed from
    the deterministic mapping `local.rs::start_local_session` builds, since
    the wire protocol never reveals which seats are bots.
  - `main_menu.rs` — main menu (online vs local fork)
  - `local_setup.rs` — local game config (bot count, color)
  - `login.rs` — online login form
  - `register.rs` — account registration form
  - `room_browser.rs` — room list + create/join controls
  - `lobby.rs` — in-room lobby (player list, add/remove bots, start)
  - `determine_order.rs` — DETERMINE ORDER floating overlay (top-left corner):
    ROUND/STEP display
  - `plant_market.rs` — AUCTION PLANTS floating overlay (top-right corner):
    actual/future plant cards, turn-dots, click-to-nominate; captures its rect
    for `state.phase_column_rects[0]` and `state.plant_market_bottom` so the
    auction action panel and info panel anchor below it
  - `overlays.rs` — remaining floating overlays: resource market (bottom-right),
    plus replenish/city-graph/payout table helpers used by the info panel
  - `left_panel.rs` — player info cards
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

### powergrid-netviz

egui desktop tool for interactively inspecting the RL Expert policy network (`powergrid_bot_strategy::policy::MlpPolicy`, 454 → H → tanh → H → tanh → 143, hidden width H read from the policy file header — 64 by default). Loads a `PGRLPOL1` policy file (CLI arg) or the embedded `expert.bin` (no arg), lets you edit the 454-dim observation with labeled sliders grouped by section, and renders the live forward pass as a four-column node graph (input cells, both hidden layers, output logits). Clicking a node traces its full weighted path through every layer (e.g. an output node back through hidden2 → hidden1 → input, an input cell forward through hidden1 → hidden2 → output) — edges are colored by sign and scaled by magnitude, normalized independently per weight matrix (input→h1 / h1→h2 / h2→out) so one large weight doesn't wash out the rest; hovering shows exact values. The right panel lists all 143 actions sorted by logit with softmax probabilities.

- `main.rs` — self-contained eframe app (`NetViz`). Usage: `cargo run -p powergrid-netviz [-- path/to/policy.bin]`.
- `obs_layout.rs` — labeled sections of the 454-dim observation vector, mirroring `encoding::build_observation`'s numbered layout (money, resources, plants, city one-hots, market, phase scalars, etc.).
- `action_labels.rs` — human-readable names for all 143 actions, derived from `encoding`'s action-base-index constants and `CITY_IDS`.
- Relies on `MlpPolicy::forward_trace` (returns every pre/post-tanh intermediate) and the `l1`/`l2`/`out` weight accessors added to `policy.rs` for this tool.
- In active-game mode, `sync_from_game` captures the real observation as `baseline_obs` whenever it's the inspected seat's turn. The "Show Δ from real observation" checkbox (network panel, enabled once a baseline exists) recolors input cells, hidden/output nodes, and edges by the *change* relative to that baseline — i.e. the impact of hand-edited sliders on the forward pass — instead of absolute values.
- `game.rs` — `GameDriver`/`GameConfig`: drives a real local game (pure-sync, mirroring `powergrid-py`'s `Game::start`/`drive_bots` — no tokio/`Session`) with one inspected seat (the host) plus heuristic bot opponents of a configurable difficulty, player count, seed, and optional end-game-cities override. The left panel's "New game" button starts it; once it's the inspected seat's turn, its real observation and action mask load into the sliders and into the output panel's legality/legal-softmax columns. Sliders stay editable afterward for hand-tweaking. "Apply policy move" samples a masked action from the policy's logits over the *current* (possibly hand-tweaked) observation and plays it; "Apply selected action" plays whichever output-list action is selected, if legal. Either advances the game (bots play their turns) and reloads the next real observation/mask.

### powergrid-py

PyO3 extension module (Python 3.14, pyo3 0.28). Exposes the game engine to the Python RL environment without any network layer.

- `src/lib.rs` — `Game` pyclass with methods: `start(names, colors)`, `apply(actor, action_json)`, `state_json(viewer=None)` (viewer's own money included when given), `current_actor()`, `legal_move_info(actor)`, `bot_decide(actor, difficulty)`, `city_ids()`, `is_terminal()`, `winner()`, `set_end_game_cities(n)` (post-`start()` override of the end-game city trigger; the trigger is part of the observation, so policies can condition on it — used by the training curriculum).
- Fast native methods (no JSON, numpy in/out): `observation(actor)`, `action_mask(actor)`, `apply_action_id(actor, id)`, `step_vs_bots(learner, id, difficulty)` + `advance_bots(learner, difficulty)` (fused step: opponents driven inside Rust; also returns the learner's powered-cities count for reward shaping), `load_opponent_policy(bytes)` (loads a frozen PGRLPOL1 policy snapshot; difficulty `"policy"` then drives opponents with it via persistent per-seat bots — frozen-opponent self-play, used by `train_selfplay.py`).
- `legal_move_info` returns a JSON blob encoding every legal move for the given actor — used by the Python env to build `info["action_mask"]` without re-implementing game rules.
- The obs/mask/action-id encoding lives in `powergrid_bot_strategy::encoding` (shared with the Expert bot); this crate only adds the PyO3/numpy wrappers. The encoding targets the **default (USA) map**; changing the default map invalidates trained checkpoints.
- `bot_decide`/`advance_bots`/`step_vs_bots` accept `"easy" | "normal" | "hard" | "expert"` (heuristic; `"expert"` plays as the hard-style heuristic here — the embedded Expert RL policy is never attached) or `"policy"` (the snapshot loaded via `load_opponent_policy`; errors if none loaded). Native Expert play exists only via `Session::add_bot` (client/lobby).
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

`assets/maps/*.toml` — list of `[[cities]]` (id, name, region) and `[[connections]]` (from, to, cost). The usa map is embedded at compile time via `powergrid_core::default_map()`, which all crates call. To use a custom map, set `MAP_FILE=/path/to/map.toml` at runtime (note: the RL env's encoding is compiled against the default map and does not follow `MAP_FILE`).