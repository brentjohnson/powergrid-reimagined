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
./scripts/sweep_selfplay.sh                      # 8 parallel self-play variants from one behavior clone (--list/--status/--compare/--h2h/--stop)
python scripts/train_selfplay.py --curriculum-start 3   # end-game-cities curriculum (trigger 3 → rulebook, +2 per --curriculum-every steps)
python scripts/orchestrate.py                    # forever-training orchestrator: train → eval → adapt loop; state + reasoning journal in runs/orch/ (see TRAINING.md §8)
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

Do not create git branches.  Make all commits on main.  Do not push or tag.

## Architecture

Nine-crate Cargo workspace:

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
  powergrid-evolve/        # offline CMA-ES tuner for the heuristic BotProfile weights (training tool)
assets/
  maps/usa.toml            # default map asset (49 cities), embedded at compile time via powergrid-core
  maps/germany.toml        # alternate map (42 cities), usable via MAP_FILE
  bots/default.toml        # default bot profiles (BotProfile weights), embedded at compile time
  policies/expert.bin      # RL policy weights for the Expert bot, embedded at compile time
  policies/expert.golden.json  # torch reference logits for the Rust↔torch parity test
python/                    # PettingZoo RL environment (see docs/rl-environment.md)
  src/powergrid_env/       # Python package: AECEnv, encoding, policies, stats,
                           #   callbacks (league snapshots, shaping anneal, curriculum,
                           #   persistent best-eval), run_metrics (TensorBoard access)
  scripts/                 # training and rollout scripts + orchestrate.py (forever loop)
  tests/                   # Python tests (encoding, parity, reseeding, random play)
  TRAINING.md              # step-by-step training runbook (start/resume/monitor)
  pyproject.toml           # hatchling build; maturin builds the Rust extension separately
  Makefile                 # make develop = build Rust + install Python
alphazero/                 # AlphaZero (MCTS self-play) training — see alphazero/README.md
```

Dependency graph (Rust): core ← bot-strategy ← {session, powergrid-py} ← {lobby, client}.
`powergrid-client` also depends on `powergrid-bot-strategy` directly (for the
bot valuation popup — see below), as does `powergrid-netviz` (for `policy`/`encoding`).

`powergrid-py` depends only on `powergrid-core` and `powergrid-bot-strategy` — no server, lobby, or async runtime.

### powergrid-core

All game state and rules. The key entry point is `rules::apply_action(state, player_id, action) -> Result<(), ActionError>`. It's pure — no I/O — and fully unit-testable.

- `types.rs` — `Player`, `PowerPlant`, `ResourceMarket`, `PlantMarket`, `Phase`, `PlayerColor`, `PlayerId` (Uuid alias), etc. `Player.stats: PlayerStats` accumulates cumulative per-game spend/activity (plants/resources/cities bought + Elektro spent on each), incremented at the purchase points in `rules`; treated as hidden info (zeroed for opponents in `GameState::view_for` alongside `money`).
- `limits.rs` — validation constants: `MAX_PLAYER_NAME`, `MAX_ROOM_NAME`, `MIN_USERNAME`, `MAX_USERNAME`, `MAX_EMAIL`, `MIN_PASSWORD`, `MAX_PASSWORD`, etc.
- `state.rs` — `GameState` struct (all game data including the map); `GameStateView` is the wire-safe projection.
- `actions/` — all wire types, split across three files:
  - `game.rs` — `Action` (game moves), `ActionError`
  - `protocol.rs` — `ServerMessage`, `ClientMessage`, `LobbyAction`, `RoomSummary`, `AuthError`, `LobbyError`
  - `hints.rs` — `HintPayload`
  - `mod.rs` — re-exports + `PROTOCOL_VERSION` constant
- `map.rs` — `Map` (runtime graph) + `MapData` (TOML-deserializable). Dijkstra routing in `Map::connection_cost_to`.
- `rules.rs` — `apply_action` dispatcher + one `handle_*` function per phase. Also `build_plant_deck()` and `finish_ranks(state) -> Vec<(PlayerId, position)>` (full 1-based final standings by the official tiebreak — powered, then money, then cities; position 1 == `determine_winner`'s winner, ties resolved identically. Used by `powergrid-evolve` fitness and, later, RL value targets / search leaf eval).

**Phase flow:** `Lobby → Auction → BuyResources → BuildCities → Bureaucracy → [next round or GameOver]`

### powergrid-session

Shared game session abstraction used by both lobby and client.

- `lib.rs` — `Session { game, subscribers, bots }`. Methods: `apply(actor, action)` (calls `apply_action`, broadcasts `StateUpdate`), `add_subscriber(Subscriber)`, `add_bot(name, color, difficulty) -> Result<PlayerId, ActionError>`, `remove_bot(id)`, `broadcast(msg)`.
- `Subscriber` — two variants: `Mpsc(UnboundedSender<String>)` serializes to JSON (WS use); `Local(crossbeam::Sender<ServerMessage>)` sends typed messages (in-process use).
- `run_bot_pump(Arc<Mutex<Session>>, delay)` — drives all in-process bots until none has a move or 500-iteration cap is hit; releases lock between turns.
- `report.rs` — `build_report(&Session) -> GameReport` (+ `SeatReport`/`PlantReport`, all `Serialize + Deserialize`): the shared end-of-game standings extraction (finish ranks, per-seat economy/plants, bot-vs-human). Both the lobby (in-memory rooms) and the client (local play, POSTed to the server) build the identical shape from it — see the `/games/local` endpoint and `powergrid-client/local.rs`.
- `MAX_PLAYERS: u8 = 6` — single workspace-level constant.

### powergrid-bot-strategy

Pure strategy + AI lib. No I/O, no tokio. Depended on by session, lobby, and client.

- `bot.rs` — `Bot { id, name, color, difficulty, profile, rng, policy }`: stateful bot with a seeded `SmallRng`. `difficulty` is reporting-only (decision strength lives in `profile`/`policy`); `Session::add_bot` tags it via `with_difficulty` so the lobby can record which kind of bot occupied each seat. `decide(&mut self, state) -> Option<Action>` is the primary call site. Holds the RNG across calls so sampling is stable within a game. When `policy` is set (Expert difficulty, via `with_policy`), `decide` plays the RL policy and only falls back to the heuristic if it's unusable.
- `profile.rs` — `BotProfile` (per-difficulty weight struct; `Serialize + Deserialize`), `ProfileRegistry` (named profile map: easy/normal/hard/expert), `default_registry()`, `embedded_registry()`. Profiles are embedded from `assets/bots/default.toml` at compile time. `default_registry()` returns a process-cached `&'static ProfileRegistry`, resolved once at first use: if `BOT_PROFILES_FILE` names a valid TOML it wins, else the embedded defaults (bad path/TOML logs and falls back). Caching also removes the per-decision `toml::from_str` the PyO3 bridge used to pay. `embedded_registry()` always returns the pristine compiled-in profiles, ignoring the env var (used by `powergrid-evolve` for its init mean / opponent yardstick). The `expert` profile (a hard clone) is only the fallback/valuation profile — Expert play is driven by the RL policy.
- `features.rs` — feature extraction helpers (plant value scoring, resource cost estimation) shared by the auction and buy-resources decision functions.
- `strategy.rs` — `decide(state, me) -> Option<Action>` (stateless, used by the Python bridge) and `decide_with_bot(state, bot) -> Option<Action>` (profile-weighted + softmax sampling). One `decide_`* helper per phase. Anti-stall guarantees (the game only ends when someone builds `end_game_cities`): `decide_build_cities` overbuilds — spends *surplus* cash (above fuel + city reserves) on cities beyond powering headroom, capped at `end_game_cities` — and the auction thresholds (`min_open_score`, `upgrade_margin`) are scaled by `late_game_urgency` so bots keep buying plants as the game closes (regression test: `tests/heuristic_termination.rs`). Also `decide_rl(state, bot) -> RlDecision`: the Expert path — strict `current_actor_id` turn gate, obs/mask encoding, MLP forward, stochastic masked-softmax sampling (never argmax: greedy play can stall).
- `encoding.rs` — the RL observation/action encoding (moved here from powergrid-py so the Expert bot can use it): `CITY_IDS`/`REGION_NAMES`/`OBS_SIZE` (582)/`N_ACTIONS` (26 — re-exported from `macro_actions::N_MACROS`) constants, `build_observation`, `build_action_mask`, `action_id_to_action`, `compute_legal_move_info`, `current_actor_id`, `map_matches_default`. Compiled against the **default (USA) map**; mirrors `python/src/powergrid_env/constants.py` (parity tests in `python/tests/test_native_bridge.py` catch drift). **2026-07-08:** obs grew 454→507→582 with three features the heuristic bot relies on but the net couldn't see — per-city connection cost from the actor's network (`Map::connection_costs_from`, the Dijkstra routing that drives build decisions), opponent per-resource fuel demand (market contention), and per-opponent plant detail (5 opp × 3 plants × number/kind/cost/cities/cap — the highest number is the turn-order tiebreaker; kinds/costs feed denial/fuel reasoning). Any change to the obs layout invalidates all trained checkpoints. **The action space is the macro menu, not primitives** — see `macro_actions.rs` below.
- `macro_actions.rs` — the **macro action space** the RL policy plays (`N_MACROS = 26`). Instead of the ~600 primitive micro-decisions per game that capped every earlier learner (compounding error over per-unit `BuildCity`/`BuyResources` sequences), the policy picks **one complete phase-plan per turn** (~50 decisions/game) which expands to a short primitive sequence the engine already accepts as a whole-turn batch (`BuildCities`, `BuyResourceBatch`). Layout: nominate market slot 0–5 (0–5), auction pass (6), auction raise +1 (7), **build the n cheapest cities for n = 0..6** (8–14), **buy: a bitmask over plant slots** (15–22) — choose which plants you intend to fire and top those up, counting current stock — discard plant slot 0–2 (23–25). The two menus are shaped differently on purpose. **Cities are interchangeable**, so build is an absolute *count*. **Fuel is spent in indivisible plant-sized chunks**, so the decision is *which plants will fire* and buy is a subset of the rack. Declaring the subset is also what makes the purchase well defined on a shared pool — "top plant A up" is ambiguous when B also burns coal, but "these plants fire" fixes the requirement as the sum over the declared set (`strategy::plan_essential_buys` with its walk restricted to the selection). Because it tops up rather than adding, the full-rack mask reproduces the champion's buy bit-for-bit, so **no buy default is needed either**. Slot `i` is the `i`-th plant by number — `rules.rs` re-sorts `player.plants` on acquisition, so this matches the observation's self-plant order and `DISCARD_PLANT`'s slots. Dedup canonicalises to the smallest equivalent mask, which makes the teacher's label name exactly the plants that needed fuel — buy is the one phase where the rebuild turned a constant imitation label into a varied one. Stockpiling is deliberately unrepresentable: CMA-ES had `buy.stockpile_rounds` over [1.0, 5.0] and the champion sits at the 1.0 floor. Every phase is exactly one decision per turn. **Powering has no macro**: `Bureaucracy` is auto-resolved with the heuristic alongside the fuel split and resource discard — the teacher fired the optimal subset in 100% of measured decisions and the only alternative offered ("power nothing") was legal everywhere and correct nowhere, costing ~9 of a seat's ~52 decisions per game for nothing. Key invariants: only the *learner* plays macros (heuristic opponents keep `strategy::decide_with_bot`); there are **no `*_DEFAULT` escape hatches**: both the build count ladder and the buy subset reproduce the champion's action bit-exactly on their own (Gate 0 test), so the dedicated fallbacks were removed rather than left permanently masked. `teacher_macro_id` returns `None` if no id reproduces the heuristic, tripping Gate 0 in test rather than silently papering over it. Fuel splits and resource discards are auto-resolved heuristically (`resolve_auto_phases`) so they never consume a policy decision. `legal_macros` masks by trial application on a clone **and dedups** — a macro whose expansion equals a lower-id macro's is illegal, so e.g. `BUILD_3` collapses onto `BUILD_2` when only 2 are affordable. Dedup also canonicalises the buy mask to the smallest equivalent subset, so a given intent always maps to the same id. `teacher_macro_id` gives the imitation/DAgger label and always returns the id that survives dedup. Both ladders are limited **only by cash** — no reserve is withheld, because the count *is* the policy's decision and a reserve would silently turn `n` into `m` (the earlier build menu capped the city budget at `money - fuel_reserve`, which zeroed it in ~85% of real decisions and deduped the whole alternative-build menu away; the earlier buy menu's stockpile variants sat behind a 140-Elektro reserve and could differ from the default in only 2.1% of decisions).
- `search.rs` — Phase 3 play-time search: PUCT MCTS over macros, guided by the policy (prior = policy softmax) with leaf values from the exported PPO value net (`policy::ValueNet`, PGRLVAL1) and exact `rules::finish_ranks` at terminals. A faithful port of `alphazero/mcts.py` adapted to macros; tractable precisely because a game is ~50 macro decisions deep. **Determinized** — the search only sees a reshuffled copy of the unseen plant deck, so it can't exploit true deck order against a human; several determinized worlds are searched and their root visit counts summed.
- `policy.rs` — native inference for the Expert policy: `MlpPolicy` (582 → H → tanh → H → tanh → 26 logits, where the input width matches `OBS_SIZE`, the output width `N_ACTIONS`, and the hidden width H is read from the policy file header — the trainer's `--net-width` sets it for fresh runs (default 128; the embedded `expert.bin` is 128-wide); the architecture is fixed at two equal-width hidden layers — plain-loop forward pass, no ML deps), `sample_masked`, `default_policy()` (parses `assets/policies/expert.bin`, embedded at compile time, once via `OnceLock`; `RL_POLICY_FILE` env var overrides). Weights are produced by `python/scripts/export_policy.py` from an sb3 MaskablePPO checkpoint; a golden-logits test pins Rust output to torch. On a non-default map or missing/corrupt weights, Expert bots degrade to the hard-style heuristic (warn log, game still starts). **Format epoch:** the magic is `PGRLPOL6`. `PGRLPOL2` was minted on 2026-07-25 when the macro ids were first reorganised — `N_ACTIONS` happened to land back on 26 across that change while the id *meanings* moved, so a dims-only check would have loaded a stale policy and silently played a scrambled action map — and `PGRLPOL6` on 2026-07-26 across the buy-menu and default-macro changes. Bump the magic whenever macro ids are renumbered, even at unchanged `N_ACTIONS`. **2026-08-03:** the embedded `expert.bin`/`expert.golden.json` are a `PGRLPOL6` export of the `p2-finish` wave-6 sweep winner (net-width 128; 68.5% vs 3x hard over 400 games in torch, 74% over the noisy 50-game native `expert_vs_hard_win_rate` harness), replacing the wave-5 `z3-batch-decay` export (66.5% torch, 68% native). `policy::tests::embedded_policy_matches_torch_golden_logits` and `tests/expert_bot.rs::expert_bot_plays_policy_action_on_its_turn` run by default. The **value** net (`expert.value.bin`/`expert.value.golden.json`, PGRLVAL1, scalar output) is re-exported from the same checkpoint whenever the policy is, so search leaf values match the policy; its golden test also runs by default.

### powergrid-lobby

Production multi-game server. Handles auth, room lifecycle, and in-process bots. Requires PostgreSQL (`DATABASE_URL` env var).

- `main.rs` — axum router: `/health`, `/rooms` (REST), `/games/local` (POST, local-play metrics ingest), `/ws`, `/auth/{register,login,logout}`, and (when `ADMIN_TOKEN` is set) the `/admin` interface merged in. `AppState { manager, bot_delay, db }`.
- `game_ingest.rs` — `POST /games/local`: accepts a `powergrid_session::GameReport` from a client running a game in-process (no server-side room) and records it via `Db::record_game`. Attribution is server-decided, never trusted from the payload: the (single) human seat is credited to the account behind the bearer token, or to `anonymous` (NULL `user_id`, name "anonymous", so it still counts toward human-vs-bot/finish stats but gets no leaderboard/career profile) when there's no token or it fails to validate. Bots are forced to NULL `user_id`; `room_name` is fixed to `"local"`.
- `ws.rs` — `ConnState { user_id, username, current_room, tx }`. Pre-auth gate: expects `ClientMessage::Authenticate { token, protocol_version }` as the first message (10s timeout); rejects mismatched `protocol_version` with `AuthError::VersionMismatch`. On success dispatches `Lobby { action }` and `Room { room, action }` messages.
- `rooms.rs` — `Room { name, session, humans, creator_user_id, started_at, results_recorded }` with `broadcast`, `broadcast_msg`, `add_bot`, `remove_bot`, `summary`, `is_game_over`. `started_at`/`results_recorded` support the game-over persistence hook. `RoomManager` owns `RwLock<HashMap<String, Arc<Mutex<Room>>>>`.
- `lobby_handler.rs` — handles `LobbyAction` variants: `ListRooms`, `CreateRoom`, `JoinRoom`, `LeaveRoom`, `AddBot`, `RemoveBot`.
- `room_handler.rs` — handles in-game `Action`: lock room, call `apply_action`, broadcast `StateUpdate`, trigger `run_bot_pump`. Records `started_at` on the Lobby→started transition, and after the pump calls `maybe_record_result` → `build_game_record` (from `finish_ranks`) → `Db::record_game`. The `results_recorded` flag (set under the room mutex) makes this fire exactly once per game, whether a human or a subsequent bot move ended it — `handle_room_action` is the sole path that can reach `Phase::GameOver`.
- `hint_handler.rs` — handles `ClientMessage::RoomHint`: forwards `HintPayload` to all other clients in the room via `PeerHint`.
- `driver.rs` — `run_bot_pump(room_arc, delay)`: polls `strategy::decide` for each in-process bot (up to 500 iterations), applies moves via `apply_action`, broadcasts state. Bots never touch the network.
- `auth.rs` — REST handlers for register/login/logout. 32-byte URL-safe-base64 tokens, 30-day TTL.
- `db.rs` — `Db { pool: PgPool }`. Methods: `register`, `login` (both set `users.last_login`), `validate_token`, `logout`, `record_game` (transactional: one `games` row + per-seat `game_players` + per-plant `game_player_plants` inserts), `admin_reset_password` (hash an admin-supplied password, rehash, revoke sessions). Uses Argon2 for password hashing. `GameRecord.seats` is `Vec<powergrid_session::SeatReport>` (the shared type, so the client can build the same record for local play); a `SeatReport` carries `bot_difficulty` (None for humans), `turn_order` (1-based seat/join index), the six cumulative economy totals from `Player::stats` (plants/resources/cities bought + spend on each), and `plant_details: Vec<PlantReport>` (every plant a seat held at game end: number/kind/capacity/resource_cost).
- `admin_queries.rs` — read-only queries backing the admin API (second `impl Db` block): `admin_list_players`, `admin_player`, `admin_player_games`, `admin_player_position_counts`, `admin_player_stats` (career averages), `admin_player_favorite_plants`, `admin_recent_games`, `admin_game`/`admin_game_seats`/`admin_game_plants` (game-detail view), `admin_metrics`. `Metrics` aggregates: totals (users/games/seats), avg rounds/players/length, games-per-day, human-vs-bot wins, winner averages, and breakdowns by bot difficulty, per-plant + per-fuel-kind effectiveness (usage + win rate + avg finish, joined through `game_player_plants`), player color, turn order, rounds histogram, table size, and per-finish-position averages (`FinishPositionAvg`: mean end-of-game + economy figures for 1st/2nd/3rd/… finishers), plus the leaderboard. `admin_player_stats` and the game-detail seats also expose the economy totals. Row structs derive `FromRow + Serialize`.
- `admin.rs` — the `/admin` router (see [docs/admin-console.md](docs/admin-console.md) for the operator-facing guide). Static UI (`admin.html`/`admin.css`/`admin.js` embedded via `include_str!` from `static/`) served token-free; `/admin/api/*` JSON endpoints gated by a constant-time `ADMIN_TOKEN` bearer check (`middleware::from_fn_with_state`). Endpoints: `GET /api/players`, `GET /api/players/:id` (detail + stats + favorite plants), `POST /api/players/:id/reset-password`, `GET /api/metrics`, `GET /api/games`, `GET /api/games/:id` (full standings + per-seat plants). Only mounted when `ADMIN_TOKEN` is set (else the routes 404). The UI (vanilla JS, no build step, dark theme) has players/player-detail/metrics/games/game-detail views with CSS bar charts, sortable plant-effectiveness table, fuel-kind/color/difficulty/turn-order and per-finish-position-averages breakdowns, per-player career economy averages, per-seat economy on game detail, and a reset-password modal (admin types or generates the new password, then saves); it stores the token in `sessionStorage`.
- `migrations/` — sqlx embedded migrations run at startup. `0001_init.sql` (`users`, `sessions`); `0002_admin.sql` adds `users.last_login` and the `games` / `game_players` result tables (`game_players.user_id` is the human's `users.id`, NULL for bots; `finish_position` 1-based); `0003_metrics.sql` adds `game_players.bot_difficulty`/`turn_order` and the `game_player_plants` table (one row per plant held at game end, FK'd to `(game_id, finish_position)`); `0004_player_economy.sql` adds the six nullable per-seat economy columns (plants/resources/cities bought + spend on each; NULL for pre-migration games so they drop out of AVG()s).
- Configured via env vars: `PORT` (3000), `DATABASE_URL` (required), `BOT_DELAY_MS` (250), `ADMIN_TOKEN` (admin disabled if unset), `MAP_FILE`, `RUST_LOG`.

### powergrid-client

egui GUI client. Supports two modes: **online** (connects to `powergrid-lobby`) and **local** (in-process session, no TCP server, no network required).

- `main.rs` — eframe app setup, Bevy-free.
- `ws.rs` — `WsChannels` resource wraps crossbeam channels + oneshot shutdown. `spawn_ws(url)` creates online channels backed by a background WS worker thread. `process_ws_events` Bevy system drains incoming `WsEvent`s each frame. Only the lobby protocol is used (`ClientMessage` envelopes). Reconnects on disconnect; shutdown propagates via `WsChannels::drop`. Three consecutive connections that die right after `Authenticate` with no server reply (a pre-flush-fix server resetting a rejected handshake) are treated as a rejected session: the saved token is cleared, the reconnect loop stops, and the Login screen shows why.
- `local.rs` — `start_local_session(LocalConfig, MetricsConfig) -> (WsChannels, LocalHandle)`. Creates a `Session` in-process (human + bots join, game auto-starts). Spawns a tokio runtime thread running `local_session_driver` which routes `ClientMessage::Room` actions to `Session::apply` and runs `BotPump` after each human action. Pre-queues `Connected + Authenticated + RoomJoined + StateUpdates` before the first frame. No loopback TCP. `LocalHandle` joins the runtime thread on drop. When the game reaches `GameOver`, `maybe_submit_metrics` builds a `GameReport` (`build_report`) once and fire-and-forgets a blocking POST to `http://{server}:{port}/games/local` (`MetricsConfig`: the client's target `server`/`port` + saved `token`); best-effort, failures are logged only. `main.rs` fills `MetricsConfig` from `AppState.server_name`/`port`/`auth_token`.
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

egui desktop tool for interactively inspecting the RL Expert policy network (`powergrid_bot_strategy::policy::MlpPolicy`, 582 → H → tanh → H → tanh → 26, hidden width H read from the policy file header — 64 by default). Loads a `PGRLPOL6` policy file (CLI arg) or the embedded `expert.bin` (no arg), lets you edit the 582-dim observation with labeled sliders grouped by section, and renders the live forward pass as a four-column node graph (input cells, both hidden layers, output logits). Clicking a node traces its full weighted path through every layer (e.g. an output node back through hidden2 → hidden1 → input, an input cell forward through hidden1 → hidden2 → output) — edges are colored by sign and scaled by magnitude, normalized independently per weight matrix (input→h1 / h1→h2 / h2→out) so one large weight doesn't wash out the rest; hovering shows exact values. The right panel lists all 26 macro actions sorted by logit with softmax probabilities.

- `main.rs` — self-contained eframe app (`NetViz`). Usage: `cargo run -p powergrid-netviz [-- path/to/policy.bin]`.
- `obs_layout.rs` — labeled sections of the 582-dim observation vector, mirroring `encoding::build_observation`'s numbered layout (money, resources, plants, city one-hots, market, phase scalars, per-city connection cost, opponent fuel demand, opponent plants, etc.).
- `action_labels.rs` — human-readable names for all 26 macro actions, derived from `macro_actions`' id constants.
- Relies on `MlpPolicy::forward_trace` (returns every pre/post-tanh intermediate) and the `l1`/`l2`/`out` weight accessors added to `policy.rs` for this tool.
- In active-game mode, `sync_from_game` captures the real observation as `baseline_obs` whenever it's the inspected seat's turn. The "Show Δ from real observation" checkbox (network panel, enabled once a baseline exists) recolors input cells, hidden/output nodes, and edges by the *change* relative to that baseline — i.e. the impact of hand-edited sliders on the forward pass — instead of absolute values.
- `game.rs` — `GameDriver`/`GameConfig`: drives a real local game (pure-sync, mirroring `powergrid-py`'s `Game::start`/`drive_bots` — no tokio/`Session`) with one inspected seat (the host) plus heuristic bot opponents of a configurable difficulty, player count, seed, and optional end-game-cities override. The left panel's "New game" button starts it; once it's the inspected seat's turn, its real observation and action mask load into the sliders and into the output panel's legality/legal-softmax columns. Sliders stay editable afterward for hand-tweaking. "Apply policy move" samples a masked action from the policy's logits over the *current* (possibly hand-tweaked) observation and plays it; "Apply selected action" plays whichever output-list action is selected, if legal. Either advances the game (bots play their turns) and reloads the next real observation/mask.

### powergrid-evolve

Offline **CMA-ES tuner** for the heuristic bot's `BotProfile` weights — Phase 1 of the "beat humans" plan (see `RL-TRAINING-JOURNAL.md`). A training tool (binary), not shipped in the server/client image (stubbed in the Dockerfile). It plays thousands of headless heuristic games per generation and optimizes the 14 strategy weights to maximize the candidate seat's finish position. Full runbook in `crates/powergrid-evolve/README.md`.

- Fitness is **paired**: all candidates in a generation share one seed block (common random numbers), seat-rotated to remove position bias, with all bots' noise silenced (`temperature = 0`, `jitter = 0`) so a fixed seed replays bit-identically. Gen-0 mean = the shipped `hard` profile, so it reproduces the known ~33% baseline and anchors the run.
- `genome.rs` (14-weight ⇄ normalized-vector map, `x = 0` is `hard`), `games.rs` (headless deterministic game via `Bot::decide` → `finish_ranks`, parallel over threads), `cmaes.rs` (self-contained `(μ/μ_w, λ)`-CMA-ES with a hand-rolled Jacobi eigensolver — no `nalgebra`/BLAS), `main.rs` (CLI, generation loop, outputs). Outputs to `--out-dir`: `history.csv`, `best.toml` (a full `ProfileRegistry`, deployable via `BOT_PROFILES_FILE`), resumable `checkpoint.json`.
- **Determinism dependency:** truly-paired eval requires the engine to be deterministic given a seed at `jitter = 0`. This surfaced (and fixed) a latent nondeterminism — `strategy.rs::decide_build_cities` sorted candidate cities by cost with no tiebreak, so equal-cost cities resolved by `Map::cities` HashMap iteration order (randomized per instance). The sort now has a city-id tiebreak. Any new heuristic decision that iterates a `HashMap` and picks by a non-total order must add a similar deterministic tiebreak.

### powergrid-py

PyO3 extension module (Python 3.14, pyo3 0.28). Exposes the game engine to the Python RL environment without any network layer.

- `src/lib.rs` — `Game` pyclass with methods: `start(names, colors)`, `apply(actor, action_json)`, `state_json(viewer=None)` (viewer's own money included when given), `current_actor()`, `legal_move_info(actor)`, `bot_decide(actor, difficulty)`, `bot_decide_id(actor, difficulty)` (the champion heuristic's move as a **macro id** — `macro_actions::teacher_macro_id`, the imitation/DAgger label), `city_ids()`, `is_terminal()`, `winner()`, `set_end_game_cities(n)` (post-`start()` override of the end-game city trigger; the trigger is part of the observation, so policies can condition on it — used by the training curriculum), `copy()` (deep-clones the `GameState`, including the seeded RNG, into a fresh independent `Game` — used by `alphazero/`'s MCTS to fork search nodes without mutating the original).
- Fast native methods (no JSON, numpy in/out): `observation(actor)`, `action_mask(actor)`, `apply_action_id(actor, id)`, `step_vs_bots(learner, id, difficulty)` + `advance_bots(learner, difficulty)` (fused step: opponents driven inside Rust; also returns the learner's powered-cities count for reward shaping), `load_opponent_policy(bytes)` (loads a frozen PGRLPOL6 policy snapshot; difficulty `"policy"` then drives opponents with it via persistent per-seat bots — frozen-opponent self-play, used by `train_selfplay.py`).
- `legal_move_info` returns a JSON blob encoding every legal *primitive* move for the given actor. It predates the macro rebuild and is now only a debugging/inspection aid — the Python env gets `info["action_mask"]` from the native `action_mask(actor)` (macro legality, `macro_actions::legal_macros`) and never re-derives rules in Python.
- The obs/mask/action-id encoding lives in `powergrid_bot_strategy::encoding` + `macro_actions` (shared with the Expert bot); this crate only adds the PyO3/numpy wrappers. The encoding targets the **default (USA) map**; changing the default map invalidates trained checkpoints.
- `bot_decide`/`advance_bots`/`step_vs_bots` accept `"easy" | "normal" | "hard" | "expert"` (heuristic; `"expert"` plays as the hard-style heuristic here — the embedded Expert RL policy is never attached) or `"policy"` (the snapshot loaded via `load_opponent_policy`; errors if none loaded). Native Expert play exists only via `Session::add_bot` (client/lobby).
- Built with `maturin develop --release` from the `python/` directory (see `python/Makefile`).
- crate-type = `["cdylib"]` — produces a `.so` wheel, not a binary.

See [docs/rl-environment.md](docs/rl-environment.md) for the full Python API and training workflow.

### alphazero/

A second, independent training approach for the Expert policy — MCTS-guided self-play (AlphaZero), tried after the PettingZoo+PPO stack above repeatedly struggled (entropy collapse, pinned eval reward, brittle shaping). Not a PyO3 crate; a plain Python package at the repo root, structured like `alpha-zero-general` (Game adapter / NNet wrapper / MCTS / Coach) but fixed at **4 players** and built around action masking from the start. Reuses `powergrid_py` and `powergrid_env.constants` — no duplicated game logic, no sb3/PettingZoo dependency. Run with the `python/` venv as a module from the repo root (`python/.venv/bin/python -m alphazero.train ...`); see `alphazero/README.md` for the full runbook.

- `game.py` — `PowerGridGame`: adapter over `powergrid_py.Game` exposing `fork()` (wraps the Rust `copy()` above), `observation()`/`action_mask()`, `apply(action_id)`, `is_terminal()`/`outcome()`. `outcome()` is **rank-based** — linearly spaced finish-position values from +1 (1st) to −1 (last), e.g. `[+1, +1/3, −1/3, −1]` for 4p — not winner-take-all, so the value head gets a gradient between losing seats. Also the finish-position helpers (`finish_positions`/`_city_count`, re-exported by `metrics.py`) and the perspective-relative value-vector helpers (`relative_order`, `to_relative_vector`, `to_absolute_dict`) — the value head's 4-vector is ordered `[self, opponents...]` exactly like `build_observation`.
- `mcts.py` — `MCTS`/`Node`: node-based PUCT search (each node holds a forked `PowerGridGame`, not a transposition-table entry — Power Grid's state doesn't canonicalize cheaply, and forking is cheap). Perfect-information search on the full seeded `GameState` (deck order, opponent money all visible to the *search*); the network itself is only ever shown the masked `observation()`, so the trained policy never cheats, only the search does. Single-actor-per-turn means a plain absolute `{player_id: value}` dict backs up the tree unchanged — no per-player Q arrays needed. Unvisited children use **FPU reduction** (scored as parent Q − `fpu_reduction`, not 0); a root with a single legal move short-circuits to the one-hot without simulating.
- `network.py` — `PGNet`/`NNetWrapper`: trunk + policy head exactly match the exportable shape (`OBS_SIZE → H → tanh → H → tanh → N_ACTIONS`); `policy_state_dict()` emits those three layers under sb3's MaskablePPO key names so `powergrid_env.export.policy_state_dict_to_bytes` serializes it to PGRLPOL6 unchanged. The value head (4-vector) is a separate training-only branch, never exported. `train(examples, num_batches=None)` does epoch-style training (pretrain) when `num_batches` is `None`, else a fixed budget of that many uniformly-sampled minibatches (the coach's windowed replay). `load()` resets the optimizer lr to `cfg.lr` so a low-lr finetune resume isn't overridden by the checkpoint's saved lr.
- `selfplay.py` — three episode flavors sharing one MCTS loop (`_mcts_episode`), differing only in how non-learner seats advance: `play_episode` (pure self-play, every seat is the shared net, all seats recorded), `play_episode_vs_bots` (learner vs Rust heuristic bots, only learner recorded), `play_episode_vs_net` (learner vs a past checkpoint playing net-only masked-softmax sampling — league play). Forced moves (single legal action) are applied but not recorded. Also `_worker_init`/`_worker_run`: picklable `multiprocessing.Pool` entry points the coach uses to parallelize self-play.
- `coach.py`/`arena.py` — the iterate-train-eval-checkpoint loop; win-rate evaluation vs. Rust heuristic bots (same methodology as `python/scripts/evaluate.py`) or net-vs-net. Coach uses a **windowed** replay buffer (a deque of `buffer_iters` per-iteration example blocks), a fixed `train_batches` training budget per iteration, a `num_workers` process pool for self-play, a `vs_bot_fraction`/`vs_past_fraction` episode mix (past-checkpoint league opponents drawn from the run dir), net-only eval by default (`eval_num_sims=0` — the exported artifact), and `coach_state.json` for resume (continues iteration numbering, protects `best.pt`, refuses to overwrite a foreign non-empty run dir). Curriculum is off by default (rulebook 17); it was measured counterproductive.
- `dagger.py` — **DAgger / expert iteration**, the recommended Phase 2 (AZ self-play finetune was measured to *regress* a good behavior clone, 10.7%→2.0% vs normal — as a ~90% underdog the value head sees ~all positions as losing, so MCTS visit-count targets flatten the clone's policy). `generate_dagger_examples` rolls out net-vs-hard-bot games and labels each learner state with the hard bot's move (`bot_first_action_id`, reusing `imitation.py`'s build/buy decomposition); `main()` aggregates, retrains, evaluates, and saves `dagger.pt`. Sharp one-hot expert targets, no value head or search in the loop → holds/improves the clone instead of collapsing it. Pipeline: `pretrain.py` (BC warm start) → `dagger.py` (close the compounding-error gap toward the ~33% hard-bot ceiling). AZ (`train.py`) is retained but only worth revisiting from an already-bot-competent start.
- `export.py` — checkpoint → `assets/policies/expert.bin` + golden JSON, reusing `python/scripts/export_policy.py`'s serializer; verify with the Rust golden-logits parity test in `policy.rs`.

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
