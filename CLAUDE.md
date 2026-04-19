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

# Run the server (from repo root)
cargo run -p powergrid-server

# Docker
docker compose up --build
```

## Workflow

Before running a build, do "cargo fmt" "cargo check" and run clippy.  Then fix any issues before building.

## Architecture

Four-crate Cargo workspace:

```
crates/
  powergrid-core/    # pure game logic, no I/O
  powergrid-server/  # axum WebSocket server, maps/germany.toml embedded at compile time
  powergrid-client/  # Bevy/egui GUI client
```

### powergrid-core

All game state and rules. The key entry point is `rules::apply_action(state, player_id, action) -> Result<(), ActionError>`. It's pure — no I/O — and fully unit-testable.

- `types.rs` — `Player`, `PowerPlant`, `ResourceMarket`, `PlantMarket`, `Phase`, `PlayerColor`, etc.
- `state.rs` — `GameState` struct (all game data including the map)
- `actions.rs` — `Action` enum (client → server), `ServerMessage` enum (server → client), `ActionError`
- `map.rs` — `Map` (runtime graph) + `MapData` (TOML-deserializable). Dijkstra routing in `Map::connection_cost_to`.
- `rules.rs` — `apply_action` dispatcher + one `handle_*` function per phase. Also `build_plant_deck()`.

**Phase flow:** `Lobby → Auction → BuyResources → BuildCities → Bureaucracy → [next round or GameOver]`

### powergrid-server

- `main.rs` — axum router: `GET /health`, `GET /ws`. Shared state is `Arc<Mutex<ServerState>>`.
- `ws.rs` — per-connection WebSocket handler. On each valid action: mutate state, broadcast full `GameState` JSON to all clients. On error: send `ActionError` only to the acting client.
- Configured via env vars: `PORT` (default 3000), `MAP_FILE` (optional override; germany map is embedded by default), `RUST_LOG`.

### powergrid-client

Bevy + egui GUI client that connects to the server over WebSocket.

- `main.rs` — Bevy app setup; accepts optional CLI args for auto-connect (`--url`, `--name`, `--color`).
- `ws.rs` — spawns a background thread with a Tokio runtime; communicates with the Bevy app via `crossbeam-channel`. Reconnects automatically on disconnect.
- `ui/` — egui UI systems split into submodules:
  - `mod.rs` — `setup_egui_theme`, `ui_system`, `game_screen` dispatch
  - `connect.rs` — Connect screen (login form, color selector)
  - `lobby.rs` — Lobby screen (player list, start button)
  - `top_panel.rs` — round/phase header + resource market
  - `phase_tracker.rs` — per-player phase progress dots
  - `left_panel.rs` — player info cards
  - `action_panel.rs` — phase-specific interactive controls
  - `right_panel.rs` — plant market, action console, event log
  - `helpers.rs` — shared widgets and utilities (`section_header`, `neon_button`, `send`, etc.)
- `state.rs` — `AppState` Bevy resource holding game state, connection status, and screen enum.
- `map_panel.rs` — renders the map with city overlays.
- `card_painter.rs` — procedurally paints power plant cards using egui primitives.
- `theme.rs` — applies a custom egui visual theme.

Run with `cargo run -p powergrid-client` or `cargo run -p powergrid-client --features dev` for fast incremental rebuilds.

### Protocol

JSON over WebSocket. `Action` (tagged by `"type"` field) client→server, `ServerMessage` server→client. Full `GameState` broadcast after every valid action.

### Map format

`crates/powergrid-server/maps/*.toml` — list of `[[cities]]` (id, name, region) and `[[connections]]` (from, to, cost). The germany map is embedded at compile time. To use a custom map, set `MAP_FILE=/path/to/map.toml`.
