# Powergrid

A multiplayer implementation of the Power Grid board game. Play locally against bots, online via a lobby server, or on a local network with friends.

## Download

Grab the latest binaries from the [Releases](../../releases/latest) page.

| File | Platform | Use |
|---|---|---|
| `powergrid-client-linux-x86_64` | Linux (x86_64) | Client (all play modes) |
| `powergrid-server-linux-x86_64` | Linux (x86_64) | LAN server (optional) |
| `powergrid-server-linux-aarch64` | Linux (ARM64) | LAN server (optional) |

## Play Modes

### Local Play (no server required)

Launch the client and choose **LOCAL PLAY** from the main menu. Configure your name, color, and bot opponents (up to 5 bots, each with Easy/Normal/Hard difficulty), then click **Start**.

No network connection or account needed.

### Online Play (lobby server)

Choose **ONLINE PLAY** from the main menu. You'll be prompted to log in or register an account. The client connects to `powergrid.onyxoryx.net` by default.

Once logged in:
1. Enter a room name and click **Create Room** or **Join Room**.
2. Add bots or wait for other players to join.
3. The room creator clicks **Start Game** once at least 2 players are in the room.

### LAN Play (self-hosted server)

One player runs the legacy server — no database required. Others connect with the client.

#### 1. Start the server

```bash
chmod +x powergrid-server-linux-x86_64
./powergrid-server-linux-x86_64
```

The server listens on port `3000`. Find your local IP address (e.g. `192.168.1.10`) — other players will use it to connect.

#### 2. Connect with the client

Launch the client. On the login screen, the server field defaults to `powergrid.onyxoryx.net` — change it to your LAN server's IP and port before logging in.

Alternatively, use CLI flags to point at your LAN server:

```bash
./powergrid-client --server 192.168.1.10 --port 3000
```

## Game Phases

Each round proceeds through these phases:

1. **Auction** — bid on power plants from the market. On your turn, click a plant to open bidding, or click **Pass**.
2. **Buy Resources** — buy coal, oil, garbage, or uranium to fuel your plants. Click a resource type to buy one unit, then **Done**.
3. **Build Cities** — connect your network to new cities. Enter a city ID and click **Build**, then **Done Building**.
4. **Bureaucracy** — power your connected cities. Click **Power Cities** to run all plants you can fuel. Earn money based on cities powered.

The game ends when a player connects enough cities to trigger the end condition. The player who powers the most cities wins.

## Client CLI Flags

```
--server <host>         Server hostname (default: powergrid.onyxoryx.net)
--port <port>           Server port (default: 3000)
--color <color>         Auto-select player color on connect
                          Choices: red, blue, green, yellow, purple, white
--room <name>           Auto-create/join this room on connect
-w, --windowed          Run in a window instead of borderless fullscreen
-n, --no-preferences    Don't load or save credentials/preferences (for testing)
```

## LAN Server Configuration

The legacy server is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | TCP port to listen on |
| `MAP_FILE` | _(built-in Germany map)_ | Path to a custom map TOML file |
| `RUST_LOG` | _(unset)_ | Log level (`info`, `debug`, etc.) |

Example:

```bash
PORT=8080 MAP_FILE=my_map.toml RUST_LOG=info ./powergrid-server-linux-x86_64
```

## Maps

Maps are TOML files that define cities and connections. The included map is **Germany**, embedded at compile time. To use a custom map, set `MAP_FILE=/path/to/map.toml` when starting the server.

A map file contains `[[cities]]` entries (id, name, region) and `[[connections]]` entries (from, to, cost).

## Docker (lobby server)

Runs the full lobby server with PostgreSQL:

```bash
docker compose up --build
```

The lobby server requires a `DATABASE_URL` pointing at a PostgreSQL instance. The Docker Compose file configures this automatically.

## Health Check

```bash
curl http://localhost:3000/health
# → ok
```

## Reinforcement Learning

A [PettingZoo 1.26.1](https://pettingzoo.farama.org/) environment wraps the game engine for training neural-network agents. The bridge is a PyO3 Rust extension — no WebSocket server required.

```bash
cd python
make develop                                       # build Rust extension + install Python package
python scripts/train_vs_bots.py                    # train MaskablePPO vs Rust strategy bots
python scripts/train_selfplay.py --num-players 4   # self-play across all seats
python scripts/play_game.py --all-bots --render    # watch a game
```

See [docs/rl-environment.md](docs/rl-environment.md) for the full API, action/observation encoding, and training instructions.
