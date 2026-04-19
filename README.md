# Powergrid

A multiplayer implementation of the Power Grid board game, playable over a local network. One player runs the server; everyone connects with the native GUI client.

## Download

Grab the latest binaries from the [Releases](../../releases/latest) page.

| File | Platform | Use |
|---|---|---|
| `powergrid-server-linux-x86_64` | Linux (x86_64) | Server |
| `powergrid-server-linux-aarch64` | Linux (ARM64) | Server |

## Quickstart (2+ players on a local network)

### 1. Start the server

One player hosts the server. Download the map file alongside the binary:

```bash
# Download the Germany map (or provide your own)
curl -O https://raw.githubusercontent.com/YOUR_ORG/powergrid-rust/main/maps/germany.toml

# Linux/macOS — make executable, then run
chmod +x powergrid-server-linux-x86_64
MAP_FILE=germany.toml ./powergrid-server-linux-x86_64
```

The server listens on port `3000`. Find your local IP address (e.g. `192.168.1.10`) — other players will need it to connect.

### 2. Connect

On the connect screen, fill in:

| Field | Value |
|---|---|
| Server URL | `ws://localhost:3000/ws` (same machine) or `ws://<host-ip>:3000/ws` (over network) |
| Your Name | Enter your name |
| Color | Pick a color |

Click **Connect** to join the lobby.

### 3. Start the game

The first player to connect is the host. Once at least 2 players have joined, the host clicks **Start Game**.

## Game Phases

Each round proceeds through these phases:

1. **Auction** — bid on power plants from the market. On your turn, click a plant to open bidding, or click **Pass**.
2. **Buy Resources** — buy coal, oil, garbage, or uranium to fuel your plants. Click a resource type to buy one unit, then **Done**.
3. **Build Cities** — connect your network to new cities. Enter a city ID and click **Build**, then **Done Building**.
4. **Bureaucracy** — power your connected cities. Click **Power Cities** to run all plants you can fuel. Earn money based on cities powered.

The game ends when a player connects enough cities to trigger the end condition. The player who powers the most cities wins.

## Server Configuration

The server is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | TCP port to listen on |
| `MAP_FILE` | `maps/germany.toml` | Path to the map TOML file |
| `RUST_LOG` | _(unset)_ | Log level (`info`, `debug`, etc.) |

Example:

```bash
PORT=8080 MAP_FILE=germany.toml RUST_LOG=info ./powergrid-server-linux-x86_64
```

## Maps

Maps are TOML files that define cities and connections. The included map is **Germany**. You can create your own — add `[[cities]]` entries (id, name, region) and `[[connections]]` entries (from, to, cost), then point `MAP_FILE` at the new file.

## Health Check

```bash
curl http://localhost:3000/health
# → ok
```
