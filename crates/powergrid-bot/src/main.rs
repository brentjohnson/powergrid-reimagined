use powergrid_bot::strategy;

use futures_util::{SinkExt, StreamExt};
use powergrid_core::{
    actions::{Action, ServerMessage},
    types::{PlayerColor, PlayerId},
};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::Message as WsMessage, MaybeTlsStream, WebSocketStream,
};
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct Args {
    name: String,
    color: PlayerColor,
    server: String,
    port: u16,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut name: Option<String> = None;
    let mut color: Option<PlayerColor> = None;
    let mut server = String::from("localhost");
    let mut port: u16 = 3000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--name" => {
                i += 1;
                name = args.get(i).cloned();
            }
            "--color" => {
                i += 1;
                let s = args.get(i).ok_or("--color requires a value")?;
                color = Some(parse_color(s)?);
            }
            "--server" => {
                i += 1;
                server = args.get(i).cloned().ok_or("--server requires a value")?;
            }
            "--port" => {
                i += 1;
                let s = args.get(i).ok_or("--port requires a value")?;
                port = s.parse::<u16>().map_err(|_| "invalid port")?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(Args {
        name: name.ok_or("--name is required")?,
        color: color.ok_or("--color is required")?,
        server,
        port,
    })
}

fn print_help() {
    println!(
        "Usage: powergrid-bot --name <name> --color <color> [options]

Options:
  --name <name>    Bot player name (required)
  --color <color>  Bot player color (required)
                     Choices: red, blue, green, yellow, purple, white
  --server <host>  Server hostname (default: localhost)
  --port <port>    Server port (default: 3000)
  -h, --help       Show this help message"
    );
}

fn parse_color(s: &str) -> Result<PlayerColor, String> {
    match s.to_lowercase().as_str() {
        "red" => Ok(PlayerColor::Red),
        "blue" => Ok(PlayerColor::Blue),
        "green" => Ok(PlayerColor::Green),
        "yellow" => Ok(PlayerColor::Yellow),
        "purple" => Ok(PlayerColor::Purple),
        "white" => Ok(PlayerColor::White),
        other => Err(format!(
            "unknown color '{other}'; expected: red, blue, green, yellow, purple, white"
        )),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("powergrid_bot=debug,info")),
        )
        .init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Usage: powergrid-bot --name <name> --color <color> [--server <host>] [--port <port>]");
            eprintln!("Colors: red, blue, green, yellow, purple, white");
            std::process::exit(1);
        }
    };

    let url = format!("ws://{}:{}/ws", args.server, args.port);
    info!("Bot '{}' ({:?}) connecting to {url}", args.name, args.color);

    run_bot(url, args.name, args.color).await;
}

// ---------------------------------------------------------------------------
// Bot loop — reconnects forever
// ---------------------------------------------------------------------------

async fn run_bot(url: String, name: String, color: PlayerColor) {
    loop {
        match connect_async(&url).await {
            Ok((stream, _)) => {
                info!("Connected to {url}");
                match bot_session(stream, &name, color).await {
                    SessionResult::GameOver => {
                        info!("Game over — exiting");
                        return;
                    }
                    SessionResult::Disconnected => {
                        warn!("Disconnected — reconnecting in 3s…");
                    }
                }
            }
            Err(e) => {
                warn!("Connection failed: {e} — retrying in 3s…");
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    }
}

enum SessionResult {
    GameOver,
    Disconnected,
}

async fn bot_session(
    stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
    name: &str,
    color: PlayerColor,
) -> SessionResult {
    let (mut write, mut read) = stream.split();
    let mut my_id: Option<PlayerId> = None;

    while let Some(msg) = read.next().await {
        let text = match msg {
            Ok(WsMessage::Text(t)) => t,
            Ok(WsMessage::Ping(_) | WsMessage::Pong(_)) => continue,
            Ok(WsMessage::Close(_)) | Err(_) => return SessionResult::Disconnected,
            Ok(_) => continue,
        };

        let server_msg = match serde_json::from_str::<ServerMessage>(&text) {
            Ok(m) => m,
            Err(e) => {
                warn!("Deserialize error: {e}");
                continue;
            }
        };

        match server_msg {
            ServerMessage::Welcome { your_id } => {
                my_id = Some(your_id);
                info!("Received Welcome as {your_id}; sending JoinGame");
                let action = Action::JoinGame {
                    name: name.to_string(),
                    color,
                };
                if write
                    .send(WsMessage::Text(
                        serde_json::to_string(&action).expect("serialize"),
                    ))
                    .await
                    .is_err()
                {
                    return SessionResult::Disconnected;
                }
            }

            ServerMessage::StateUpdate(gs) => {
                let Some(id) = my_id else { continue };

                if let powergrid_core::types::Phase::GameOver { winner } = gs.phase {
                    if let Some(winner_player) = gs.player(winner) {
                        info!(
                            "Game over! Winner: {} ({:?})",
                            winner_player.name, winner_player.color
                        );
                    }
                    return SessionResult::GameOver;
                }

                // Small delay so humans can follow along in the UI.
                tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;

                if let Some(action) = strategy::decide(&gs, id) {
                    info!("Sending action: {:?}", action);
                    if write
                        .send(WsMessage::Text(
                            serde_json::to_string(&action).expect("serialize"),
                        ))
                        .await
                        .is_err()
                    {
                        return SessionResult::Disconnected;
                    }
                }
            }

            ServerMessage::ActionError { message } => {
                error!("Action rejected by server: {message}");
            }

            ServerMessage::Event { message } => {
                info!("Game event: {message}");
            }

            // Lobby-protocol messages — not expected from the legacy single-game server.
            _ => {}
        }
    }

    SessionResult::Disconnected
}
