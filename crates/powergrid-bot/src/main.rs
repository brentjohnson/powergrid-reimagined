use powergrid_bot::runtime::run_bot;
use powergrid_core::types::PlayerColor;

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
    tracing::info!("Bot '{}' ({:?}) connecting to {url}", args.name, args.color);

    run_bot(url, args.name, args.color).await;
}
