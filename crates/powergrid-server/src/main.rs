use powergrid_core::map::Map;
use powergrid_server::serve_embedded;
use tracing::info;

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: powergrid-server

Environment variables:
  PORT       Port to listen on (default: 3000)
  MAP_FILE   Path to a custom map TOML file (default: embedded Germany map)
  RUST_LOG   Log filter, e.g. debug or info (default: info)

Options:
  -h, --help   Show this help message"
        );
        std::process::exit(0);
    }

    tracing_subscriber::fmt::init();

    const DEFAULT_MAP: &str = include_str!("../maps/germany.toml");

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());

    let map_str = if let Ok(path) = std::env::var("MAP_FILE") {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read map file {path}: {e}"))
    } else {
        DEFAULT_MAP.to_string()
    };
    let map = Map::load(&map_str).unwrap_or_else(|e| panic!("Failed to parse map: {e}"));
    info!("Loaded map: {}", map.name);

    let addr = format!("0.0.0.0:{port}");
    let (bound_addr, fut) = serve_embedded(map, &addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind {addr}: {e}"));
    info!("Listening on {bound_addr}");
    fut.await.unwrap_or_else(|e| panic!("Server error: {e}"));
}
