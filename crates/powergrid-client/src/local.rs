use bevy::prelude::*;
use powergrid_core::{map::Map, types::PlayerColor};
use tokio::sync::oneshot;

use crate::ws::{spawn_ws, WsChannels, WsMode};

pub struct LocalConfig {
    pub human_name: String,
    pub human_color: PlayerColor,
    pub bot_count: u8,
}

#[derive(Resource)]
pub struct LocalHandle {
    shutdown: Option<oneshot::Sender<()>>,
    runtime_thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for LocalHandle {
    fn drop(&mut self) {
        drop(self.shutdown.take());
        if let Some(t) = self.runtime_thread.take() {
            t.join().ok();
        }
    }
}

pub fn start_local_session(cfg: LocalConfig) -> (WsChannels, LocalHandle) {
    const DEFAULT_MAP: &str = include_str!("../../powergrid-server/maps/germany.toml");
    let map = Map::load(DEFAULT_MAP).expect("embedded map must be valid");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();

    let all_colors = [
        PlayerColor::Red,
        PlayerColor::Blue,
        PlayerColor::Green,
        PlayerColor::Yellow,
        PlayerColor::Purple,
        PlayerColor::White,
    ];
    let bot_colors: Vec<PlayerColor> = all_colors
        .iter()
        .copied()
        .filter(|&c| c != cfg.human_color)
        .take(cfg.bot_count as usize)
        .collect();

    let runtime_thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        rt.block_on(async move {
            let (addr, server_fut) = powergrid_server::serve_embedded(map, "127.0.0.1:0")
                .await
                .expect("bind local server");

            tokio::spawn(server_fut);
            addr_tx.send(addr).ok();

            // Give the human's WS thread time to connect and join before bots do,
            // so the human becomes host (players.first()) and can start the game.
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            for (i, color) in bot_colors.into_iter().enumerate() {
                let url = format!("ws://{addr}/ws");
                let name = format!("Bot {}", i + 1);
                tokio::spawn(powergrid_bot::runtime::run_bot(url, name, color));
            }

            let _ = shutdown_rx.await;
        });
    });

    let addr = addr_rx.recv().expect("server must bind before returning");

    let channels = spawn_ws(
        format!("ws://{addr}/ws"),
        WsMode::Legacy {
            name: cfg.human_name,
            color: cfg.human_color,
        },
    );

    (
        channels,
        LocalHandle {
            shutdown: Some(shutdown_tx),
            runtime_thread: Some(runtime_thread),
        },
    )
}
