mod assets;
mod map_panel;
mod state;
mod theme;
mod ui;
mod ws;

use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use state::{AppState, CliArgs};

fn main() {
    let cli = CliArgs::parse();
    let auto_connect = cli.auto_connect;
    let url = cli
        .url
        .clone()
        .unwrap_or_else(|| "ws://localhost:3000/ws".to_string());
    let pending_join = if auto_connect {
        cli.name.as_ref().map(|name| {
            (
                name.clone(),
                cli.color.unwrap_or(powergrid_core::types::PlayerColor::Red),
            )
        })
    } else {
        None
    };

    let mut app_state = AppState::new(cli);
    // If all three args provided, kick off connection immediately.
    if auto_connect {
        let channels = ws::spawn_ws(url.clone());
        app_state.pending_join = pending_join;

        App::new()
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Power Grid: Reimagined".into(),
                    resolution: (1600.0, 900.0).into(),
                    ..default()
                }),
                ..default()
            }))
            .add_plugins(EguiPlugin)
            .insert_resource(app_state)
            .insert_resource(channels)
            .add_systems(Startup, assets::setup_assets)
            .add_systems(Startup, ui::setup_egui_theme)
            .add_systems(Update, (ws::process_ws_events, ui::ui_system).chain())
            .run();
    } else {
        App::new()
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Power Grid: Reimagined".into(),
                    resolution: (1600.0, 900.0).into(),
                    ..default()
                }),
                ..default()
            }))
            .add_plugins(EguiPlugin)
            .insert_resource(app_state)
            .add_systems(Startup, assets::setup_assets)
            .add_systems(Startup, ui::setup_egui_theme)
            .add_systems(Update, (ws::process_ws_events, ui::ui_system).chain())
            .run();
    }
}
