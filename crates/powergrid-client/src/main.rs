mod auth;
mod card_painter;
mod local;
mod map_panel;
mod state;
mod theme;
mod ui;
mod ws;

use bevy::prelude::*;
use bevy::window::{MonitorSelection, WindowMode};
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use state::{AppState, CliArgs};
use std::time::Duration;

fn main() {
    let cli = CliArgs::parse();
    let windowed = cli.windowed;

    let window_mode = if windowed {
        WindowMode::Windowed
    } else {
        WindowMode::BorderlessFullscreen(MonitorSelection::Current)
    };

    let app_state = AppState::new(cli);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Power Grid: Reimagined".into(),
                resolution: (1600_u32, 900_u32).into(),
                mode: window_mode,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive(Duration::from_millis(100)),
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(1000)),
        })
        .insert_resource(app_state)
        .add_systems(Startup, spawn_camera)
        .add_systems(Update, (state::process_auth_events, ws::process_ws_events))
        .add_systems(EguiPrimaryContextPass, ui::ui_system)
        .run();
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
