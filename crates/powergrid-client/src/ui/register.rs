use egui::RichText;
use powergrid_core::limits::{MAX_EMAIL, MAX_PASSWORD, MAX_USERNAME, MIN_PASSWORD, MIN_USERNAME};

use crate::{
    auth::{do_register, AuthEvent},
    state::{AppState, Screen},
    theme,
};

pub(super) fn register_screen(ctx: &egui::Context, state: &mut AppState) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::NONE
                .fill(theme::BG_DEEP)
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            theme::draw_backdrop(ui);
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                ui.label(
                    RichText::new("POWER GRID")
                        .size(theme::HEADING_XL)
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
                ui.label(
                    RichText::new("REIMAGINED")
                        .size(theme::HEADING_M)
                        .color(theme::NEON_CYAN_DIM)
                        .monospace(),
                );

                ui.add_space(40.0);

                theme::neon_frame().show(ui, |ui| {
                    ui.set_width(420.0);
                    ui.spacing_mut().item_spacing.y = 10.0;

                    ui.label(
                        RichText::new("CREATE ACCOUNT")
                            .color(theme::NEON_CYAN)
                            .monospace(),
                    );
                    ui.add_space(4.0);

                    ui.label(RichText::new("EMAIL").color(theme::TEXT_DIM).small());
                    ui.add(
                        egui::TextEdit::singleline(&mut state.register_email).char_limit(MAX_EMAIL),
                    );

                    ui.label(
                        RichText::new(format!(
                            "USERNAME ({MIN_USERNAME}–{MAX_USERNAME} chars, letters/digits/-/_)"
                        ))
                        .color(theme::TEXT_DIM)
                        .small(),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut state.register_username)
                            .char_limit(MAX_USERNAME),
                    );

                    ui.label(
                        RichText::new(format!("PASSWORD ({MIN_PASSWORD}–{MAX_PASSWORD} chars)"))
                            .color(theme::TEXT_DIM)
                            .small(),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut state.register_password)
                            .password(true)
                            .char_limit(MAX_PASSWORD),
                    );

                    if let Some(ref err) = state.auth_error.clone() {
                        ui.add_space(4.0);
                        ui.label(RichText::new(err).color(theme::NEON_RED).small());
                    }

                    ui.add_space(8.0);

                    let can_submit = !state.auth_in_flight
                        && !state.register_email.trim().is_empty()
                        && !state.register_username.trim().is_empty()
                        && !state.register_password.is_empty();

                    let btn = egui::Button::new(
                        RichText::new(if state.auth_in_flight {
                            "[ CREATING… ]"
                        } else {
                            "[ CREATE ACCOUNT ]"
                        })
                        .color(if can_submit {
                            theme::BG_DEEP
                        } else {
                            theme::TEXT_DIM
                        })
                        .monospace(),
                    )
                    .fill(if can_submit {
                        theme::NEON_CYAN
                    } else {
                        theme::BG_WIDGET
                    })
                    .stroke(egui::Stroke::new(
                        1.5,
                        if can_submit {
                            theme::NEON_CYAN
                        } else {
                            theme::NEON_CYAN_DARK
                        },
                    ));

                    if ui.add_enabled(can_submit, btn).clicked() && can_submit {
                        submit_register(state);
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("Already have an account?")
                                .color(theme::TEXT_DIM)
                                .small(),
                        );
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("[ LOG IN ]")
                                        .color(theme::NEON_CYAN_DIM)
                                        .small()
                                        .monospace(),
                                )
                                .fill(egui::Color32::TRANSPARENT)
                                .stroke(egui::Stroke::NONE),
                            )
                            .clicked()
                        {
                            state.auth_error = None;
                            state.screen = Screen::Login;
                        }
                    });
                });
            });
        });
}

fn submit_register(state: &mut AppState) {
    state.auth_error = None;
    state.auth_in_flight = true;

    let server = state.server_name.clone();
    let port = state.port;
    let email = state.register_email.clone();
    let username = state.register_username.clone();
    let password = state.register_password.clone();
    let slot = state.auth_pending.0.clone();

    std::thread::spawn(move || {
        let event = match do_register(&server, port, &email, &username, &password) {
            Ok(c) => AuthEvent::Success(c),
            Err(e) => AuthEvent::Failure(e),
        };
        *slot.lock().unwrap() = Some(event);
    });
}
