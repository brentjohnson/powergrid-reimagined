use egui::{Color32, FontFamily, FontId, Rounding, Stroke, TextStyle, Vec2, Visuals};

// ---------------------------------------------------------------------------
// Neon palette
// ---------------------------------------------------------------------------

pub const BG_DEEP: Color32 = Color32::from_rgb(4, 6, 12);
pub const BG_PANEL: Color32 = Color32::from_rgb(6, 10, 18);
pub const BG_WIDGET: Color32 = Color32::from_rgb(8, 16, 28);
pub const BG_HOVER: Color32 = Color32::from_rgb(0, 40, 55);
pub const BG_ACTIVE: Color32 = Color32::from_rgb(0, 70, 80);

pub const NEON_CYAN: Color32 = Color32::from_rgb(0, 220, 190);
pub const NEON_CYAN_DIM: Color32 = Color32::from_rgb(0, 130, 110);
pub const NEON_CYAN_DARK: Color32 = Color32::from_rgb(0, 55, 48);
pub const NEON_GREEN: Color32 = Color32::from_rgb(50, 240, 100);
pub const NEON_AMBER: Color32 = Color32::from_rgb(255, 175, 0);
pub const NEON_RED: Color32 = Color32::from_rgb(255, 45, 75);

pub const TEXT_BRIGHT: Color32 = Color32::from_rgb(200, 240, 230);
pub const TEXT_MID: Color32 = Color32::from_rgb(120, 180, 165);
pub const TEXT_DIM: Color32 = Color32::from_rgb(60, 100, 90);

pub const BORDER_PANEL: Color32 = NEON_CYAN_DARK;

// ---------------------------------------------------------------------------
// Apply theme to egui context
// ---------------------------------------------------------------------------

pub fn apply(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();

    visuals.override_text_color = Some(TEXT_BRIGHT);
    visuals.window_fill = BG_DEEP;
    visuals.panel_fill = BG_PANEL;
    visuals.faint_bg_color = BG_WIDGET;
    visuals.extreme_bg_color = BG_DEEP;

    visuals.window_rounding = Rounding::same(3.0);
    visuals.window_stroke = Stroke::new(1.0, NEON_CYAN_DARK);

    // Non-interactive widgets (labels, separators)
    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_PANEL);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_MID);
    visuals.widgets.noninteractive.rounding = Rounding::same(2.0);

    // Inactive (e.g. text input at rest)
    visuals.widgets.inactive.bg_fill = BG_WIDGET;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, NEON_CYAN_DARK);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_MID);
    visuals.widgets.inactive.rounding = Rounding::same(2.0);

    // Hovered
    visuals.widgets.hovered.bg_fill = BG_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.5, NEON_CYAN_DIM);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, NEON_CYAN);
    visuals.widgets.hovered.rounding = Rounding::same(2.0);

    // Active (clicked / focused)
    visuals.widgets.active.bg_fill = BG_ACTIVE;
    visuals.widgets.active.bg_stroke = Stroke::new(2.0, NEON_CYAN);
    visuals.widgets.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
    visuals.widgets.active.rounding = Rounding::same(2.0);

    // Open (dropdown open state)
    visuals.widgets.open.bg_fill = BG_HOVER;
    visuals.widgets.open.bg_stroke = Stroke::new(1.5, NEON_CYAN_DIM);
    visuals.widgets.open.fg_stroke = Stroke::new(1.5, NEON_CYAN);
    visuals.widgets.open.rounding = Rounding::same(2.0);

    visuals.selection.bg_fill = Color32::from_rgb(0, 90, 75);
    visuals.selection.stroke = Stroke::new(1.0, NEON_CYAN);

    visuals.hyperlink_color = NEON_CYAN;
    visuals.warn_fg_color = NEON_AMBER;
    visuals.error_fg_color = NEON_RED;

    ctx.set_visuals(visuals);

    // Typography: use the built-in proportional font but scaled up slightly.
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Small, FontId::new(11.0, FontFamily::Monospace)),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Monospace)),
        (
            TextStyle::Monospace,
            FontId::new(13.0, FontFamily::Monospace),
        ),
        (TextStyle::Button, FontId::new(13.0, FontFamily::Monospace)),
        (TextStyle::Heading, FontId::new(18.0, FontFamily::Monospace)),
    ]
    .into();
    style.spacing.item_spacing = Vec2::new(6.0, 4.0);
    style.spacing.button_padding = Vec2::new(10.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(10.0);
    ctx.set_style(style);
}

// ---------------------------------------------------------------------------
// Convenience: draw a neon-bordered panel frame
// ---------------------------------------------------------------------------

pub fn neon_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.0, BORDER_PANEL))
        .inner_margin(egui::Margin::same(8.0))
        .rounding(Rounding::same(3.0))
}

pub fn neon_frame_bright() -> egui::Frame {
    egui::Frame::none()
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.5, NEON_CYAN_DIM))
        .inner_margin(egui::Margin::same(8.0))
        .rounding(Rounding::same(3.0))
}
