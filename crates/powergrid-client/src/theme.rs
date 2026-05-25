use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Vec2, Visuals};

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
// Resource market
// ---------------------------------------------------------------------------

pub const RES_COAL: Color32 = Color32::from_rgb(150, 100, 55);
pub const RES_OIL: Color32 = Color32::from_rgb(110, 110, 140);
pub const RES_GAS: Color32 = Color32::from_rgb(70, 130, 220);
pub const RES_URANIUM: Color32 = Color32::from_rgb(200, 30, 30);

// ---------------------------------------------------------------------------
// Map overlays (opaque as const, translucent as fn)
// ---------------------------------------------------------------------------

pub const BG_MAP: Color32 = Color32::from_rgb(2, 4, 8);
pub const MAP_CONN_COST_LABEL: Color32 = Color32::from_rgb(255, 240, 160);
pub const MAP_SLOT_ACTIVE_BORDER: Color32 = Color32::from_rgb(0, 230, 255);
pub const MAP_CITY_LABEL_ACTIVE: Color32 = Color32::from_rgb(200, 220, 200);

pub fn map_conn_color() -> Color32 {
    Color32::from_rgba_unmultiplied(90, 80, 65, 180)
}
pub fn map_conn_glow() -> Color32 {
    Color32::from_rgba_unmultiplied(180, 160, 120, 60)
}
pub fn map_city_inactive_house() -> Color32 {
    Color32::from_rgba_unmultiplied(60, 60, 60, 100)
}
pub fn map_city_bg() -> Color32 {
    Color32::from_rgba_unmultiplied(18, 22, 28, 235)
}
pub fn map_city_border() -> Color32 {
    Color32::from_rgba_unmultiplied(70, 80, 90, 160)
}
pub fn map_slot_active_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(0, 150, 210, 150)
}
pub fn map_slot_locked() -> Color32 {
    Color32::from_rgba_unmultiplied(60, 100, 90, 40)
}
pub fn map_city_label_dim() -> Color32 {
    Color32::from_rgba_unmultiplied(130, 150, 130, 200)
}

// ---------------------------------------------------------------------------
// Power plant card kinds
// ---------------------------------------------------------------------------

pub const CARD_COAL: Color32 = Color32::from_rgb(140, 90, 45);
pub const CARD_OIL: Color32 = Color32::from_rgb(100, 100, 120);
pub const CARD_GAS_OIL: Color32 = Color32::from_rgb(70, 110, 150);
pub const CARD_GAS: Color32 = Color32::from_rgb(40, 140, 210);
pub const CARD_URANIUM: Color32 = Color32::from_rgb(220, 40, 60);
pub const CARD_WIND: Color32 = Color32::from_rgb(0, 200, 170);

// ---------------------------------------------------------------------------
// City-history graph indicators
// ---------------------------------------------------------------------------

pub fn city_graph_step2() -> Color32 {
    Color32::from_rgba_unmultiplied(180, 180, 60, 180)
}
pub fn city_graph_end() -> Color32 {
    Color32::from_rgba_unmultiplied(220, 80, 80, 200)
}

// ---------------------------------------------------------------------------
// Heading / label font sizes
// ---------------------------------------------------------------------------

pub const HEADING_XL: f32 = 42.0;
pub const HEADING_L: f32 = 32.0;
pub const HEADING_M: f32 = 20.0;
pub const LABEL_S: f32 = 14.0;

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

    visuals.window_corner_radius = CornerRadius::same(3);
    visuals.window_stroke = Stroke::new(1.0, NEON_CYAN_DARK);

    // Non-interactive widgets (labels, separators)
    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_PANEL);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_MID);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(2);

    // Inactive (e.g. text input at rest)
    visuals.widgets.inactive.bg_fill = BG_WIDGET;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, NEON_CYAN_DARK);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_MID);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(2);

    // Hovered
    visuals.widgets.hovered.bg_fill = BG_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.5, NEON_CYAN_DIM);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, NEON_CYAN);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(2);

    // Active (clicked / focused)
    visuals.widgets.active.bg_fill = BG_ACTIVE;
    visuals.widgets.active.bg_stroke = Stroke::new(2.0, NEON_CYAN);
    visuals.widgets.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(2);

    // Open (dropdown open state)
    visuals.widgets.open.bg_fill = BG_HOVER;
    visuals.widgets.open.bg_stroke = Stroke::new(1.5, NEON_CYAN_DIM);
    visuals.widgets.open.fg_stroke = Stroke::new(1.5, NEON_CYAN);
    visuals.widgets.open.corner_radius = CornerRadius::same(2);

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
    style.spacing.window_margin = egui::Margin::same(10);
    ctx.set_style(style);
}

// ---------------------------------------------------------------------------
// Convenience: draw a neon-bordered panel frame
// ---------------------------------------------------------------------------

pub fn neon_frame() -> egui::Frame {
    egui::Frame::NONE
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.0, BORDER_PANEL))
        .inner_margin(egui::Margin::same(8))
        .corner_radius(CornerRadius::same(3))
}

pub fn neon_frame_bright() -> egui::Frame {
    egui::Frame::NONE
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.5, NEON_CYAN_DIM))
        .inner_margin(egui::Margin::same(8))
        .corner_radius(CornerRadius::same(3))
}

pub fn panel_frame(inner_margin: i8) -> egui::Frame {
    egui::Frame::NONE
        .fill(BG_DEEP)
        .stroke(Stroke::new(1.0, NEON_CYAN_DARK))
        .inner_margin(egui::Margin::same(inner_margin))
}

pub fn draw_backdrop(ui: &mut egui::Ui) {
    let panel = ui.max_rect();
    let img_aspect = 1920.0_f32 / 1280.0;
    let panel_aspect = panel.width() / panel.height();
    let size = if panel_aspect > img_aspect {
        egui::vec2(panel.width(), panel.width() / img_aspect)
    } else {
        egui::vec2(panel.height() * img_aspect, panel.height())
    };
    let bg_rect = egui::Rect::from_center_size(panel.center(), size);
    egui::Image::new(egui::include_image!("../assets/splash.png")).paint_at(ui, bg_rect);
    ui.painter()
        .rect_filled(panel, 0.0, egui::Color32::from_black_alpha(140));
}
