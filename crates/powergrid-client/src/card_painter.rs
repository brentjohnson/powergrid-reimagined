use bevy_egui::egui;
use egui::{Color32, CornerRadius, FontFamily, FontId, Rect, Stroke, StrokeKind, Vec2};
use powergrid_core::types::{PlantKind, PowerPlant};

use crate::theme;

// ---------------------------------------------------------------------------
// PlantKind color + label
// ---------------------------------------------------------------------------

fn kind_color(kind: PlantKind) -> Color32 {
    match kind {
        PlantKind::Coal => Color32::from_rgb(140, 90, 45),
        PlantKind::Oil => Color32::from_rgb(100, 100, 120),
        PlantKind::CoalOrOil => Color32::from_rgb(120, 95, 55),
        PlantKind::Garbage => Color32::from_rgb(190, 175, 30),
        PlantKind::Uranium => Color32::from_rgb(220, 40, 60),
        PlantKind::Wind => Color32::from_rgb(0, 200, 170),
        PlantKind::Fusion => Color32::from_rgb(180, 100, 255),
    }
}

fn kind_label(kind: PlantKind) -> &'static str {
    match kind {
        PlantKind::Coal => "COAL",
        PlantKind::Oil => "OIL",
        PlantKind::CoalOrOil => "HYBRID",
        PlantKind::Garbage => "GARBAGE",
        PlantKind::Uranium => "URANIUM",
        PlantKind::Wind => "WIND",
        PlantKind::Fusion => "FUSION",
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Draw a power plant card at the given size and return the egui Response.
/// The response can be checked for `.clicked()` and `.hovered()`.
pub fn draw_plant_card(ui: &mut egui::Ui, plant: &PowerPlant, size: f32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::click());

    if ui.is_rect_visible(rect) {
        paint_card(ui, rect, plant, size);
    }

    response
}

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------

fn paint_card(ui: &mut egui::Ui, rect: Rect, plant: &PowerPlant, size: f32) {
    let painter = ui.painter_at(rect);
    let rounding = CornerRadius::same(3);

    // Step 3 special card
    if plant.number == 0 {
        painter.rect_filled(rect, rounding, theme::BG_WIDGET);
        painter.rect_stroke(
            rect,
            rounding,
            Stroke::new(1.5, theme::NEON_AMBER),
            StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "STEP\n 3",
            FontId::new(size * 0.22, FontFamily::Monospace),
            theme::NEON_AMBER,
        );
        return;
    }

    let color = kind_color(plant.kind);

    // Background + border
    painter.rect_filled(rect, rounding, theme::BG_WIDGET);
    painter.rect_stroke(rect, rounding, Stroke::new(1.5, color), StrokeKind::Inside);

    if size >= 60.0 {
        // ---- Medium card (70px) ----
        // Color bar at top
        let top_bar = Rect::from_min_size(rect.min, Vec2::new(size, size * 0.20));
        painter.rect_filled(
            top_bar,
            CornerRadius {
                nw: 3,
                ne: 3,
                sw: 0,
                se: 0,
            },
            color.linear_multiply(0.35),
        );

        // Kind label inside top bar
        painter.text(
            top_bar.center(),
            egui::Align2::CENTER_CENTER,
            kind_label(plant.kind),
            FontId::new(size * 0.12, FontFamily::Monospace),
            color,
        );

        // Plant number — large, centered in the middle area
        let mid_center = rect.center() + Vec2::new(0.0, size * 0.06);
        painter.text(
            mid_center,
            egui::Align2::CENTER_CENTER,
            plant.number.to_string(),
            FontId::new(size * 0.32, FontFamily::Monospace),
            theme::TEXT_BRIGHT,
        );

        // Bottom row: cost → cities (or just cities for free plants)
        let bottom_y = rect.max.y - size * 0.13;
        let stats = if plant.kind.needs_resources() {
            format!("{}  \u{2192}  {}", plant.cost, plant.cities)
        } else {
            format!("\u{2192}  {}", plant.cities)
        };
        painter.text(
            egui::pos2(rect.center().x, bottom_y),
            egui::Align2::CENTER_CENTER,
            stats,
            FontId::new(size * 0.13, FontFamily::Monospace),
            theme::TEXT_MID,
        );
    } else {
        // ---- Small card (44px) ----
        // Thin color bar at the bottom
        let bar_h = size * 0.15;
        let bar = Rect::from_min_size(
            egui::pos2(rect.min.x, rect.max.y - bar_h),
            Vec2::new(size, bar_h),
        );
        painter.rect_filled(
            bar,
            CornerRadius {
                nw: 0,
                ne: 0,
                sw: 3,
                se: 3,
            },
            color.linear_multiply(0.5),
        );

        // Plant number — large, centered
        painter.text(
            rect.center() - Vec2::new(0.0, bar_h * 0.4),
            egui::Align2::CENTER_CENTER,
            plant.number.to_string(),
            FontId::new(size * 0.38, FontFamily::Monospace),
            theme::TEXT_BRIGHT,
        );
    }
}
