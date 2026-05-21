use egui::{Color32, CornerRadius, FontFamily, FontId, Rect, Stroke, StrokeKind, Vec2};
use powergrid_core::types::{PlantKind, PowerPlant};

use crate::theme;

// ---------------------------------------------------------------------------
// Card dimensions
// ---------------------------------------------------------------------------

pub const CARD_W: f32 = 120.0;
pub const CARD_H: f32 = 26.0;

// ---------------------------------------------------------------------------
// PlantKind color + label
// ---------------------------------------------------------------------------

fn kind_color(kind: PlantKind) -> Color32 {
    match kind {
        PlantKind::Coal => theme::CARD_COAL,
        PlantKind::Oil => theme::CARD_OIL,
        PlantKind::GasOrOil => theme::CARD_GAS_OIL,
        PlantKind::Gas => theme::CARD_GAS,
        PlantKind::Uranium => theme::CARD_URANIUM,
        PlantKind::Wind => theme::CARD_WIND,
    }
}

fn kind_label(kind: PlantKind) -> &'static str {
    match kind {
        PlantKind::Coal => "COAL",
        PlantKind::Oil => "OIL",
        PlantKind::GasOrOil => "GAS/OIL",
        PlantKind::Gas => "GAS",
        PlantKind::Uranium => "URANIUM",
        PlantKind::Wind => "WIND",
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Draw a power plant card (CARD_W × CARD_H) and return the egui Response.
/// The response can be checked for `.clicked()` and `.hovered()`.
pub fn draw_plant_card(ui: &mut egui::Ui, plant: &PowerPlant) -> egui::Response {
    draw_plant_card_ex(ui, plant, false, false)
}

/// Like `draw_plant_card` but shows a discount badge when `discounted` is true.
/// Pass `nominated = true` to render the card with the resource color as background
/// (used to highlight the plant currently up for auction in the market column).
pub fn draw_plant_card_ex(
    ui: &mut egui::Ui,
    plant: &PowerPlant,
    discounted: bool,
    nominated: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(CARD_W, CARD_H), egui::Sense::click());

    if ui.is_rect_visible(rect) {
        paint_card(ui, rect, plant, discounted, nominated);
    }

    response
}

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------

fn paint_card(
    ui: &mut egui::Ui,
    rect: Rect,
    plant: &PowerPlant,
    discounted: bool,
    nominated: bool,
) {
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
            "STEP 3",
            FontId::new(10.0, FontFamily::Monospace),
            theme::NEON_AMBER,
        );
        return;
    }

    let color = kind_color(plant.kind);

    // Background + border (cyan border when discount token; resource color fill when nominated)
    let bg = if nominated { color } else { theme::BG_WIDGET };
    painter.rect_filled(rect, rounding, bg);
    let border_color = if discounted { theme::NEON_CYAN } else { color };
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(1.5, border_color),
        StrokeKind::Inside,
    );

    // Left number box — colored background, plant number centered
    let num_box_w = CARD_H; // square: height × height
    let num_box = Rect::from_min_size(rect.min, Vec2::new(num_box_w, CARD_H));
    painter.rect_filled(
        num_box,
        CornerRadius {
            nw: 3,
            ne: 0,
            sw: 3,
            se: 0,
        },
        color.linear_multiply(0.45),
    );
    painter.text(
        num_box.center(),
        egui::Align2::CENTER_CENTER,
        plant.number.to_string(),
        FontId::new(13.0, FontFamily::Monospace),
        theme::TEXT_BRIGHT,
    );

    // Kind label — left of center, after number box
    let label_x = num_box_w + 6.0 + rect.min.x;
    let label_color = if nominated { Color32::BLACK } else { color };
    painter.text(
        egui::pos2(label_x, rect.center().y),
        egui::Align2::LEFT_CENTER,
        kind_label(plant.kind),
        FontId::new(9.0, FontFamily::Monospace),
        label_color,
    );

    // Stats — right-aligned: "2 → 1" or "→ 1"
    let stats = if plant.kind.needs_resources() {
        format!("{} \u{2192} {}", plant.cost, plant.cities)
    } else {
        format!("\u{2192} {}", plant.cities)
    };
    painter.text(
        egui::pos2(rect.max.x - 5.0, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        stats,
        FontId::new(9.0, FontFamily::Monospace),
        if nominated {
            Color32::BLACK
        } else {
            theme::TEXT_MID
        },
    );

    // Discount token — "$1" in white centered horizontally, raised to upper half
    if discounted {
        ui.painter().text(
            egui::pos2(rect.center().x + 10.0, rect.min.y + CARD_H * 0.10),
            egui::Align2::CENTER_CENTER,
            "$1",
            FontId::new(15.0, FontFamily::Monospace),
            Color32::WHITE,
        );
    }
}
