use egui::{Color32, CornerRadius, FontFamily, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use powergrid_core::types::{PlantKind, PowerPlant, Resource};

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

/// Draw an empty plant slot (CARD_W × CARD_H) as a subtle grey outline, reserving the
/// same space as a real plant card so player panels don't resize when plants are bought.
pub fn draw_plant_placeholder(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(CARD_W, CARD_H), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::same(3),
            Stroke::new(1.0, Color32::from_gray(40)),
            StrokeKind::Inside,
        );
    }
    response
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

/// Side length of a square "full" power-plant card used in the plant market.
pub const FULL_CARD_SIZE: f32 = 96.0;

/// Draw a full square power-plant card (`FULL_CARD_SIZE × FULL_CARD_SIZE`).
/// Features a large fuel icon in the centre, plant number top-left, and a
/// bottom row of per-resource-unit icons → cities → house outline.
/// Returns the egui Response (click + hover).
pub fn draw_plant_card_full(
    ui: &mut egui::Ui,
    plant: &PowerPlant,
    discounted: bool,
    nominated: bool,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::splat(FULL_CARD_SIZE), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        paint_full_card(ui, rect, plant, discounted, nominated);
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

    // Background + border (resource color fill when nominated)
    let bg = if nominated { color } else { theme::BG_WIDGET };
    painter.rect_filled(rect, rounding, bg);
    painter.rect_stroke(rect, rounding, Stroke::new(1.5, color), StrokeKind::Inside);

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

    // Discount token — "$1" in white centered on the card
    if discounted {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "$1",
            FontId::new(45.0, FontFamily::Monospace),
            Color32::WHITE,
        );
    }
}

// ---------------------------------------------------------------------------
// Full square card
// ---------------------------------------------------------------------------

fn paint_full_card(
    ui: &mut egui::Ui,
    rect: Rect,
    plant: &PowerPlant,
    discounted: bool,
    nominated: bool,
) {
    const PAD: f32 = 4.0;
    const BADGE_SIZE: f32 = 22.0;
    const BOTTOM_H: f32 = 22.0;
    const CENTER_ICON_SIZE: f32 = 50.0;

    let painter = ui.painter_at(rect);
    let rounding = CornerRadius::same(4);

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
    let bg = if nominated { color } else { theme::BG_WIDGET };

    painter.rect_filled(rect, rounding, bg);
    painter.rect_stroke(rect, rounding, Stroke::new(1.5, color), StrokeKind::Inside);

    // ── Number badge (top-left) ───────────────────────────────────────────────
    let badge_rect = Rect::from_min_size(rect.min + Vec2::splat(PAD), Vec2::splat(BADGE_SIZE));
    painter.rect_filled(
        badge_rect,
        CornerRadius::same(3),
        color.linear_multiply(0.45),
    );
    painter.text(
        badge_rect.center(),
        egui::Align2::CENTER_CENTER,
        plant.number.to_string(),
        FontId::new(12.0, FontFamily::Monospace),
        theme::TEXT_BRIGHT,
    );

    // ── Centre fuel icon ──────────────────────────────────────────────────────
    // Vertically centred in the area above the bottom row.
    let center_area_mid_y = (rect.min.y + (rect.max.y - BOTTOM_H)) / 2.0;
    let center_icon_rect = Rect::from_center_size(
        egui::pos2(rect.center().x, center_area_mid_y),
        Vec2::splat(CENTER_ICON_SIZE),
    );
    paint_fuel(ui, center_icon_rect, plant.kind, None);
    // painter is still valid — Painter holds Arc<Context>, not a &Ui borrow.

    // ── Bottom row: fuel icons → cities → house outline ───────────────────────
    let text_color = if nominated {
        Color32::BLACK
    } else {
        theme::TEXT_MID
    };
    let bright_color = if nominated {
        Color32::BLACK
    } else {
        theme::TEXT_BRIGHT
    };

    // Dynamically shrink per-unit icons so they never overflow the row.
    // Reserve ~35 px for the arrow glyph, cities digit(s), and house outline.
    const FIXED_BOTTOM_W: f32 = 35.0;
    let available_icon_w = (rect.width() - 2.0 * PAD - FIXED_BOTTOM_W).max(0.0);
    let icon_s: f32 = if plant.kind.needs_resources() && plant.cost > 0 {
        (available_icon_w / plant.cost as f32).clamp(6.0, 13.0)
    } else {
        13.0
    };

    let row_y = rect.max.y - BOTTOM_H / 2.0;
    let icon_top = row_y - icon_s / 2.0;
    let house_r = 5.5_f32;

    // Right-justify: compute total width then start from the right edge.
    let fuel_w = if plant.kind.needs_resources() {
        plant.cost as f32 * (icon_s + 1.0) + 1.0
    } else {
        0.0
    };
    // arrow(10) + cities(8) + gap(2) + house diameter(2*house_r)
    let total_w = fuel_w + 10.0 + 8.0 + 2.0 + house_r * 2.0;
    let mut x = rect.max.x - PAD - total_w;

    let fuel_tint_override = if nominated {
        Some(Color32::BLACK)
    } else {
        None
    };
    if plant.kind.needs_resources() {
        for _ in 0..plant.cost {
            let icon_rect = Rect::from_min_size(egui::pos2(x, icon_top), Vec2::splat(icon_s));
            paint_fuel(ui, icon_rect, plant.kind, fuel_tint_override);
            x += icon_s + 1.0;
        }
        x += 1.0; // extra gap before arrow
    }

    // Arrow
    painter.text(
        egui::pos2(x, row_y),
        egui::Align2::LEFT_CENTER,
        "\u{2192}",
        FontId::new(9.0, FontFamily::Monospace),
        text_color,
    );
    x += 10.0;

    // Cities count
    painter.text(
        egui::pos2(x, row_y),
        egui::Align2::LEFT_CENTER,
        plant.cities.to_string(),
        FontId::new(9.0, FontFamily::Monospace),
        bright_color,
    );
    x += 8.0;

    // House glyph — outline only (TRANSPARENT fill), matching the map style
    let house_center = egui::pos2(x + 2.0 + house_r, row_y - 1.0);
    let pts = crate::map_panel::house_points(house_center, house_r);
    painter.add(egui::Shape::convex_polygon(
        pts,
        Color32::TRANSPARENT,
        Stroke::new(1.2, text_color),
    ));

    // Discount badge
    if discounted {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "$1",
            FontId::new(36.0, FontFamily::Monospace),
            Color32::WHITE,
        );
    }
}

// ---------------------------------------------------------------------------
// Fuel icon rendering helpers
// ---------------------------------------------------------------------------

/// Paint a fuel icon scaled to `rect` for the given `PlantKind`.
/// Wind → wind turbine SVG; GasOrOil → diagonal split; others → tinted SVG.
fn paint_fuel(ui: &mut egui::Ui, rect: Rect, kind: PlantKind, tint_override: Option<Color32>) {
    match kind {
        PlantKind::Wind => {
            egui::Image::new(egui::include_image!("../assets/wind-svgrepo-com.svg"))
                .tint(tint_override.unwrap_or(theme::CARD_WIND))
                .paint_at(ui, rect);
        }
        PlantKind::GasOrOil => {
            paint_hybrid_diagonal(ui, rect, tint_override);
        }
        _ => {
            let resources = kind.resources();
            if let Some(&res) = resources.first() {
                egui::Image::new(crate::ui::helpers::resource_image(res))
                    .tint(tint_override.unwrap_or_else(|| crate::ui::helpers::resource_color(res)))
                    .paint_at(ui, rect);
            }
        }
    }
}

/// Gas/Oil hybrid: single icon split diagonally like a "/" slash.
/// Gas occupies the top-left triangle; oil the bottom-right triangle.
/// Uses GPU mesh triangles so the split is a true diagonal, not a rect clip.
/// If a texture is still `Pending` on the first frame it is skipped silently
/// (SVGs are cached and ready from frame 2 onward).
fn paint_hybrid_diagonal(ui: &mut egui::Ui, rect: Rect, tint_override: Option<Color32>) {
    use egui::load::TexturePoll;

    let gas_color =
        tint_override.unwrap_or_else(|| crate::ui::helpers::resource_color(Resource::Gas));
    let oil_color =
        tint_override.unwrap_or_else(|| crate::ui::helpers::resource_color(Resource::Oil));

    let gas_src = egui::Image::new(crate::ui::helpers::resource_image(Resource::Gas));
    let oil_src = egui::Image::new(crate::ui::helpers::resource_image(Resource::Oil));

    let tl = rect.left_top();
    let tr = rect.right_top();
    let bl = rect.left_bottom();
    let br = rect.right_bottom();

    // Gas: top-left triangle (TL → TR → BL), above the "/" diagonal
    if let Ok(TexturePoll::Ready { texture }) = gas_src.load_for_size(ui.ctx(), rect.size()) {
        let mut mesh = egui::epaint::Mesh::with_texture(texture.id);
        mesh.vertices.extend([
            egui::epaint::Vertex {
                pos: tl,
                uv: Pos2::new(0.0, 0.0),
                color: gas_color,
            },
            egui::epaint::Vertex {
                pos: tr,
                uv: Pos2::new(1.0, 0.0),
                color: gas_color,
            },
            egui::epaint::Vertex {
                pos: bl,
                uv: Pos2::new(0.0, 1.0),
                color: gas_color,
            },
        ]);
        mesh.indices.extend([0, 1, 2]);
        ui.painter_at(rect).add(egui::Shape::mesh(mesh));
    }

    // Oil: bottom-right triangle (TR → BR → BL), below the "/" diagonal
    if let Ok(TexturePoll::Ready { texture }) = oil_src.load_for_size(ui.ctx(), rect.size()) {
        let mut mesh = egui::epaint::Mesh::with_texture(texture.id);
        mesh.vertices.extend([
            egui::epaint::Vertex {
                pos: tr,
                uv: Pos2::new(1.0, 0.0),
                color: oil_color,
            },
            egui::epaint::Vertex {
                pos: br,
                uv: Pos2::new(1.0, 1.0),
                color: oil_color,
            },
            egui::epaint::Vertex {
                pos: bl,
                uv: Pos2::new(0.0, 1.0),
                color: oil_color,
            },
        ]);
        mesh.indices.extend([0, 1, 2]);
        ui.painter_at(rect).add(egui::Shape::mesh(mesh));
    }
}
