use egui::{
    Color32, CornerRadius, FontFamily, FontId, Layout, Rect, RichText, Stroke, StrokeKind, Vec2,
};
use powergrid_core::types::{PlantKind, PowerPlant, Resource};

use crate::{
    theme,
    ui::helpers::{resource_color, resource_image},
};

const RES_ICON: f32 = 22.0;
const RES_ICON_GAP: f32 = 1.5;

// ---------------------------------------------------------------------------
// Card dimensions
// ---------------------------------------------------------------------------

pub const CARD_W: f32 = 150.0;
pub const CARD_H: f32 = 34.0;

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
        FontId::new(16.0, FontFamily::Monospace),
        theme::TEXT_BRIGHT,
    );

    // Stats — right side: resource icons (cost × kind) then "→ cities"
    let text_color = if nominated {
        Color32::BLACK
    } else {
        theme::TEXT_MID
    };
    let arrow_text = format!("\u{2192} {}", plant.cities);
    let is_gas_oil = plant.kind == PlantKind::GasOrOil;

    let right_rect = Rect::from_min_max(egui::pos2(rect.min.x + num_box_w, rect.min.y), rect.max);
    ui.scope_builder(egui::UiBuilder::new().max_rect(right_rect), |ui| {
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = RES_ICON_GAP;
            ui.label(
                RichText::new(&arrow_text)
                    .monospace()
                    .size(11.0)
                    .color(text_color),
            );
            if is_gas_oil {
                // Each slot is a single forward-slash split icon: gas (top) / oil (bottom)
                for _ in 0..plant.cost {
                    let (slot_rect, _) =
                        ui.allocate_exact_size(Vec2::splat(RES_ICON), egui::Sense::hover());
                    draw_split_icon(ui, slot_rect, nominated);
                }
            } else {
                let res_types = plant.kind.resources();
                if !res_types.is_empty() {
                    for i in (0..plant.cost as usize).rev() {
                        let res = res_types[i % res_types.len()];
                        let tint = if nominated {
                            Color32::BLACK
                        } else {
                            resource_color(res)
                        };
                        ui.add(
                            egui::Image::new(resource_image(res))
                                .tint(tint)
                                .fit_to_exact_size(Vec2::new(RES_ICON, RES_ICON)),
                        );
                    }
                }
            }
        });
    });

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

// ---------------------------------------------------------------------------
// Hybrid split icon
// ---------------------------------------------------------------------------

/// Draw a forward-slash `/` split icon in `rect`.
///
/// The upper-left triangle (above the diagonal) shows the **gas** icon; the
/// lower-right triangle (below the diagonal) shows the **oil** icon.  Vertices:
///   gas  triangle: TL, TR, BL   (UVs 0,0 / 1,0 / 0,1)
///   oil  triangle: TR, BR, BL   (UVs 1,0 / 1,1 / 0,1)
///
/// When `nominated` both halves are tinted black; otherwise each uses its
/// resource color (matching the behavior of single-resource icons).
fn draw_split_icon(ui: &mut egui::Ui, rect: Rect, nominated: bool) {
    let gas_tint = if nominated {
        Color32::BLACK
    } else {
        resource_color(Resource::Gas)
    };
    let oil_tint = if nominated {
        Color32::BLACK
    } else {
        resource_color(Resource::Oil)
    };

    let tl = rect.left_top();
    let tr = rect.right_top();
    let bl = rect.left_bottom();
    let br = rect.right_bottom();

    let gas_tex =
        egui::Image::new(resource_image(Resource::Gas)).load_for_size(ui.ctx(), rect.size());
    let oil_tex =
        egui::Image::new(resource_image(Resource::Oil)).load_for_size(ui.ctx(), rect.size());

    let painter = ui.painter_at(rect);

    match (gas_tex, oil_tex) {
        (
            Ok(egui::load::TexturePoll::Ready { texture: gas }),
            Ok(egui::load::TexturePoll::Ready { texture: oil }),
        ) => {
            // Gas: upper-left triangle — TL, TR, BL
            let mut mesh = egui::Mesh::with_texture(gas.id);
            mesh.vertices.push(egui::epaint::Vertex {
                pos: tl,
                uv: egui::pos2(0.0, 0.0),
                color: gas_tint,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: tr,
                uv: egui::pos2(1.0, 0.0),
                color: gas_tint,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: bl,
                uv: egui::pos2(0.0, 1.0),
                color: gas_tint,
            });
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));

            // Oil: lower-right triangle — TR, BR, BL
            let mut mesh = egui::Mesh::with_texture(oil.id);
            mesh.vertices.push(egui::epaint::Vertex {
                pos: tr,
                uv: egui::pos2(1.0, 0.0),
                color: oil_tint,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: br,
                uv: egui::pos2(1.0, 1.0),
                color: oil_tint,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: bl,
                uv: egui::pos2(0.0, 1.0),
                color: oil_tint,
            });
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));
        }
        _ => {
            // Textures not yet loaded — fall back to flat colored triangles so the
            // slot still conveys the gas/oil split on the first frame.
            let mut mesh = egui::Mesh::default();
            mesh.colored_vertex(tl, gas_tint);
            mesh.colored_vertex(tr, gas_tint);
            mesh.colored_vertex(bl, gas_tint);
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));

            let mut mesh = egui::Mesh::default();
            mesh.colored_vertex(tr, oil_tint);
            mesh.colored_vertex(br, oil_tint);
            mesh.colored_vertex(bl, oil_tint);
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));
        }
    }
}
