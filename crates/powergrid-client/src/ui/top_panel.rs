use bevy_egui::egui;
use egui::{Align2, Color32, FontId, Rect, RichText, Sense, Ui};
use powergrid_core::{types::ResourceMarket, GameState};

use crate::theme;

use super::helpers::{dim_color, section_header};
use super::phase_tracker::phase_tracker;

pub(super) fn top_panel_contents(ui: &mut Ui, gs: GameState) {
    ui.horizontal(|ui| {
        // Round header
        ui.vertical(|ui| {
            theme::neon_frame_bright().show(ui, |ui| {
                ui.label(
                    RichText::new(format!("ROUND {}", gs.round))
                        .color(theme::NEON_CYAN)
                        .monospace(),
                );
            });
        });

        ui.add_space(8.0);

        // Phase tracker
        phase_tracker(ui, &gs);

        ui.add_space(8.0);

        // Resource market
        ui.vertical(|ui| {
            section_header(ui, "RESOURCE MARKET");
            theme::neon_frame().show(ui, |ui| {
                resource_market_grid(ui, &gs.resources);
            });
        });
    });
}

fn resource_market_grid(ui: &mut Ui, market: &ResourceMarket) {
    const SQ: f32 = 8.0; // square size (height and base width)
    const INNER_GAP: f32 = 1.0; // gap between squares within a COG price group
    const GROUP_GAP: f32 = 5.0; // gap between price groups (COG) or uranium slots
    const URAN_W: f32 = 11.0; // uranium slot width (slightly wider to fit price labels)
    const LABEL_W: f32 = 30.0; // width of row labels (COAL, OIL, etc.)
    const HEADER_H: f32 = 12.0; // height of price label header row
    const ROW_H: f32 = SQ;
    const ROW_GAP: f32 = 2.0; // gap between resource rows
    const SECTION_GAP: f32 = 6.0; // gap between COG section and uranium section

    // COG: 8 price groups × 3 slots each
    let cog_group_w = 3.0 * SQ + 2.0 * INNER_GAP; // 26px per group
    let cog_total_w = 8.0 * cog_group_w + 7.0 * GROUP_GAP; // 208+35 = 243px

    // Uranium: 12 individual slots, each treated as its own group
    let uran_total_w = 12.0 * URAN_W + 11.0 * GROUP_GAP; // 132+55 = 187px

    let content_w = cog_total_w.max(uran_total_w);
    let total_w = LABEL_W + content_w;

    // Total height: COG header + 3 rows + uranium header + uranium row
    let total_h = HEADER_H
        + ROW_GAP
        + ROW_H
        + ROW_GAP
        + ROW_H
        + ROW_GAP
        + ROW_H
        + SECTION_GAP
        + HEADER_H
        + ROW_GAP
        + ROW_H;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(total_w, total_h), Sense::hover());

    if !ui.is_rect_visible(rect) {
        return;
    }

    let painter = ui.painter();
    let ox = rect.min.x;
    let oy = rect.min.y;

    let coal_color = Color32::from_rgb(150, 100, 55);
    let oil_color = Color32::from_rgb(110, 110, 140);
    let garb_color = Color32::from_rgb(200, 170, 20);
    let uran_color = Color32::from_rgb(200, 30, 30);

    // ── COG price header ($1 on left = cheapest, $8 on right = most expensive) ──
    for g in 0..8usize {
        let price = g + 1;
        let gx = ox + LABEL_W + g as f32 * (cog_group_w + GROUP_GAP);
        let cx = gx + cog_group_w / 2.0;
        painter.text(
            egui::pos2(cx, oy + HEADER_H / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(9.0),
            theme::TEXT_DIM,
        );
    }

    // ── COG rows (coal, oil, garbage) ──
    // Price table: index 0 = most expensive ($8), index 23 = cheapest ($1).
    // `count` resources occupy indices 0..(count-1).
    // Display pos 0 (leftmost, $1) maps to array index (total-1 - display_pos).
    // Slot is filled when array_idx < count.
    let cog_rows: &[(&str, Color32, u8, usize)] = &[
        ("COAL", coal_color, market.coal, 24),
        ("OIL", oil_color, market.oil, 24),
        ("GARB", garb_color, market.garbage, 24),
    ];

    for (i, (label, color, count, total)) in cog_rows.iter().enumerate() {
        let row_y = oy + HEADER_H + ROW_GAP + i as f32 * (ROW_H + ROW_GAP);

        painter.text(
            egui::pos2(ox + LABEL_W - 2.0, row_y + ROW_H / 2.0),
            Align2::RIGHT_CENTER,
            *label,
            FontId::monospace(9.0),
            *color,
        );

        for g in 0..8usize {
            let gx = ox + LABEL_W + g as f32 * (cog_group_w + GROUP_GAP);
            for s in 0..3usize {
                let display_pos = g * 3 + s;
                let array_idx = (total - 1) - display_pos;
                let filled = array_idx < *count as usize;
                let sq_x = gx + s as f32 * (SQ + INNER_GAP);
                let sq_rect = Rect::from_min_size(egui::pos2(sq_x, row_y), egui::vec2(SQ, ROW_H));
                painter.rect_filled(
                    sq_rect,
                    1.0,
                    if filled { *color } else { dim_color(*color) },
                );
            }
        }
    }

    // ── Uranium section ──
    // Price table (cheap→expensive display): [1,2,3,4,5,6,7,8,10,12,14,16]
    // Array index 0 = $16, index 11 = $1.
    // Display pos i (0=cheapest) maps to array index 11-i.
    let usec_y = oy + HEADER_H + ROW_GAP + 3.0 * (ROW_H + ROW_GAP) + SECTION_GAP;
    let uran_prices: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];

    for (i, &price) in uran_prices.iter().enumerate() {
        let sx = ox + LABEL_W + i as f32 * (URAN_W + GROUP_GAP);
        let cx = sx + URAN_W / 2.0;
        painter.text(
            egui::pos2(cx, usec_y + HEADER_H / 2.0),
            Align2::CENTER_CENTER,
            format!("${price}"),
            FontId::monospace(8.0),
            theme::TEXT_DIM,
        );
    }

    let uran_row_y = usec_y + HEADER_H + ROW_GAP;

    painter.text(
        egui::pos2(ox + LABEL_W - 2.0, uran_row_y + ROW_H / 2.0),
        Align2::RIGHT_CENTER,
        "URAN",
        FontId::monospace(9.0),
        uran_color,
    );

    for i in 0..12usize {
        let array_idx = 11 - i; // display 0 ($1) maps to array 11
        let filled = array_idx < market.uranium as usize;
        let sx = ox + LABEL_W + i as f32 * (URAN_W + GROUP_GAP);
        let sq_rect = Rect::from_min_size(egui::pos2(sx, uran_row_y), egui::vec2(URAN_W, ROW_H));
        painter.rect_filled(
            sq_rect,
            1.0,
            if filled {
                uran_color
            } else {
                dim_color(uran_color)
            },
        );
    }
}
