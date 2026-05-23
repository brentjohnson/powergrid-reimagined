//! Map editor tool for creating Power Grid map TOML files.
//! Usage: maptool <image_path> [output.toml]
//! If output.toml already exists it is loaded automatically.

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::path::PathBuf;

const CITY_RADIUS: f32 = 9.0;
const REGION_COLORS: &[Color32] = &[
    Color32::from_rgb(220, 80, 80),
    Color32::from_rgb(80, 180, 80),
    Color32::from_rgb(80, 130, 220),
    Color32::from_rgb(220, 170, 50),
    Color32::from_rgb(170, 80, 200),
    Color32::from_rgb(60, 190, 190),
    Color32::from_rgb(200, 110, 50),
    Color32::from_rgb(130, 130, 50),
];

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct City {
    id: String,
    name: String,
    region: String,
    #[serde(default = "half")]
    x: f32,
    #[serde(default = "half")]
    y: f32,
}

fn half() -> f32 {
    0.5
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Conn {
    from: String,
    to: String,
    cost: u32,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Default, PartialEq)]
enum Mode {
    #[default]
    Select,
    AddCity,
    Connect,
}

#[derive(Default, PartialEq, Clone)]
enum Selection {
    #[default]
    None,
    City(usize),
    Conn(usize),
}

struct MapEditor {
    map_name: String,
    board_width: f32,
    board_height: f32,
    regions: Vec<String>,
    image_filename: Option<String>,
    cities: Vec<City>,
    connections: Vec<Conn>,

    mode: Mode,
    selection: Selection,
    pending_from: Option<usize>,
    dragging_city: Option<usize>,

    // Edit buffers — kept in sync when selection changes
    edit_id: String,
    edit_name: String,
    edit_region: String,
    edit_cost: String,
    new_region_buf: String,

    image: Option<egui::TextureHandle>,
    output_path: PathBuf,
    status: String,
    city_counter: usize,
}

impl MapEditor {
    fn new(output_path: PathBuf) -> Self {
        Self {
            map_name: "New Map".into(),
            board_width: 1.0,
            board_height: 1.0,
            regions: vec!["region1".into()],
            image_filename: None,
            cities: vec![],
            connections: vec![],
            mode: Mode::Select,
            selection: Selection::None,
            pending_from: None,
            dragging_city: None,
            edit_id: String::new(),
            edit_name: String::new(),
            edit_region: String::new(),
            edit_cost: String::new(),
            new_region_buf: String::new(),
            image: None,
            output_path,
            status: "Use 'Add City' to place cities, then 'Connect' to link them.".into(),
            city_counter: 0,
        }
    }

    fn load_image(&mut self, ctx: &egui::Context, path: &std::path::Path) {
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    rgba.as_raw(),
                );
                self.image =
                    Some(ctx.load_texture("map_bg", color_image, egui::TextureOptions::default()));
                self.image_filename = path.file_name().map(|n| n.to_string_lossy().into_owned());
                self.status = format!("Loaded image {}×{}: {}", w, h, path.display());
            }
            Err(e) => {
                self.status = format!("Failed to load image: {e}");
            }
        }
    }

    fn load_toml(&mut self, path: &std::path::Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;

        #[derive(serde::Deserialize)]
        struct TomlMap {
            name: String,
            #[serde(default)]
            width: f32,
            #[serde(default)]
            height: f32,
            #[serde(default)]
            regions: Vec<String>,
            #[serde(default)]
            image: Option<String>,
            #[serde(default)]
            cities: Vec<City>,
            #[serde(default)]
            connections: Vec<Conn>,
        }

        let m: TomlMap = toml::from_str(&content).map_err(|e| e.to_string())?;
        self.map_name = m.name;
        self.board_width = m.width;
        self.board_height = m.height;
        self.regions = if m.regions.is_empty() {
            vec!["region1".into()]
        } else {
            m.regions
        };
        self.image_filename = m.image;
        self.cities = m.cities;
        self.connections = m.connections;
        self.city_counter = self.cities.len();
        self.status = format!(
            "Loaded {} cities, {} connections from {}",
            self.cities.len(),
            self.connections.len(),
            path.display()
        );
        Ok(())
    }

    fn save_toml(&self) -> Result<(), String> {
        #[derive(serde::Serialize)]
        struct TomlMap<'a> {
            name: &'a str,
            width: f32,
            height: f32,
            #[serde(skip_serializing_if = "Option::is_none")]
            image: Option<&'a str>,
            regions: &'a [String],
            cities: &'a [City],
            connections: &'a [Conn],
        }

        let map = TomlMap {
            name: &self.map_name,
            width: self.board_width,
            height: self.board_height,
            image: self.image_filename.as_deref(),
            regions: &self.regions,
            cities: &self.cities,
            connections: &self.connections,
        };
        let s = toml::to_string_pretty(&map).map_err(|e| e.to_string())?;
        std::fs::write(&self.output_path, s).map_err(|e| e.to_string())?;
        Ok(())
    }

    // ---- helpers ----

    fn slugify(s: &str) -> String {
        let mut out = String::new();
        let mut prev_under = true; // start true to trim leading
        for c in s.chars() {
            if c.is_alphanumeric() {
                out.push(c.to_lowercase().next().unwrap());
                prev_under = false;
            } else if !prev_under {
                out.push('_');
                prev_under = true;
            }
        }
        if out.ends_with('_') {
            out.pop();
        }
        out
    }

    fn unique_id(&self, base: &str) -> String {
        let base = if base.is_empty() { "city" } else { base };
        if !self.cities.iter().any(|c| c.id == base) {
            return base.to_string();
        }
        let mut i = 2u32;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.cities.iter().any(|c| c.id == candidate) {
                return candidate;
            }
            i += 1;
        }
    }

    fn region_color(&self, region: &str) -> Color32 {
        let idx = self.regions.iter().position(|r| r == region).unwrap_or(0);
        REGION_COLORS[idx % REGION_COLORS.len()]
    }

    fn city_screen_pos(city: &City, rect: Rect) -> Pos2 {
        Pos2::new(
            rect.min.x + city.x * rect.width(),
            rect.min.y + city.y * rect.height(),
        )
    }

    fn city_at(cities: &[City], pos: Pos2, rect: Rect) -> Option<usize> {
        cities.iter().enumerate().find_map(|(i, c)| {
            if Self::city_screen_pos(c, rect).distance(pos) <= CITY_RADIUS + 4.0 {
                Some(i)
            } else {
                None
            }
        })
    }

    fn conn_at(&self, pos: Pos2, rect: Rect) -> Option<usize> {
        const THRESH: f32 = 7.0;
        self.connections.iter().enumerate().find_map(|(i, conn)| {
            let a = self.cities.iter().find(|c| c.id == conn.from)?;
            let b = self.cities.iter().find(|c| c.id == conn.to)?;
            let pa = Self::city_screen_pos(a, rect);
            let pb = Self::city_screen_pos(b, rect);
            if seg_dist(pos, pa, pb) <= THRESH {
                Some(i)
            } else {
                None
            }
        })
    }

    // ---- selection sync ----

    fn select_city(&mut self, idx: usize) {
        if idx >= self.cities.len() {
            return;
        }
        let c = &self.cities[idx];
        self.edit_id = c.id.clone();
        self.edit_name = c.name.clone();
        self.edit_region = c.region.clone();
        self.selection = Selection::City(idx);
    }

    fn select_conn(&mut self, idx: usize) {
        if idx >= self.connections.len() {
            return;
        }
        self.edit_cost = self.connections[idx].cost.to_string();
        self.selection = Selection::Conn(idx);
    }

    fn apply_city_edits(&mut self) {
        if let Selection::City(idx) = self.selection {
            if idx >= self.cities.len() {
                return;
            }
            let old_id = self.cities[idx].id.clone();
            let new_id = self.edit_id.trim().to_string();
            let new_name = self.edit_name.trim().to_string();
            let new_region = self.edit_region.trim().to_string();

            if !new_id.is_empty() && new_id != old_id {
                // Check uniqueness
                if !self
                    .cities
                    .iter()
                    .enumerate()
                    .any(|(i, c)| i != idx && c.id == new_id)
                {
                    for conn in &mut self.connections {
                        if conn.from == old_id {
                            conn.from = new_id.clone();
                        }
                        if conn.to == old_id {
                            conn.to = new_id.clone();
                        }
                    }
                    self.cities[idx].id = new_id;
                } else {
                    self.edit_id = old_id; // revert
                    self.status = "ID already in use — reverted.".into();
                }
            }
            if !new_name.is_empty() {
                self.cities[idx].name = new_name;
            }
            if !new_region.is_empty() {
                self.cities[idx].region = new_region;
            }
        }
    }

    fn apply_conn_edits(&mut self) {
        if let Selection::Conn(idx) = self.selection {
            if idx >= self.connections.len() {
                return;
            }
            if let Ok(cost) = self.edit_cost.trim().parse::<u32>() {
                self.connections[idx].cost = cost;
            }
        }
    }

    fn delete_selected(&mut self) {
        match self.selection.clone() {
            Selection::City(idx) if idx < self.cities.len() => {
                let id = self.cities[idx].id.clone();
                self.cities.remove(idx);
                self.connections.retain(|c| c.from != id && c.to != id);
                self.selection = Selection::None;
                self.status = format!("Deleted city '{id}'");
            }
            Selection::Conn(idx) if idx < self.connections.len() => {
                let c = &self.connections[idx];
                let desc = format!("{} → {}", c.from, c.to);
                self.connections.remove(idx);
                self.selection = Selection::None;
                self.status = format!("Deleted connection {desc}");
            }
            _ => {}
        }
    }

    // ---- panel renderers ----

    fn show_left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Map Editor");
        ui.separator();

        // Map name
        ui.label("Map name:");
        ui.text_edit_singleline(&mut self.map_name);

        ui.add_space(4.0);
        ui.separator();

        // Mode
        ui.label("Tool:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, Mode::Select, "Select");
            ui.selectable_value(&mut self.mode, Mode::AddCity, "Add City");
            ui.selectable_value(&mut self.mode, Mode::Connect, "Connect");
        });
        if self.mode != Mode::Connect {
            self.pending_from = None;
        }

        ui.add_space(4.0);
        ui.separator();

        // Regions
        ui.label("Regions:");
        let mut to_remove: Option<usize> = None;
        for (i, region) in self.regions.iter().enumerate() {
            let color = REGION_COLORS[i % REGION_COLORS.len()];
            ui.horizontal(|ui| {
                let (swatch, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                ui.painter().rect_filled(swatch, 2.0, color);
                ui.label(region.as_str());
                if ui.small_button("✕").clicked() && self.regions.len() > 1 {
                    to_remove = Some(i);
                }
            });
        }
        if let Some(i) = to_remove {
            self.regions.remove(i);
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.new_region_buf)
                    .hint_text("new region…")
                    .desired_width(120.0),
            );
            if ui.button("+").clicked() {
                let name = self.new_region_buf.trim().to_string();
                if !name.is_empty() && !self.regions.contains(&name) {
                    self.regions.push(name);
                    self.new_region_buf.clear();
                }
            }
        });

        ui.add_space(4.0);
        ui.separator();

        // Selected item editor
        let sel = self.selection.clone();
        match sel {
            Selection::City(idx) if idx < self.cities.len() => {
                ui.label(egui::RichText::new("City").strong());

                ui.label("ID:");
                let id_r = ui.text_edit_singleline(&mut self.edit_id);
                ui.label("Name:");
                let name_r = ui.text_edit_singleline(&mut self.edit_name);

                ui.label("Region:");
                let regions = self.regions.clone();
                egui::ComboBox::from_id_salt("region_combo")
                    .selected_text(self.edit_region.as_str())
                    .show_ui(ui, |ui| {
                        for r in &regions {
                            if ui
                                .selectable_label(self.edit_region == *r, r.as_str())
                                .clicked()
                            {
                                self.edit_region = r.clone();
                                self.apply_city_edits();
                            }
                        }
                    });

                if id_r.lost_focus() || name_r.lost_focus() {
                    self.apply_city_edits();
                }

                if ui.button("Auto-ID from name").clicked() {
                    let base = Self::slugify(&self.edit_name);
                    self.edit_id = self.unique_id(&base);
                    self.apply_city_edits();
                }

                if ui.button("🗑 Delete city").clicked() {
                    self.delete_selected();
                }
            }
            Selection::Conn(idx) if idx < self.connections.len() => {
                ui.label(egui::RichText::new("Connection").strong());
                let (from, to) = {
                    let c = &self.connections[idx];
                    (c.from.clone(), c.to.clone())
                };
                ui.label(format!("{from} → {to}"));
                ui.label("Cost:");
                let cost_r = ui.text_edit_singleline(&mut self.edit_cost);
                if cost_r.lost_focus() {
                    self.apply_conn_edits();
                }
                if ui.button("🗑 Delete connection").clicked() {
                    self.delete_selected();
                }
            }
            _ => {
                ui.label(egui::RichText::new("Nothing selected").weak());
                match self.mode {
                    Mode::AddCity => {
                        ui.label("Click the map to place a city.");
                    }
                    Mode::Connect => {
                        if let Some(from_idx) = self.pending_from {
                            if from_idx < self.cities.len() {
                                ui.label(format!("From: {}", self.cities[from_idx].name));
                                ui.label("Click another city to connect.");
                                if ui.button("Cancel").clicked() {
                                    self.pending_from = None;
                                }
                            }
                        } else {
                            ui.label("Click a city to start a connection.");
                        }
                    }
                    Mode::Select => {
                        ui.label("Click a city or connection.");
                    }
                }
            }
        }

        ui.add_space(4.0);
        ui.separator();

        // Save
        if ui.button("💾  Save TOML").clicked() {
            match self.save_toml() {
                Ok(()) => {
                    self.status = format!("Saved → {}", self.output_path.display());
                }
                Err(e) => {
                    self.status = format!("Save error: {e}");
                }
            }
        }
        ui.label(
            egui::RichText::new(format!(
                "{} cities · {} connections",
                self.cities.len(),
                self.connections.len()
            ))
            .small(),
        );

        ui.add_space(4.0);
        ui.separator();

        // City list
        ui.label("Cities:");
        let sel_city = if let Selection::City(i) = self.selection {
            Some(i)
        } else {
            None
        };
        let mut to_select_city: Option<usize> = None;
        egui::ScrollArea::vertical()
            .id_salt("city_list")
            .max_height(160.0)
            .show(ui, |ui| {
                for (i, city) in self.cities.iter().enumerate() {
                    let label = format!("{} ({})", city.name, city.region);
                    if ui.selectable_label(sel_city == Some(i), label).clicked() {
                        to_select_city = Some(i);
                    }
                }
            });
        if let Some(i) = to_select_city {
            self.select_city(i);
        }

        ui.add_space(2.0);
        ui.label("Connections:");
        let sel_conn = if let Selection::Conn(i) = self.selection {
            Some(i)
        } else {
            None
        };
        let mut to_select_conn: Option<usize> = None;
        egui::ScrollArea::vertical()
            .id_salt("conn_list")
            .max_height(120.0)
            .show(ui, |ui| {
                for (i, conn) in self.connections.iter().enumerate() {
                    let label = format!("{} ↔ {} ({})", conn.from, conn.to, conn.cost);
                    if ui.selectable_label(sel_conn == Some(i), label).clicked() {
                        to_select_conn = Some(i);
                    }
                }
            });
        if let Some(i) = to_select_conn {
            self.select_conn(i);
        }
    }

    fn show_canvas(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
        let rect = response.rect;

        // Background
        if let Some(tex) = &self.image {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        } else {
            painter.rect_filled(rect, 0.0, Color32::from_rgb(20, 28, 40));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No image loaded\nPass image path as: maptool <image>",
                egui::FontId::proportional(16.0),
                Color32::from_gray(140),
            );
        }

        // --- input handling ---

        // Drag to move city (Select mode only)
        if self.mode == Mode::Select {
            if response.dragged() {
                // Identify dragging target on first frame (dragging_city not yet set)
                if self.dragging_city.is_none() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.dragging_city = Self::city_at(&self.cities, pos, rect);
                        if let Some(idx) = self.dragging_city {
                            self.select_city(idx);
                        }
                    }
                }
                if let Some(idx) = self.dragging_city {
                    let d = response.drag_delta();
                    let c = &mut self.cities[idx];
                    c.x = (c.x + d.x / rect.width()).clamp(0.0, 1.0);
                    c.y = (c.y + d.y / rect.height()).clamp(0.0, 1.0);
                }
            } else {
                self.dragging_city = None;
            }
        }

        // Click
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let hit_city = Self::city_at(&self.cities, pos, rect);

                match self.mode {
                    Mode::AddCity => {
                        let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                        let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
                        self.city_counter += 1;
                        let base = format!("city{}", self.city_counter);
                        let id = self.unique_id(&base);
                        let region = self.regions.first().cloned().unwrap_or_default();
                        let name = format!("City {}", self.city_counter);
                        self.cities.push(City {
                            id: id.clone(),
                            name,
                            region,
                            x: nx,
                            y: ny,
                        });
                        let new_idx = self.cities.len() - 1;
                        self.select_city(new_idx);
                        self.status = format!("Added '{id}' — edit name/ID in panel");
                    }
                    Mode::Select => {
                        if let Some(idx) = hit_city {
                            self.select_city(idx);
                        } else {
                            let hit_conn = self.conn_at(pos, rect);
                            if let Some(idx) = hit_conn {
                                self.select_conn(idx);
                            } else {
                                self.selection = Selection::None;
                            }
                        }
                    }
                    Mode::Connect => {
                        if let Some(idx) = hit_city {
                            if let Some(from_idx) = self.pending_from {
                                if from_idx != idx {
                                    let from_id = self.cities[from_idx].id.clone();
                                    let to_id = self.cities[idx].id.clone();
                                    let existing = self.connections.iter().position(|c| {
                                        (c.from == from_id && c.to == to_id)
                                            || (c.from == to_id && c.to == from_id)
                                    });
                                    if let Some(ci) = existing {
                                        self.select_conn(ci);
                                        self.status = format!(
                                            "{from_id} ↔ {to_id} already connected — selected existing"
                                        );
                                    } else {
                                        self.connections.push(Conn {
                                            from: from_id.clone(),
                                            to: to_id.clone(),
                                            cost: 10,
                                        });
                                        let ci = self.connections.len() - 1;
                                        self.select_conn(ci);
                                        self.status = format!(
                                            "Connected {from_id} → {to_id} (set cost in panel)"
                                        );
                                    }
                                    self.pending_from = None;
                                }
                            } else {
                                self.pending_from = Some(idx);
                                self.status = format!(
                                    "From '{}' — click destination city",
                                    self.cities[idx].name
                                );
                            }
                        }
                    }
                }
            }
        }

        // --- drawing ---

        // Connections
        for (i, conn) in self.connections.iter().enumerate() {
            let Some(ca) = self.cities.iter().find(|c| c.id == conn.from) else {
                continue;
            };
            let Some(cb) = self.cities.iter().find(|c| c.id == conn.to) else {
                continue;
            };
            let pa = Self::city_screen_pos(ca, rect);
            let pb = Self::city_screen_pos(cb, rect);
            let selected = self.selection == Selection::Conn(i);
            let (line_color, width) = if selected {
                (Color32::YELLOW, 3.0f32)
            } else {
                (Color32::from_rgba_unmultiplied(220, 220, 220, 160), 1.5)
            };
            painter.line_segment([pa, pb], Stroke::new(width, line_color));

            // Cost label at midpoint with dark background for legibility
            let mid = Pos2::new((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0);
            let label = conn.cost.to_string();
            let font = egui::FontId::monospace(11.0);
            for (dx, dy) in [(-1.0f32, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                painter.text(
                    Pos2::new(mid.x + dx, mid.y + dy),
                    egui::Align2::CENTER_CENTER,
                    &label,
                    font.clone(),
                    Color32::BLACK,
                );
            }
            painter.text(
                mid,
                egui::Align2::CENTER_CENTER,
                &label,
                font,
                Color32::WHITE,
            );
        }

        // Pending connection ghost line
        if self.mode == Mode::Connect {
            if let Some(from_idx) = self.pending_from {
                if from_idx < self.cities.len() {
                    let fp = Self::city_screen_pos(&self.cities[from_idx], rect);
                    let mouse = ui.input(|i| i.pointer.hover_pos()).unwrap_or(fp);
                    painter.line_segment(
                        [fp, mouse],
                        Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 220, 50, 140)),
                    );
                    ui.ctx().request_repaint();
                }
            }
        }

        // Cities
        for (i, city) in self.cities.iter().enumerate() {
            let pos = Self::city_screen_pos(city, rect);
            let selected = self.selection == Selection::City(i);
            let is_pending = self.pending_from == Some(i);
            let fill = self.region_color(&city.region);
            let (ring_color, ring_w) = if selected || is_pending {
                (Color32::YELLOW, 3.0f32)
            } else {
                (Color32::from_black_alpha(200), 1.5)
            };

            painter.circle(pos, CITY_RADIUS, fill, Stroke::new(ring_w, ring_color));

            // Name label above the dot
            let lp = Pos2::new(pos.x, pos.y - CITY_RADIUS - 3.0);
            let lbl = &city.name;
            let font = egui::FontId::proportional(12.0);
            for (dx, dy) in [(-1.0f32, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                painter.text(
                    Pos2::new(lp.x + dx, lp.y + dy),
                    egui::Align2::CENTER_BOTTOM,
                    lbl,
                    font.clone(),
                    Color32::BLACK,
                );
            }
            painter.text(lp, egui::Align2::CENTER_BOTTOM, lbl, font, Color32::WHITE);
        }
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for MapEditor {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .max_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.show_left_panel(ui);
                });
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(egui::RichText::new(&self.status).small());
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_canvas(ui);
        });
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab: Vec2 = b - a;
    let ap: Vec2 = p - a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-6 {
        return ap.length();
    }
    let t = (ap.dot(ab) / len_sq).clamp(0.0, 1.0);
    let closest = a + t * ab;
    (p - closest).length()
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    let mut image_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("map.toml");
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = PathBuf::from(&args[i]);
                }
            }
            arg if !arg.starts_with('-') => {
                if image_path.is_none() {
                    image_path = Some(PathBuf::from(arg));
                } else {
                    output_path = PathBuf::from(arg);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let load_existing = output_path.exists();
    let op_for_load = output_path.clone();

    eframe::run_native(
        "Power Grid Map Editor",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1400.0, 900.0])
                .with_title("Power Grid Map Editor"),
            ..Default::default()
        },
        Box::new(move |cc| {
            let mut app = MapEditor::new(output_path);

            if load_existing {
                if let Err(e) = app.load_toml(&op_for_load) {
                    app.status = format!("Could not load existing TOML: {e}");
                }
            }

            if let Some(ref path) = image_path {
                app.load_image(&cc.egui_ctx, path);
            }

            Ok(Box::new(app))
        }),
    )
}
