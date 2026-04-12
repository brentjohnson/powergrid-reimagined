use iced::{
    widget::{button, canvas, column, container, row, scrollable, stack, text},
    Color, ContentFit, Element, Length, Point, Rectangle, Renderer, Size, Theme,
};
use powergrid_core::map::MapData;
use std::{env, fs, path::PathBuf};

// ---------------------------------------------------------------------------
// A positioned resource slot (in-memory working state)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Slot {
    resource: String,
    index: usize,
    /// Fractional (0..1) position on the map image. None = not yet placed.
    pos: Option<(f32, f32)>,
}

impl Slot {
    fn label(&self) -> String {
        format!("{} {}", self.resource, self.index)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Message {
    /// Cursor moved over image: carries (x_pct, y_pct).
    CursorMoved(Point),
    /// Left-click on image: set position of selected slot.
    Clicked(Point),
    /// User selected a slot in the sidebar list.
    SelectSlot(usize),
    /// Save coordinates back to the TOML file.
    Save,
}

struct App {
    image_handle: iced::widget::image::Handle,
    img_w: f32,
    img_h: f32,
    toml_path: PathBuf,
    /// Original file content up to (but not including) the first [[resource_slots]] block.
    toml_prefix: String,
    slots: Vec<Slot>,
    /// Index into `slots` of the currently selected slot, if any.
    selected: Option<usize>,
    cursor_pct: Option<(f32, f32)>,
    status_msg: String,
}

impl App {
    fn new() -> (Self, iced::Task<Message>) {
        let mut args = env::args().skip(1);
        let image_path = args
            .next()
            .expect("Usage: map-tool <image_path> <toml_path> [width] [height]");
        let toml_path: PathBuf = args
            .next()
            .expect("Usage: map-tool <image_path> <toml_path> [width] [height]")
            .into();
        let img_w: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1536.0);
        let img_h: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2048.0);

        let image_handle = iced::widget::image::Handle::from_path(&image_path);

        let raw = fs::read_to_string(&toml_path)
            .unwrap_or_else(|e| panic!("Cannot read {}: {e}", toml_path.display()));

        // Split off anything at or after the first [[resource_slots]] block so
        // we can regenerate that section on save while preserving everything else.
        let toml_prefix = if let Some(pos) = raw.find("[[resource_slots]]") {
            raw[..pos].trim_end().to_string()
        } else {
            raw.trim_end().to_string()
        };

        let map_data: MapData = toml::from_str(&raw)
            .unwrap_or_else(|e| panic!("Cannot parse {}: {e}", toml_path.display()));

        // Build slot list.  Pre-populate positions from any existing entries in the file.
        // We need to know all resources and their slot counts.  Derive from price table
        // lengths as reflected in the existing resource_slots, or fall back to the
        // standard counts if none are present.
        let mut slots = build_slot_list(&map_data);

        // Populate existing positions.
        for rs in &map_data.resource_slots {
            if let Some(slot) = slots
                .iter_mut()
                .find(|s| s.resource == rs.resource && s.index == rs.index)
            {
                slot.pos = Some((rs.x, rs.y));
            }
        }

        let placed = slots.iter().filter(|s| s.pos.is_some()).count();
        let total = slots.len();
        let status_msg =
            format!("{placed}/{total} slots placed. Select a slot, then click the map.");

        (
            Self {
                image_handle,
                img_w,
                img_h,
                toml_path,
                toml_prefix,
                slots,
                selected: None,
                cursor_pct: None,
                status_msg,
            },
            iced::Task::none(),
        )
    }

    fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::CursorMoved(pct) => {
                self.cursor_pct = Some((pct.x, pct.y));
            }
            Message::Clicked(pct) => {
                if let Some(idx) = self.selected {
                    self.slots[idx].pos = Some((pct.x, pct.y));
                    // Advance selection to the next slot.
                    if idx + 1 < self.slots.len() {
                        self.selected = Some(idx + 1);
                    }
                    let placed = self.slots.iter().filter(|s| s.pos.is_some()).count();
                    let total = self.slots.len();
                    self.status_msg = format!("{placed}/{total} slots placed.");
                }
            }
            Message::SelectSlot(idx) => {
                self.selected = Some(idx);
            }
            Message::Save => match self.save_toml() {
                Ok(()) => {
                    self.status_msg = format!("Saved to {}", self.toml_path.display());
                }
                Err(e) => {
                    self.status_msg = format!("Save failed: {e}");
                }
            },
        }
        iced::Task::none()
    }

    fn save_toml(&self) -> Result<(), String> {
        let mut out = self.toml_prefix.clone();
        out.push('\n');
        for slot in &self.slots {
            if let Some((x, y)) = slot.pos {
                out.push('\n');
                out.push_str("[[resource_slots]]\n");
                out.push_str(&format!("resource = \"{}\"\n", slot.resource));
                out.push_str(&format!("index = {}\n", slot.index));
                out.push_str(&format!("x = {x:.4}\n"));
                out.push_str(&format!("y = {y:.4}\n"));
            }
        }
        fs::write(&self.toml_path, &out).map_err(|e| e.to_string())
    }

    fn view(&self) -> Element<'_, Message> {
        // ---- Sidebar ----
        let placed = self.slots.iter().filter(|s| s.pos.is_some()).count();
        let total = self.slots.len();
        let header = text(format!("Slots: {placed}/{total}"))
            .size(14)
            .color(Color::WHITE);

        let slot_list: Element<_> = scrollable(self.slots.iter().enumerate().fold(
            column![].spacing(2),
            |col, (i, slot)| {
                let is_selected = self.selected == Some(i);
                let label = if slot.pos.is_some() {
                    format!("✓ {}", slot.label())
                } else {
                    format!("  {}", slot.label())
                };
                let btn = button(text(label).size(13))
                    .width(Length::Fill)
                    .on_press(Message::SelectSlot(i))
                    .style(move |_theme, status| {
                        let base = button::Style {
                            background: Some(
                                if is_selected {
                                    Color::from_rgb(0.2, 0.4, 0.7)
                                } else {
                                    Color::from_rgb(0.15, 0.15, 0.15)
                                }
                                .into(),
                            ),
                            text_color: Color::WHITE,
                            border: iced::Border {
                                radius: 3.0.into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        match status {
                            button::Status::Hovered if !is_selected => button::Style {
                                background: Some(Color::from_rgb(0.25, 0.25, 0.25).into()),
                                ..base
                            },
                            _ => base,
                        }
                    });
                col.push(btn)
            },
        ))
        .height(Length::Fill)
        .into();

        let save_btn = button(text("Save").size(14).color(Color::WHITE))
            .on_press(Message::Save)
            .width(Length::Fill)
            .style(|_, status| button::Style {
                background: Some(
                    match status {
                        button::Status::Hovered => Color::from_rgb(0.1, 0.5, 0.2),
                        _ => Color::from_rgb(0.08, 0.38, 0.15),
                    }
                    .into(),
                ),
                text_color: Color::WHITE,
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            });

        let sidebar = container(column![header, slot_list, save_btn].spacing(6).padding(8))
            .width(Length::Fixed(150.0))
            .height(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(Color::from_rgb(0.1, 0.1, 0.1).into()),
                ..Default::default()
            });

        // ---- Map + overlay ----
        let placed_positions: Vec<(f32, f32, bool)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.pos.map(|(x, y)| (x, y, self.selected == Some(i))))
            .collect();

        let overlay = CoordOverlay {
            img_w: self.img_w,
            img_h: self.img_h,
            placed: placed_positions,
        };

        // Use a column for the map area so the image+canvas stack gets laid out
        // the same way the original tool did (column > stack).
        let map_col = column![
            stack![
                iced::widget::image(self.image_handle.clone())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .content_fit(ContentFit::Contain),
                canvas(overlay).width(Length::Fill).height(Length::Fill),
            ],
        ]
        .width(Length::Fill)
        .height(Length::Fill);

        // ---- Status bar ----
        let coord_str = match self.cursor_pct {
            Some((x, y)) => format!("{} | cursor: x={x:.4} y={y:.4}", self.status_msg),
            None => self.status_msg.clone(),
        };
        let status_bar = container(text(coord_str).size(14).color(Color::WHITE))
            .style(|_theme: &Theme| container::Style {
                background: Some(Color::from_rgb(0.08, 0.08, 0.08).into()),
                ..Default::default()
            })
            .width(Length::Fill)
            .padding([5, 10]);

        // Outer layout: sidebar + map side by side, status bar below.
        let main_row = row![sidebar, map_col]
            .width(Length::Fill)
            .height(Length::Fill);

        column![main_row, status_bar]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

// ---------------------------------------------------------------------------
// Build the slot list from MapData
// ---------------------------------------------------------------------------

/// Standard slot counts per resource (matches price_table lengths in powergrid-core).
const STANDARD_SLOTS: &[(&str, usize)] =
    &[("coal", 24), ("oil", 18), ("garbage", 10), ("uranium", 10)];

fn build_slot_list(map_data: &MapData) -> Vec<Slot> {
    // If the TOML already declares resource_slots, use the set of (resource, index) pairs
    // from there to know what slots need positioning.  Otherwise fall back to the standard list.
    if !map_data.resource_slots.is_empty() {
        // Collect unique (resource, index) pairs in the order they appear.
        let mut seen = std::collections::HashSet::new();
        let mut slots = Vec::new();
        for rs in &map_data.resource_slots {
            let key = (rs.resource.clone(), rs.index);
            if seen.insert(key) {
                slots.push(Slot {
                    resource: rs.resource.clone(),
                    index: rs.index,
                    pos: None,
                });
            }
        }
        // Ensure they are sorted by resource (in STANDARD_SLOTS order) then index.
        let resource_order: std::collections::HashMap<&str, usize> = STANDARD_SLOTS
            .iter()
            .enumerate()
            .map(|(i, (r, _))| (*r, i))
            .collect();
        slots.sort_by_key(|s| {
            (
                resource_order
                    .get(s.resource.as_str())
                    .copied()
                    .unwrap_or(99),
                s.index,
            )
        });
        slots
    } else {
        // Generate from standard counts.
        STANDARD_SLOTS
            .iter()
            .flat_map(|(resource, count)| {
                (0..*count).map(move |i| Slot {
                    resource: resource.to_string(),
                    index: i,
                    pos: None,
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Compute the ContentFit::Contain rendered dimensions and offsets within a canvas.
/// Returns (disp_w, disp_h, offset_x, offset_y).
fn contain_rect(canvas_w: f32, canvas_h: f32, img_w: f32, img_h: f32) -> (f32, f32, f32, f32) {
    let img_ratio = img_w / img_h;
    let canvas_ratio = canvas_w / canvas_h;
    let (disp_w, disp_h) = if canvas_ratio < img_ratio {
        let s = canvas_w / img_w;
        (canvas_w, img_h * s)
    } else {
        let s = canvas_h / img_h;
        (img_w * s, canvas_h)
    };
    let offset_x = (canvas_w - disp_w) / 2.0;
    let offset_y = (canvas_h - disp_h) / 2.0;
    (disp_w, disp_h, offset_x, offset_y)
}

/// Convert a canvas-local point to image-relative percentages (0.0–1.0).
/// Returns None when the point is in the letterbox area outside the image.
fn to_pct(local: Point, disp_w: f32, disp_h: f32, off_x: f32, off_y: f32) -> Option<(f32, f32)> {
    let x = (local.x - off_x) / disp_w;
    let y = (local.y - off_y) / disp_h;
    ((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)).then_some((x, y))
}

// ---------------------------------------------------------------------------
// Canvas overlay — handles mouse events and draws placed markers
// ---------------------------------------------------------------------------

struct CoordOverlay {
    img_w: f32,
    img_h: f32,
    /// (x_pct, y_pct, is_selected_slot) for each already-placed slot.
    placed: Vec<(f32, f32, bool)>,
}

impl canvas::Program<Message> for CoordOverlay {
    type State = ();

    fn update(
        &self,
        _state: &mut (),
        event: canvas::Event,
        bounds: Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        let pct_msg = |make: fn(Point) -> Message| -> Option<Message> {
            cursor.position_in(bounds).and_then(|local| {
                let (disp_w, disp_h, off_x, off_y) =
                    contain_rect(bounds.width, bounds.height, self.img_w, self.img_h);
                to_pct(local, disp_w, disp_h, off_x, off_y).map(|(x, y)| make(Point::new(x, y)))
            })
        };

        match event {
            canvas::Event::Mouse(iced::mouse::Event::CursorMoved { .. }) => (
                canvas::event::Status::Ignored,
                pct_msg(Message::CursorMoved),
            ),
            canvas::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                (canvas::event::Status::Ignored, pct_msg(Message::Clicked))
            }
            _ => (canvas::event::Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        if self.placed.is_empty() {
            return vec![];
        }
        let (disp_w, disp_h, off_x, off_y) =
            contain_rect(bounds.width, bounds.height, self.img_w, self.img_h);
        let radius = (disp_w * 0.008).max(4.0);

        let mut frame = canvas::Frame::new(renderer, bounds.size());
        for (x_pct, y_pct, is_selected) in &self.placed {
            let cx = off_x + x_pct * disp_w;
            let cy = off_y + y_pct * disp_h;
            let color = if *is_selected {
                Color::from_rgba(1.0, 1.0, 0.0, 0.9)
            } else {
                Color::from_rgba(0.0, 0.8, 1.0, 0.7)
            };
            let circle = canvas::Path::circle(Point::new(cx, cy), radius);
            frame.fill(&circle, color);
        }
        vec![frame.into_geometry()]
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

pub fn main() -> iced::Result {
    iced::application("Map Tool", App::update, App::view)
        .window(iced::window::Settings {
            size: Size::new(900.0, 900.0),
            ..Default::default()
        })
        .run_with(App::new)
}
