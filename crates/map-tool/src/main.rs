use iced::{
    widget::{button, canvas, column, container, row, scrollable, stack, text},
    Color, ContentFit, Element, Length, Point, Rectangle, Renderer, Size, Theme,
};
use powergrid_core::map::MapData;
use std::{env, fs, path::PathBuf};

// ---------------------------------------------------------------------------
// A positioned slot (in-memory working state)
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
    /// User selected a resource slot in the sidebar list.
    SelectResourceSlot(usize),
    /// User selected a turn order slot in the sidebar list.
    SelectTurnOrderSlot(usize),
    /// Save coordinates back to the TOML file.
    Save,
}

struct App {
    image_handle: iced::widget::image::Handle,
    img_w: f32,
    img_h: f32,
    toml_path: PathBuf,
    /// Original file content up to (but not including) the first generated section.
    toml_prefix: String,
    resource_slots: Vec<Slot>,
    turn_order_slots: Vec<Slot>,
    /// Index into `resource_slots` of the currently selected slot, if any.
    selected_resource: Option<usize>,
    /// Index into `turn_order_slots` of the currently selected slot, if any.
    selected_turn_order: Option<usize>,
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

        // Split at whichever generated section comes first so we can regenerate both.
        let split_pos = [
            raw.find("[[resource_slots]]"),
            raw.find("[[turn_order_slots]]"),
        ]
        .into_iter()
        .flatten()
        .min();
        let toml_prefix = if let Some(pos) = split_pos {
            raw[..pos].trim_end().to_string()
        } else {
            raw.trim_end().to_string()
        };

        let map_data: MapData = toml::from_str(&raw)
            .unwrap_or_else(|e| panic!("Cannot parse {}: {e}", toml_path.display()));

        let mut resource_slots = build_resource_slot_list(&map_data);
        for rs in &map_data.resource_slots {
            if let Some(slot) = resource_slots
                .iter_mut()
                .find(|s| s.resource == rs.resource && s.index == rs.index)
            {
                slot.pos = Some((rs.x, rs.y));
            }
        }

        let mut turn_order_slots = build_turn_order_slot_list();
        for ts in &map_data.turn_order_slots {
            if let Some(slot) = turn_order_slots.iter_mut().find(|s| s.index == ts.index) {
                slot.pos = Some((ts.x, ts.y));
            }
        }

        let res_placed = resource_slots.iter().filter(|s| s.pos.is_some()).count();
        let to_placed = turn_order_slots.iter().filter(|s| s.pos.is_some()).count();
        let status_msg = format!(
            "Resources: {}/{} placed. Turn order: {}/6 placed.",
            res_placed,
            resource_slots.len(),
            to_placed
        );

        (
            Self {
                image_handle,
                img_w,
                img_h,
                toml_path,
                toml_prefix,
                resource_slots,
                turn_order_slots,
                selected_resource: None,
                selected_turn_order: None,
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
                if let Some(idx) = self.selected_resource {
                    self.resource_slots[idx].pos = Some((pct.x, pct.y));
                    if idx + 1 < self.resource_slots.len() {
                        self.selected_resource = Some(idx + 1);
                    }
                } else if let Some(idx) = self.selected_turn_order {
                    self.turn_order_slots[idx].pos = Some((pct.x, pct.y));
                    if idx + 1 < self.turn_order_slots.len() {
                        self.selected_turn_order = Some(idx + 1);
                    }
                }
                self.refresh_status();
            }
            Message::SelectResourceSlot(idx) => {
                self.selected_resource = Some(idx);
                self.selected_turn_order = None;
            }
            Message::SelectTurnOrderSlot(idx) => {
                self.selected_turn_order = Some(idx);
                self.selected_resource = None;
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

    fn refresh_status(&mut self) {
        let res_placed = self
            .resource_slots
            .iter()
            .filter(|s| s.pos.is_some())
            .count();
        let to_placed = self
            .turn_order_slots
            .iter()
            .filter(|s| s.pos.is_some())
            .count();
        self.status_msg = format!(
            "Resources: {}/{} placed. Turn order: {}/6 placed.",
            res_placed,
            self.resource_slots.len(),
            to_placed
        );
    }

    fn save_toml(&self) -> Result<(), String> {
        let mut out = self.toml_prefix.clone();
        out.push('\n');
        for slot in &self.resource_slots {
            if let Some((x, y)) = slot.pos {
                out.push('\n');
                out.push_str("[[resource_slots]]\n");
                out.push_str(&format!("resource = \"{}\"\n", slot.resource));
                out.push_str(&format!("index = {}\n", slot.index));
                out.push_str(&format!("x = {x:.4}\n"));
                out.push_str(&format!("y = {y:.4}\n"));
            }
        }
        for slot in &self.turn_order_slots {
            if let Some((x, y)) = slot.pos {
                out.push('\n');
                out.push_str("[[turn_order_slots]]\n");
                out.push_str(&format!("index = {}\n", slot.index));
                out.push_str(&format!("x = {x:.4}\n"));
                out.push_str(&format!("y = {y:.4}\n"));
            }
        }
        fs::write(&self.toml_path, &out).map_err(|e| e.to_string())
    }

    fn view(&self) -> Element<'_, Message> {
        let res_placed = self
            .resource_slots
            .iter()
            .filter(|s| s.pos.is_some())
            .count();
        let to_placed = self
            .turn_order_slots
            .iter()
            .filter(|s| s.pos.is_some())
            .count();
        let header = text(format!(
            "Resources: {}/{}\nTurn order: {}/6",
            res_placed,
            self.resource_slots.len(),
            to_placed
        ))
        .size(13)
        .color(Color::WHITE);

        // Resource slots list
        let res_header = text("-- Resources --")
            .size(12)
            .color(Color::from_rgb(0.7, 0.7, 0.7));
        let res_list = self.resource_slots.iter().enumerate().fold(
            column![res_header].spacing(2),
            |col, (i, slot)| {
                let is_selected = self.selected_resource == Some(i);
                let label = if slot.pos.is_some() {
                    format!("✓ {}", slot.label())
                } else {
                    format!("  {}", slot.label())
                };
                col.push(slot_button(
                    label,
                    is_selected,
                    Message::SelectResourceSlot(i),
                ))
            },
        );

        // Turn order slots list
        let to_header = text("-- Turn Order --")
            .size(12)
            .color(Color::from_rgb(0.7, 0.7, 0.7));
        let to_list = self.turn_order_slots.iter().enumerate().fold(
            column![to_header].spacing(2),
            |col, (i, slot)| {
                let is_selected = self.selected_turn_order == Some(i);
                let label = if slot.pos.is_some() {
                    format!("✓ pos {}", slot.index)
                } else {
                    format!("  pos {}", slot.index)
                };
                col.push(slot_button(
                    label,
                    is_selected,
                    Message::SelectTurnOrderSlot(i),
                ))
            },
        );

        let slot_list: Element<_> = scrollable(column![res_list, to_list].spacing(8))
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
            .width(Length::Fixed(160.0))
            .height(Length::Fill)
            .style(|_theme: &Theme| container::Style {
                background: Some(Color::from_rgb(0.1, 0.1, 0.1).into()),
                ..Default::default()
            });

        // Build placed positions for the overlay.
        // (x, y, is_selected, is_turn_order)
        let mut placed_positions: Vec<(f32, f32, bool, bool)> = self
            .resource_slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.pos
                    .map(|(x, y)| (x, y, self.selected_resource == Some(i), false))
            })
            .collect();
        placed_positions.extend(
            self.turn_order_slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    s.pos
                        .map(|(x, y)| (x, y, self.selected_turn_order == Some(i), true))
                }),
        );

        let overlay = CoordOverlay {
            img_w: self.img_w,
            img_h: self.img_h,
            placed: placed_positions,
        };

        let map_col = column![stack![
            iced::widget::image(self.image_handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .content_fit(ContentFit::Contain),
            canvas(overlay).width(Length::Fill).height(Length::Fill),
        ],]
        .width(Length::Fill)
        .height(Length::Fill);

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

        let main_row = row![sidebar, map_col]
            .width(Length::Fill)
            .height(Length::Fill);

        column![main_row, status_bar]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

fn slot_button(label: String, is_selected: bool, msg: Message) -> Element<'static, Message> {
    button(text(label).size(13))
        .width(Length::Fill)
        .on_press(msg)
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
        })
        .into()
}

// ---------------------------------------------------------------------------
// Build slot lists from MapData
// ---------------------------------------------------------------------------

/// Standard slot counts per resource (matches price_table lengths in powergrid-core).
const STANDARD_SLOTS: &[(&str, usize)] =
    &[("coal", 24), ("oil", 24), ("garbage", 24), ("uranium", 12)];

fn build_resource_slot_list(map_data: &MapData) -> Vec<Slot> {
    if !map_data.resource_slots.is_empty() {
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

fn build_turn_order_slot_list() -> Vec<Slot> {
    (0..6)
        .map(|i| Slot {
            resource: "turn_order".to_string(),
            index: i,
            pos: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

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

fn to_pct(local: Point, disp_w: f32, disp_h: f32, off_x: f32, off_y: f32) -> Option<(f32, f32)> {
    let x = (local.x - off_x) / disp_w;
    let y = (local.y - off_y) / disp_h;
    ((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)).then_some((x, y))
}

// ---------------------------------------------------------------------------
// Canvas overlay
// ---------------------------------------------------------------------------

struct CoordOverlay {
    img_w: f32,
    img_h: f32,
    /// (x_pct, y_pct, is_selected, is_turn_order) for each placed slot.
    placed: Vec<(f32, f32, bool, bool)>,
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
        for (x_pct, y_pct, is_selected, is_turn_order) in &self.placed {
            let cx = off_x + x_pct * disp_w;
            let cy = off_y + y_pct * disp_h;
            let color = if *is_selected {
                Color::from_rgba(1.0, 1.0, 0.0, 0.9)
            } else if *is_turn_order {
                Color::from_rgba(1.0, 1.0, 1.0, 0.85)
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
