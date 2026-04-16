use crate::app::Message;
use iced::{
    widget::{button, canvas, column, container, row, scrollable, stack, text, text_input, Action},
    Color, Element, Length, Point, Rectangle, Renderer, Theme, Vector,
};
use powergrid_core::{
    map::{City, ConnectionEdge, ResourceSlot, TurnOrderSlot},
    types::{connection_cost, Phase, PlayerColor, PlayerId, Resource},
    GameState,
};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

static GERMANY_MAP_HANDLE: LazyLock<iced::widget::image::Handle> = LazyLock::new(|| {
    iced::widget::image::Handle::from_bytes(include_bytes!("../assets/maps/germany.png").as_slice())
});

static PLANT_CARD_HANDLES: LazyLock<HashMap<u8, iced::widget::image::Handle>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert(
            3,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_03.png").as_slice(),
            ),
        );
        m.insert(
            4,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_04.png").as_slice(),
            ),
        );
        m.insert(
            5,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_05.png").as_slice(),
            ),
        );
        m.insert(
            6,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_06.png").as_slice(),
            ),
        );
        m.insert(
            7,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_07.png").as_slice(),
            ),
        );
        m.insert(
            8,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_08.png").as_slice(),
            ),
        );
        m.insert(
            9,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_09.png").as_slice(),
            ),
        );
        m.insert(
            10,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_10.png").as_slice(),
            ),
        );
        m.insert(
            11,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_11.png").as_slice(),
            ),
        );
        m.insert(
            12,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_12.png").as_slice(),
            ),
        );
        m.insert(
            13,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_13.png").as_slice(),
            ),
        );
        m.insert(
            14,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_14.png").as_slice(),
            ),
        );
        m.insert(
            15,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_15.png").as_slice(),
            ),
        );
        m.insert(
            16,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_16.png").as_slice(),
            ),
        );
        m.insert(
            17,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_17.png").as_slice(),
            ),
        );
        m.insert(
            18,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_18.png").as_slice(),
            ),
        );
        m.insert(
            19,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_19.png").as_slice(),
            ),
        );
        m.insert(
            20,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_20.png").as_slice(),
            ),
        );
        m.insert(
            21,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_21.png").as_slice(),
            ),
        );
        m.insert(
            22,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_22.png").as_slice(),
            ),
        );
        m.insert(
            23,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_23.png").as_slice(),
            ),
        );
        m.insert(
            24,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_24.png").as_slice(),
            ),
        );
        m.insert(
            25,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_25.png").as_slice(),
            ),
        );
        m.insert(
            26,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_26.png").as_slice(),
            ),
        );
        m.insert(
            27,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_27.png").as_slice(),
            ),
        );
        m.insert(
            28,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_28.png").as_slice(),
            ),
        );
        m.insert(
            29,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_29.png").as_slice(),
            ),
        );
        m.insert(
            30,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_30.png").as_slice(),
            ),
        );
        m.insert(
            31,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_31.png").as_slice(),
            ),
        );
        m.insert(
            32,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_32.png").as_slice(),
            ),
        );
        m.insert(
            33,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_33.png").as_slice(),
            ),
        );
        m.insert(
            34,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_34.png").as_slice(),
            ),
        );
        m.insert(
            35,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_35.png").as_slice(),
            ),
        );
        m.insert(
            36,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_36.png").as_slice(),
            ),
        );
        m.insert(
            37,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_37.png").as_slice(),
            ),
        );
        m.insert(
            38,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_38.png").as_slice(),
            ),
        );
        m.insert(
            39,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_39.png").as_slice(),
            ),
        );
        m.insert(
            40,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_40.png").as_slice(),
            ),
        );
        m.insert(
            42,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_42.png").as_slice(),
            ),
        );
        m.insert(
            44,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_44.png").as_slice(),
            ),
        );
        m.insert(
            46,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_46.png").as_slice(),
            ),
        );
        m.insert(
            50,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_50.png").as_slice(),
            ),
        );
        // Sentinel key 0 = step3 fallback card
        m.insert(
            0,
            iced::widget::image::Handle::from_bytes(
                include_bytes!("../assets/cards/card_step3.png").as_slice(),
            ),
        );
        m
    });

// ---------------------------------------------------------------------------
// Resource market overlay — draws colored circles on the map's market board
// ---------------------------------------------------------------------------

/// Original map image dimensions (germany.jpg is 1869 × 2593).
const IMG_W: f32 = 1869.0;
const IMG_H: f32 = 2593.0;

/// Circle radius expressed as a fraction of the displayed image width.
const SLOT_RADIUS_FRAC: f32 = 0.009;

/// Hit-test radius for city clicks, as a fraction of image width.
const CITY_HIT_RADIUS: f32 = 0.03;

/// Display radius for city markers, as a fraction of image width.
const CITY_RADIUS_FRAC: f32 = 0.006;

/// Compute the ContentFit::Contain image layout within the given bounds.
/// Returns (img_w, img_h, offset_x, offset_y).
fn base_image_layout(bounds: Rectangle) -> (f32, f32, f32, f32) {
    let img_ratio = IMG_W / IMG_H;
    let bounds_ratio = bounds.width / bounds.height;
    let (img_w, img_h) = if bounds_ratio < img_ratio {
        let s = bounds.width / IMG_W;
        (bounds.width, IMG_H * s)
    } else {
        let s = bounds.height / IMG_H;
        (IMG_W * s, bounds.height)
    };
    let offset_x = (bounds.width - img_w) / 2.0;
    let offset_y = (bounds.height - img_h) / 2.0;
    (img_w, img_h, offset_x, offset_y)
}

/// Ephemeral drag-tracking state for the map overlay canvas.
#[derive(Default)]
struct MapDragState {
    dragging: bool,
    last_cursor: Point,
    drag_distance: f32,
}

/// Bottom canvas: renders only the map image at the current zoom/pan.
struct MapBackground {
    zoom: f32,
    pan: Vector,
}

impl canvas::Program<Message> for MapBackground {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let (img_w, img_h, offset_x, offset_y) = base_image_layout(bounds);
        frame.with_save(|frame| {
            frame.translate(self.pan);
            frame.scale(self.zoom);
            frame.draw_image(
                Rectangle::new(
                    Point::new(offset_x, offset_y),
                    iced::Size::new(img_w, img_h),
                ),
                canvas::Image::new(GERMANY_MAP_HANDLE.clone()),
            );
        });
        vec![frame.into_geometry()]
    }
}

struct MarketOverlay<'a> {
    coal: u8,
    oil: u8,
    garbage: u8,
    uranium: u8,
    slots: &'a [ResourceSlot],
    turn_order_slots: &'a [TurnOrderSlot],
    /// (slot_index, player_color) for each player in turn order (index 0 = first place).
    turn_order_players: Vec<(usize, PlayerColor)>,
    cities: &'a HashMap<String, City>,
    phase: &'a Phase,
    my_id: PlayerId,
    selected_build_cities: HashSet<String>,
    highlighted_edges: Vec<ConnectionEdge>,
    zoom: f32,
    pan: Vector,
}

impl canvas::Program<Message> for MarketOverlay<'_> {
    type State = MapDragState;

    fn update(
        &self,
        state: &mut MapDragState,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> Option<Action<Message>> {
        match event {
            canvas::Event::Mouse(mouse_event) => match mouse_event {
                iced::mouse::Event::WheelScrolled { delta } => {
                    let screen_pos = cursor.position_in(bounds)?;
                    let delta_y = match delta {
                        iced::mouse::ScrollDelta::Lines { y, .. } => *y,
                        iced::mouse::ScrollDelta::Pixels { y, .. } => *y / 20.0,
                    };
                    let factor = 1.15f32.powf(delta_y);
                    Some(
                        Action::publish(Message::MapZoom {
                            factor,
                            cursor_x: screen_pos.x,
                            cursor_y: screen_pos.y,
                        })
                        .and_capture(),
                    )
                }
                iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left) => {
                    if let Some(pos) = cursor.position_in(bounds) {
                        state.dragging = true;
                        state.last_cursor = pos;
                        state.drag_distance = 0.0;
                        Some(Action::capture())
                    } else {
                        None
                    }
                }
                iced::mouse::Event::CursorMoved { .. } => {
                    if state.dragging {
                        if let Some(pos) = cursor.position_in(bounds) {
                            let dx = pos.x - state.last_cursor.x;
                            let dy = pos.y - state.last_cursor.y;
                            state.drag_distance += (dx * dx + dy * dy).sqrt();
                            state.last_cursor = pos;
                            Some(Action::publish(Message::MapPan { dx, dy }).and_capture())
                        } else {
                            state.dragging = false;
                            None
                        }
                    } else {
                        None
                    }
                }
                iced::mouse::Event::ButtonReleased(iced::mouse::Button::Left) => {
                    let was_dragging = state.dragging;
                    let drag_dist = state.drag_distance;
                    let release_pos = state.last_cursor;
                    state.dragging = false;

                    // Small movement = treat as a click for city selection.
                    if was_dragging && drag_dist < 5.0 {
                        let is_my_build_turn = matches!(&self.phase, Phase::BuildCities { remaining }
                            if remaining.first() == Some(&self.my_id));
                        if is_my_build_turn {
                            let (img_w, img_h, offset_x, offset_y) = base_image_layout(bounds);
                            // Inverse of the canvas transform: screen = pan + zoom * local
                            let local_x = (release_pos.x - self.pan.x) / self.zoom;
                            let local_y = (release_pos.y - self.pan.y) / self.zoom;
                            let x_pct = (local_x - offset_x) / img_w;
                            let y_pct = (local_y - offset_y) / img_h;

                            for (city_id, city) in self.cities {
                                if let (Some(cx), Some(cy)) = (city.x, city.y) {
                                    let dx = x_pct - cx;
                                    let dy = y_pct - cy;
                                    if dx * dx + dy * dy <= CITY_HIT_RADIUS * CITY_HIT_RADIUS {
                                        return Some(
                                            Action::publish(Message::ToggleBuildCity(
                                                city_id.clone(),
                                            ))
                                            .and_capture(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Some(Action::capture())
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn mouse_interaction(
        &self,
        state: &MapDragState,
        bounds: Rectangle,
        cursor: iced::mouse::Cursor,
    ) -> iced::mouse::Interaction {
        if cursor.is_over(bounds) {
            if state.dragging {
                iced::mouse::Interaction::Grabbing
            } else {
                iced::mouse::Interaction::Grab
            }
        } else {
            iced::mouse::Interaction::default()
        }
    }

    fn draw(
        &self,
        _state: &MapDragState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let (img_w, img_h, offset_x, offset_y) = base_image_layout(bounds);
        let radius = SLOT_RADIUS_FRAC * img_w;
        let city_radius = CITY_RADIUS_FRAC * img_w;
        let is_build_phase = matches!(&self.phase, Phase::BuildCities { remaining }
            if remaining.first() == Some(&self.my_id));

        // Only overlay paths go here; the map image lives in the MapBackground
        // canvas below in the stack, which gets its own rendering layer.
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        frame.with_save(|frame| {
            frame.translate(self.pan);
            frame.scale(self.zoom);

            let draw_resource =
                |frame: &mut canvas::Frame, color: Color, resource_name: &str, current: u8| {
                    let mut resource_slots: Vec<&ResourceSlot> = self
                        .slots
                        .iter()
                        .filter(|s| s.resource == resource_name)
                        .collect();
                    resource_slots.sort_by_key(|s| s.index);
                    let total = resource_slots.len();
                    if total == 0 || current == 0 {
                        return;
                    }
                    let occupied_from = total.saturating_sub(current as usize);
                    for slot in &resource_slots[occupied_from..] {
                        let cx = offset_x + slot.x * img_w;
                        let cy = offset_y + slot.y * img_h;
                        let circle = canvas::Path::circle(Point::new(cx, cy), radius);
                        frame.fill(&circle, color);
                    }
                };

            draw_resource(frame, Color::from_rgb(0.42, 0.27, 0.14), "coal", self.coal);
            draw_resource(frame, Color::from_rgb(0.1, 0.1, 0.1), "oil", self.oil);
            draw_resource(
                frame,
                Color::from_rgb(0.95, 0.85, 0.1),
                "garbage",
                self.garbage,
            );
            draw_resource(
                frame,
                Color::from_rgb(0.85, 0.1, 0.1),
                "uranium",
                self.uranium,
            );

            for (slot_idx, player_color) in &self.turn_order_players {
                if let Some(slot) = self.turn_order_slots.iter().find(|s| s.index == *slot_idx) {
                    let cx = offset_x + slot.x * img_w;
                    let cy = offset_y + slot.y * img_h;
                    let outline = canvas::Path::circle(Point::new(cx, cy), radius + 1.5);
                    frame.fill(&outline, Color::WHITE);
                    let fill = canvas::Path::circle(Point::new(cx, cy), radius);
                    frame.fill(&fill, player_color_to_iced(*player_color));
                }
            }

            for city in self.cities.values() {
                if let (Some(cx), Some(cy)) = (city.x, city.y) {
                    let px = offset_x + cx * img_w;
                    let py = offset_y + cy * img_h;
                    let color = if self.selected_build_cities.contains(&city.id) {
                        Color::from_rgba(0.1, 0.7, 1.0, 0.95)
                    } else if !city.owners.is_empty() {
                        Color::from_rgba(0.2, 0.9, 0.2, 0.8)
                    } else if is_build_phase {
                        Color::from_rgba(1.0, 1.0, 1.0, 0.7)
                    } else {
                        Color::from_rgba(1.0, 1.0, 1.0, 0.35)
                    };
                    let circle = canvas::Path::circle(Point::new(px, py), city_radius);
                    frame.fill(&circle, color);
                }
            }

            for edge in &self.highlighted_edges {
                let Some(from_city) = self.cities.get(&edge.from) else {
                    continue;
                };
                let Some(to_city) = self.cities.get(&edge.to) else {
                    continue;
                };
                let (Some(from_x), Some(from_y)) = (from_city.x, from_city.y) else {
                    continue;
                };
                let (Some(to_x), Some(to_y)) = (to_city.x, to_city.y) else {
                    continue;
                };
                let start = Point::new(offset_x + from_x * img_w, offset_y + from_y * img_h);
                let end = Point::new(offset_x + to_x * img_w, offset_y + to_y * img_h);
                let line = canvas::Path::line(start, end);
                frame.stroke(
                    &line,
                    canvas::Stroke::default()
                        .with_color(Color::from_rgba(0.1, 0.7, 1.0, 0.95))
                        .with_width((city_radius * 0.7).max(2.0)),
                );
            }
        });

        vec![frame.into_geometry()]
    }
}

fn player_color_to_iced(color: PlayerColor) -> Color {
    match color {
        PlayerColor::Red => Color::from_rgb(0.9, 0.1, 0.1),
        PlayerColor::Blue => Color::from_rgb(0.1, 0.3, 0.9),
        PlayerColor::Green => Color::from_rgb(0.1, 0.7, 0.2),
        PlayerColor::Yellow => Color::from_rgb(0.95, 0.85, 0.1),
        PlayerColor::Purple => Color::from_rgb(0.6, 0.1, 0.8),
        PlayerColor::Black => Color::from_rgb(0.15, 0.15, 0.15),
    }
}

// ---------------------------------------------------------------------------
// Connect screen
// ---------------------------------------------------------------------------

pub struct ConnectScreen {
    pub server_url: String,
    pub player_name: String,
    pub selected_color: PlayerColor,
    pub error: Option<String>,
}

impl ConnectScreen {
    pub fn new() -> Self {
        Self {
            server_url: "ws://localhost:3000/ws".to_string(),
            player_name: String::new(),
            selected_color: PlayerColor::Red,
            error: None,
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let color_buttons = row![
            color_button(PlayerColor::Red, self.selected_color),
            color_button(PlayerColor::Blue, self.selected_color),
            color_button(PlayerColor::Green, self.selected_color),
            color_button(PlayerColor::Yellow, self.selected_color),
            color_button(PlayerColor::Purple, self.selected_color),
            color_button(PlayerColor::Black, self.selected_color),
        ]
        .spacing(8);

        let mut connect_btn = button("Connect");
        if !self.player_name.is_empty() {
            connect_btn = connect_btn.on_press(Message::Connect);
        }

        let mut col = column![
            text("Powergrid").size(32),
            text("Server URL"),
            text_input("ws://localhost:3000/ws", &self.server_url)
                .on_input(Message::ServerUrlChanged),
            text("Your Name"),
            text_input("Enter your name", &self.player_name).on_input(Message::NameChanged),
            text("Color"),
            color_buttons,
            connect_btn,
        ]
        .spacing(12)
        .padding(40)
        .max_width(400);

        if let Some(err) = &self.error {
            col = col.push(text(err.as_str()));
        }

        container(col)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
    }
}

fn color_button(color: PlayerColor, selected: PlayerColor) -> Element<'static, Message> {
    let label = match color {
        PlayerColor::Red => "Red",
        PlayerColor::Blue => "Blue",
        PlayerColor::Green => "Green",
        PlayerColor::Yellow => "Yellow",
        PlayerColor::Purple => "Purple",
        PlayerColor::Black => "Black",
    };
    let btn = button(label).on_press(Message::ColorSelected(color));
    if color == selected {
        button(text(label).size(14))
            .on_press(Message::ColorSelected(color))
            .into()
    } else {
        btn.into()
    }
}

// ---------------------------------------------------------------------------
// Lobby screen
// ---------------------------------------------------------------------------

pub fn lobby_view<'a>(
    state: &'a GameState,
    is_host: bool,
    error: Option<&'a str>,
) -> Element<'a, Message> {
    let players_list = state
        .players
        .iter()
        .fold(column![text("Players:").size(18)].spacing(4), |col, p| {
            col.push(text(format!("  {} ({:?})", p.name, p.color)))
        });

    let mut col = column![text("Powergrid — Lobby").size(28), players_list,]
        .spacing(16)
        .padding(40);

    if is_host {
        let start_btn = if state.players.len() >= 2 {
            button("Start Game").on_press(Message::StartGame)
        } else {
            button("Start Game (need 2+ players)")
        };
        col = col.push(start_btn);
    } else {
        col = col.push(text("Waiting for host to start..."));
    }

    if let Some(err) = error {
        col = col.push(text(format!("Error: {err}")).color([0.8, 0.1, 0.1]));
    }

    container(col)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}

// ---------------------------------------------------------------------------
// Game screen
// ---------------------------------------------------------------------------

pub fn game_view<'a>(
    state: &'a GameState,
    my_id: uuid::Uuid,
    bid_amount: &'a str,
    error: Option<&'a str>,
    map_zoom: f32,
    map_pan: Vector,
    selected_build_cities: &'a [String],
) -> Element<'a, Message> {
    let phase_label = phase_description(&state.phase);

    let me = state.player(my_id);

    let active_player_id = active_player(state);

    // Player status panel — listed in join order (turn order is shown on the map).
    let ordered_players: Vec<_> = state.players.iter().collect();
    let player_panel =
        ordered_players
            .iter()
            .fold(column![text("Players").size(16)].spacing(4), |col, p| {
                let mut markers = String::new();
                if Some(p.id) == active_player_id {
                    markers.push_str(" *");
                }
                if p.id == my_id {
                    markers.push_str(" (you)");
                }
                col.push(text(format!(
                    "{}{}: {} cities, ${}, plants: {}",
                    p.name,
                    markers,
                    p.cities.len(),
                    p.money,
                    p.plants
                        .iter()
                        .map(|pl| pl.number.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )))
            });

    // Power plant market.
    let market = column![
        text("Power Plant Market").size(16),
        text("Actual:"),
        plants_row(&state.market.actual),
        text("Future:"),
        plants_row(&state.market.future),
    ]
    .spacing(4);

    // Resource market.
    let res = &state.resources;
    let resources = column![
        text("Resources").size(16),
        text(format!(
            "Coal: {}  Oil: {}  Garbage: {}  Uranium: {}",
            res.coal, res.oil, res.garbage, res.uranium
        )),
    ]
    .spacing(4);

    let mut build_selection = Vec::new();
    let mut build_preview_edges = Vec::new();
    let mut build_projected_total_cost: Option<u32> = None;
    let mut build_selection_error: Option<String> = None;
    let is_my_build_turn = matches!(&state.phase, Phase::BuildCities { remaining }
        if remaining.first() == Some(&my_id));

    if is_my_build_turn {
        let mut seen = HashSet::new();
        for city_id in selected_build_cities {
            if seen.insert(city_id.clone()) {
                build_selection.push(city_id.clone());
            }
        }
        if let Some(me) = me {
            for city_id in &build_selection {
                let Some(city) = state.map.cities.get(city_id) else {
                    build_selection_error = Some(format!("Unknown city: {city_id}"));
                    break;
                };
                if city.owners.len() >= 3 {
                    build_selection_error = Some(format!("City is full: {city_id}"));
                    break;
                }
                if city.owners.contains(&my_id) {
                    build_selection_error = Some(format!("You already built there: {city_id}"));
                    break;
                }
            }
            if build_selection_error.is_none() {
                if let Some(network) = state
                    .map
                    .connection_network_for(&me.cities, &build_selection)
                {
                    let slot_cost: u32 = build_selection
                        .iter()
                        .map(|city_id| connection_cost(state.map.cities[city_id].owners.len()))
                        .sum();
                    build_preview_edges = network.edges;
                    build_projected_total_cost = Some(network.route_cost + slot_cost);
                }
            }
        }
    }

    // My resources + actions.
    let my_panel: Element<Message> = if let Some(me) = me {
        let res = &me.resources;
        let mut col = column![
            text(format!("You: {} | ${}", me.name, me.money)).size(16),
            text(format!(
                "Resources — Coal: {}  Oil: {}  Garbage: {}  Uranium: {}",
                res.coal, res.oil, res.garbage, res.uranium
            )),
            text("Your Plants:").size(14),
            owned_plants_row(&me.plants),
        ]
        .spacing(8);
        if let Some(err) = error {
            col = col.push(text(format!("Error: {err}")).color([0.8, 0.1, 0.1]));
        }
        col.push(action_panel(
            state,
            my_id,
            bid_amount,
            build_selection.len(),
            build_projected_total_cost,
            build_selection_error.clone(),
        ))
        .into()
    } else {
        text("Spectating").into()
    };

    // Event log.
    let log = state
        .event_log
        .iter()
        .rev()
        .take(10)
        .fold(column![text("Log").size(14)].spacing(2), |col, entry| {
            col.push(text(entry.as_str()).size(12))
        });

    let turn_order_players: Vec<(usize, PlayerColor)> = state
        .player_order
        .iter()
        .enumerate()
        .filter_map(|(i, pid)| state.player(*pid).map(|p| (i, p.color)))
        .collect();
    let bg = MapBackground {
        zoom: map_zoom,
        pan: map_pan,
    };
    let overlay = MarketOverlay {
        coal: state.resources.coal,
        oil: state.resources.oil,
        garbage: state.resources.garbage,
        uranium: state.resources.uranium,
        slots: &state.map.resource_slots,
        turn_order_slots: &state.map.turn_order_slots,
        turn_order_players,
        cities: &state.map.cities,
        phase: &state.phase,
        my_id,
        selected_build_cities: build_selection.iter().cloned().collect(),
        highlighted_edges: build_preview_edges,
        zoom: map_zoom,
        pan: map_pan,
    };
    // stack! renders each child after the first in its own layer, so the
    // overlay canvas (layer 1) draws on top of the background image (layer 0).
    let map_panel = container(stack![
        canvas(bg).width(Length::Fill).height(Length::Fill),
        canvas(overlay).width(Length::Fill).height(Length::Fill),
    ])
    .width(Length::FillPortion(3))
    .height(Length::Fill)
    .clip(true);

    let info_panel = scrollable(
        column![
            text(format!("Round {} — {}", state.round, phase_label)).size(20),
            player_panel,
            market,
            resources,
            my_panel,
            log,
        ]
        .spacing(16)
        .padding(8),
    )
    .width(Length::FillPortion(2))
    .height(Length::Fill);

    container(row![map_panel, info_panel].spacing(16).padding(16))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn plants_row(plants: &[powergrid_core::types::PowerPlant]) -> Element<'static, Message> {
    plants
        .iter()
        .fold(row![].spacing(4), |r, p| {
            let handle = plant_card_handle(p.number);
            r.push(
                button(iced::widget::image(handle).width(54).height(54))
                    .padding(0)
                    .on_press(Message::SelectPlant(p.number)),
            )
        })
        .into()
}

fn owned_plants_row(plants: &[powergrid_core::types::PowerPlant]) -> Element<'static, Message> {
    if plants.is_empty() {
        return text("(none)").size(12).into();
    }
    plants
        .iter()
        .fold(row![].spacing(4), |r, p| {
            let handle = plant_card_handle(p.number);
            r.push(iced::widget::image(handle).width(54).height(54))
        })
        .into()
}

fn plant_card_handle(number: u8) -> iced::widget::image::Handle {
    PLANT_CARD_HANDLES
        .get(&number)
        .unwrap_or_else(|| &PLANT_CARD_HANDLES[&0])
        .clone()
}

fn action_panel<'a>(
    state: &'a GameState,
    my_id: uuid::Uuid,
    bid_amount: &'a str,
    selected_build_count: usize,
    projected_build_cost: Option<u32>,
    build_selection_error: Option<String>,
) -> Element<'a, Message> {
    match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            ..
        } => {
            let my_turn = state.player_order.get(*current_bidder_idx) == Some(&my_id);
            if let Some(bid) = active_bid {
                let is_my_bid_turn = bid.remaining_bidders.first() == Some(&my_id);
                if is_my_bid_turn {
                    let bid_valid = bid_amount.trim().parse::<u32>().is_ok();
                    column![
                        text(format!(
                            "Active bid on plant #{}: ${}",
                            bid.plant_number, bid.amount
                        )),
                        text("Enter amount and press Bid, or Pass"),
                        row![
                            text_input("Enter bid amount", bid_amount)
                                .on_input(Message::BidAmountChanged)
                                .width(150),
                            button("Bid").on_press_maybe(bid_valid.then_some(Message::PlaceBid)),
                            button("Pass Bid").on_press(Message::PassAuction),
                        ]
                        .spacing(8),
                    ]
                    .spacing(8)
                    .into()
                } else {
                    text(format!(
                        "Bidding on plant #{} — waiting...",
                        bid.plant_number
                    ))
                    .into()
                }
            } else if my_turn {
                column![
                    text("Your turn — select a plant from the market above, or:"),
                    button("Pass").on_press(Message::PassAuction),
                ]
                .spacing(8)
                .into()
            } else {
                text("Waiting for other players to bid...").into()
            }
        }
        Phase::BuyResources { remaining } => {
            if remaining.first() == Some(&my_id) {
                column![
                    text("Buy resources (click to buy 1 unit):"),
                    row![
                        button("Coal").on_press(Message::BuyResource(Resource::Coal)),
                        button("Oil").on_press(Message::BuyResource(Resource::Oil)),
                        button("Garbage").on_press(Message::BuyResource(Resource::Garbage)),
                        button("Uranium").on_press(Message::BuyResource(Resource::Uranium)),
                        button("Done").on_press(Message::DoneBuying),
                    ]
                    .spacing(8),
                ]
                .spacing(8)
                .into()
            } else {
                text("Waiting for other players to buy resources...").into()
            }
        }
        Phase::BuildCities { remaining } => {
            if remaining.first() == Some(&my_id) {
                let cost_text = projected_build_cost
                    .map(|cost| format!("Projected total cost: ${cost}"))
                    .unwrap_or_else(|| "Projected total cost: --".to_string());
                column![
                    text("Build cities — click to select/deselect on the map:"),
                    text(format!("Selected cities: {}", selected_build_count)),
                    text(cost_text),
                    text(build_selection_error.unwrap_or_default()),
                    row![button("Done Building").on_press(Message::SubmitBuildCities),].spacing(8),
                ]
                .spacing(8)
                .into()
            } else {
                text("Waiting for other players to build...").into()
            }
        }
        Phase::Bureaucracy { remaining } => {
            if remaining.first() == Some(&my_id) {
                column![
                    text("Power your cities — press to fire all plants you can:"),
                    button("Power Cities").on_press(Message::PowerCities),
                ]
                .spacing(8)
                .into()
            } else {
                text("Waiting for other players...").into()
            }
        }
        Phase::GameOver { winner } => {
            let name = state
                .player(*winner)
                .map(|p| p.name.as_str())
                .unwrap_or("Unknown");
            text(format!("Game Over! {} wins!", name)).size(24).into()
        }
        _ => text("").into(),
    }
}

fn active_player(state: &GameState) -> Option<PlayerId> {
    match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            ..
        } => {
            if let Some(bid) = active_bid {
                bid.remaining_bidders.first().copied()
            } else {
                state.player_order.get(*current_bidder_idx).copied()
            }
        }
        Phase::BuyResources { remaining } => remaining.first().copied(),
        Phase::BuildCities { remaining } => remaining.first().copied(),
        Phase::Bureaucracy { remaining } => remaining.first().copied(),
        _ => None,
    }
}

fn phase_description(phase: &Phase) -> &'static str {
    match phase {
        Phase::Lobby => "Lobby",
        Phase::PlayerOrder => "Determining Player Order",
        Phase::Auction { .. } => "Auction",
        Phase::BuyResources { .. } => "Buy Resources",
        Phase::BuildCities { .. } => "Build Cities",
        Phase::Bureaucracy { .. } => "Bureaucracy",
        Phase::GameOver { .. } => "Game Over",
    }
}
