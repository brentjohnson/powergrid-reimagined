use iced::{futures::channel::mpsc, Element, Subscription, Vector};
use powergrid_core::{
    actions::Action,
    connection_cost,
    map::Map,
    types::{Phase, PlayerColor, Resource},
    GameState,
};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::connection::{self, WsEvent};
use crate::screens::{self, ConnectScreen};

#[derive(Debug, Clone)]
pub enum Message {
    // Connect screen
    ServerUrlChanged(String),
    NameChanged(String),
    ColorSelected(PlayerColor),
    Connect,

    // WebSocket events
    WsEvent(WsEvent),

    // Lobby
    StartGame,

    // Auction
    SelectPlant(u8),
    BidAmountChanged(String),
    PlaceBid,
    PassAuction,

    // Buy resources cart
    AddResourceToCart(Resource),
    RemoveResourceFromCart(Resource),
    ClearResourceCart,
    DoneBuying,

    // Build
    ToggleBuildCity(String),
    ClearBuildSelection,
    DoneBuilding,

    // Bureaucracy
    PowerCities,

    // Map viewport
    MapZoom {
        factor: f32,
        cursor_x: f32,
        cursor_y: f32,
    },
    MapPan {
        dx: f32,
        dy: f32,
    },
}

pub enum Screen {
    Connect(ConnectScreen),
    Game,
}

/// Pre-computed preview of a pending multi-city build.
#[derive(Default)]
pub struct BuildPreview {
    /// Cities in the order that minimises total per-city cost.
    pub ordered: Vec<String>,
    /// Sum of all routing (edge) costs.
    pub total_route_cost: u32,
    /// Sum of all city-slot fees.
    pub total_slot_cost: u32,
    /// Total cost (route + slots).
    pub total_cost: u32,
    /// Set of map edges (each as `(smaller_id, larger_id)`) that form the cheapest paths.
    pub edges: HashSet<(String, String)>,
}

pub struct App {
    screen: Screen,
    game_state: Option<GameState>,
    my_id: Option<Uuid>,
    ws_sender: Option<mpsc::Sender<Action>>,
    /// Set when the user clicks Connect; drives the subscription.
    connect_url: Option<String>,
    /// Name + color saved at Connect time, sent to server after Welcome arrives.
    pending_join: Option<(String, PlayerColor)>,
    /// Current text in the bid amount input field.
    bid_amount: String,
    /// Last action error received from the server; cleared on next successful state update.
    error_message: Option<String>,
    map_zoom: f32,
    map_pan: Vector,
    /// Cities the player has toggled for the pending batch build.
    selected_build_cities: Vec<String>,
    /// Cached preview for the current selection (routes, costs, edges to draw).
    build_preview: BuildPreview,
    /// Resources staged for the pending batch purchase.
    resource_cart: HashMap<Resource, u8>,
    /// Cached total cost of the current cart contents (None if cart is empty).
    resource_cart_cost: Option<u32>,
}

impl App {
    pub fn new() -> (Self, iced::Task<Message>) {
        let cli = CliArgs::parse();

        let mut connect_screen = ConnectScreen::new();
        if let Some(ref name) = cli.name {
            connect_screen.player_name = name.clone();
        }
        if let Some(color) = cli.color {
            connect_screen.selected_color = color;
        }
        if let Some(ref url) = cli.url {
            connect_screen.server_url = url.clone();
        }

        // Auto-connect when all three args are provided.
        let (connect_url, pending_join, screen) =
            if cli.name.is_some() && cli.color.is_some() && cli.url.is_some() {
                (
                    Some(connect_screen.server_url.clone()),
                    Some((
                        connect_screen.player_name.clone(),
                        connect_screen.selected_color,
                    )),
                    Screen::Connect(connect_screen),
                )
            } else {
                (None, None, Screen::Connect(connect_screen))
            };

        let task = if connect_url.is_some() {
            iced::Task::done(Message::Connect)
        } else {
            iced::Task::none()
        };

        (
            Self {
                screen,
                game_state: None,
                my_id: None,
                ws_sender: None,
                connect_url,
                pending_join,
                bid_amount: String::new(),
                error_message: None,
                map_zoom: 1.0,
                map_pan: Vector::default(),
                selected_build_cities: Vec::new(),
                build_preview: BuildPreview::default(),
                resource_cart: HashMap::new(),
                resource_cart_cost: None,
            },
            task,
        )
    }

    pub fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::ServerUrlChanged(url) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.server_url = url;
                }
            }
            Message::NameChanged(name) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.player_name = name;
                }
            }
            Message::ColorSelected(color) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.selected_color = color;
                }
            }
            Message::Connect => {
                // Only act if not already connecting (auto-connect sets these in new()).
                if self.connect_url.is_none() {
                    if let Screen::Connect(s) = &self.screen {
                        self.connect_url = Some(s.server_url.clone());
                        self.pending_join = Some((s.player_name.clone(), s.selected_color));
                    }
                }
            }

            Message::WsEvent(event) => match event {
                WsEvent::Connected(sender) => {
                    self.ws_sender = Some(sender);
                    // my_id will be set when we receive the Welcome message.
                    // Switch to Game screen to show "Waiting for server..." while
                    // Welcome + JoinGame round-trip completes.
                    self.screen = Screen::Game;
                }
                WsEvent::MessageReceived(msg) => {
                    use powergrid_core::actions::ServerMessage;
                    match msg {
                        ServerMessage::Welcome { your_id } => {
                            self.my_id = Some(your_id);
                            // Now that we know our ID, send JoinGame.
                            // We need the name/color from the connect screen, which is
                            // stored in connect_url's companion — save them before switching.
                            if let Some((name, color)) = self.pending_join.take() {
                                self.send(Action::JoinGame { name, color });
                            }
                        }
                        ServerMessage::StateUpdate(state) => {
                            // Clear build selection once it's no longer our build turn.
                            let still_my_build_turn = self
                                .my_id
                                .map(|id| {
                                    matches!(&state.phase, Phase::BuildCities { remaining }
                                    if remaining.first() == Some(&id))
                                })
                                .unwrap_or(false);
                            if !still_my_build_turn {
                                self.selected_build_cities.clear();
                                self.build_preview = BuildPreview::default();
                            }
                            // Clear resource cart once it's no longer our buy turn.
                            let still_my_buy_turn = self
                                .my_id
                                .map(|id| {
                                    matches!(&state.phase, Phase::BuyResources { remaining }
                                    if remaining.first() == Some(&id))
                                })
                                .unwrap_or(false);
                            if !still_my_buy_turn {
                                self.resource_cart.clear();
                                self.resource_cart_cost = None;
                            }
                            self.game_state = Some(*state);
                            self.error_message = None;
                        }
                        ServerMessage::ActionError { message } => {
                            self.error_message = Some(message);
                        }
                        ServerMessage::Event { .. } => {}
                    }
                }
                WsEvent::Disconnected => {
                    self.ws_sender = None;
                    if matches!(self.screen, Screen::Game) {
                        // Stay on game screen; reconnect will be attempted automatically.
                    }
                }
            },

            Message::StartGame => {
                self.send(Action::StartGame);
            }
            Message::SelectPlant(num) => {
                self.send(Action::SelectPlant { plant_number: num });
            }
            Message::BidAmountChanged(val) => {
                self.bid_amount = val;
            }
            Message::PlaceBid => {
                if let Ok(amount) = self.bid_amount.trim().parse::<u32>() {
                    self.send(Action::PlaceBid { amount });
                    self.bid_amount = String::new();
                }
            }
            Message::PassAuction => {
                self.send(Action::PassAuction);
            }
            Message::AddResourceToCart(resource) => {
                if let Some(state) = &self.game_state {
                    if let Some(my_id) = self.my_id {
                        if let Some(player) = state.player(my_id) {
                            // Check market availability.
                            let cart_count = self.resource_cart.get(&resource).copied().unwrap_or(0);
                            let new_count = cart_count + 1;
                            if state.resources.available(resource) < new_count {
                                return iced::Task::none();
                            }
                            // Check capacity: simulate player with current resources + cart + 1 more.
                            let mut sim = player.clone();
                            for (&r, &amt) in &self.resource_cart {
                                sim.resources.add(r, amt);
                            }
                            if !sim.can_add_resource(resource, 1) {
                                return iced::Task::none();
                            }
                            *self.resource_cart.entry(resource).or_insert(0) += 1;
                            self.refresh_resource_preview();
                        }
                    }
                }
            }
            Message::RemoveResourceFromCart(resource) => {
                let count = self.resource_cart.entry(resource).or_insert(0);
                if *count > 0 {
                    *count -= 1;
                }
                self.refresh_resource_preview();
            }
            Message::ClearResourceCart => {
                self.resource_cart.clear();
                self.resource_cart_cost = None;
            }
            Message::DoneBuying => {
                let purchases: Vec<(Resource, u8)> = [
                    Resource::Coal,
                    Resource::Oil,
                    Resource::Garbage,
                    Resource::Uranium,
                ]
                .iter()
                .filter_map(|&r| {
                    let amt = self.resource_cart.get(&r).copied().unwrap_or(0);
                    if amt > 0 { Some((r, amt)) } else { None }
                })
                .collect();
                if purchases.is_empty() {
                    self.send(Action::DoneBuying);
                } else {
                    self.send(Action::BuyResourceBatch { purchases });
                }
            }
            Message::ToggleBuildCity(city_id) => {
                if let Some(state) = &self.game_state {
                    if let Some(my_id) = self.my_id {
                        // Ignore clicks on cities we already own or that are full.
                        if let Some(city) = state.map.cities.get(&city_id) {
                            if city.owners.contains(&my_id) || city.owners.len() >= 3 {
                                return iced::Task::none();
                            }
                        }
                    }
                }
                if let Some(pos) = self
                    .selected_build_cities
                    .iter()
                    .position(|c| c == &city_id)
                {
                    self.selected_build_cities.remove(pos);
                } else {
                    self.selected_build_cities.push(city_id);
                }
                self.refresh_build_preview();
            }
            Message::ClearBuildSelection => {
                self.selected_build_cities.clear();
                self.build_preview = BuildPreview::default();
            }
            Message::DoneBuilding => {
                if self.selected_build_cities.is_empty() {
                    self.send(Action::DoneBuilding);
                } else {
                    let city_ids = self.build_preview.ordered.clone();
                    self.send(Action::BuildCities { city_ids });
                }
            }
            Message::PowerCities => {
                // Fire all plants by default.
                if let Some(state) = &self.game_state {
                    if let Some(id) = self.my_id {
                        if let Some(player) = state.player(id) {
                            let plant_numbers: Vec<u8> =
                                player.plants.iter().map(|p| p.number).collect();
                            self.send(Action::PowerCities { plant_numbers });
                        }
                    }
                }
            }
            Message::MapZoom {
                factor,
                cursor_x,
                cursor_y,
            } => {
                let new_zoom = (self.map_zoom * factor).clamp(0.3, 8.0);
                let ratio = new_zoom / self.map_zoom;
                self.map_pan.x = cursor_x - (cursor_x - self.map_pan.x) * ratio;
                self.map_pan.y = cursor_y - (cursor_y - self.map_pan.y) * ratio;
                self.map_zoom = new_zoom;
            }
            Message::MapPan { dx, dy } => {
                self.map_pan.x += dx;
                self.map_pan.y += dy;
            }
        }
        iced::Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        match &self.screen {
            Screen::Connect(s) => s.view(),
            Screen::Game => {
                if let Some(state) = &self.game_state {
                    let my_id = self.my_id.unwrap_or(Uuid::nil());
                    let is_host = state.host_id() == Some(my_id);
                    if matches!(state.phase, powergrid_core::types::Phase::Lobby) {
                        screens::lobby_view(state, is_host, self.error_message.as_deref())
                    } else {
                        screens::game_view(
                            state,
                            my_id,
                            &self.bid_amount,
                            self.error_message.as_deref(),
                            self.map_zoom,
                            self.map_pan,
                            &self.selected_build_cities,
                            &self.build_preview,
                            &self.resource_cart,
                            self.resource_cart_cost,
                        )
                    }
                } else {
                    iced::widget::text("Connecting...").into()
                }
            }
        }
    }

    pub fn subscription(&self) -> Subscription<Message> {
        match &self.connect_url {
            Some(url) => connection::connect(url.clone()).map(Message::WsEvent),
            None => Subscription::none(),
        }
    }

    fn send(&mut self, action: Action) {
        if let Some(tx) = &mut self.ws_sender {
            let _ = tx.try_send(action);
        }
    }

    fn refresh_resource_preview(&mut self) {
        let Some(state) = &self.game_state else {
            self.resource_cart_cost = None;
            return;
        };
        let purchases: Vec<(Resource, u8)> = [
            Resource::Coal,
            Resource::Oil,
            Resource::Garbage,
            Resource::Uranium,
        ]
        .iter()
        .filter_map(|&r| {
            let amt = self.resource_cart.get(&r).copied().unwrap_or(0);
            if amt > 0 { Some((r, amt)) } else { None }
        })
        .collect();
        self.resource_cart_cost = if purchases.is_empty() {
            None
        } else {
            state.resources.batch_price(&purchases)
        };
    }

    fn refresh_build_preview(&mut self) {
        let Some(state) = &self.game_state else {
            self.build_preview = BuildPreview::default();
            return;
        };
        let Some(my_id) = self.my_id else {
            self.build_preview = BuildPreview::default();
            return;
        };
        let owned = state
            .player(my_id)
            .map(|p| p.cities.clone())
            .unwrap_or_default();
        self.build_preview = compute_build_preview(
            &state.map,
            &owned,
            &self.selected_build_cities,
            &state.map.cities,
        );
    }
}

// ---------------------------------------------------------------------------
// Build preview helpers
// ---------------------------------------------------------------------------

fn compute_build_preview(
    map: &Map,
    owned: &[String],
    selected: &[String],
    cities: &std::collections::HashMap<String, powergrid_core::map::City>,
) -> BuildPreview {
    if selected.is_empty() {
        return BuildPreview::default();
    }

    let ordered = optimal_build_order(map, owned, selected);

    let mut current_owned: Vec<String> = owned.to_vec();
    let mut total_route_cost = 0u32;
    let mut total_slot_cost = 0u32;
    let mut edges: HashSet<(String, String)> = HashSet::new();

    for city_id in &ordered {
        if let Some(path) = map.shortest_path_to(&current_owned, city_id) {
            total_route_cost = total_route_cost.saturating_add(path.cost);
            for edge in path.edges {
                edges.insert(edge);
            }
        }
        let slot_cost = cities
            .get(city_id)
            .map(|c| connection_cost(c.owners.len()))
            .unwrap_or(10);
        total_slot_cost = total_slot_cost.saturating_add(slot_cost);
        current_owned.push(city_id.clone());
    }

    let total_cost = total_route_cost.saturating_add(total_slot_cost);
    BuildPreview {
        ordered,
        total_route_cost,
        total_slot_cost,
        total_cost,
        edges,
    }
}

fn simulate_per_city_route_cost(map: &Map, owned: &[String], order: &[String]) -> u32 {
    let mut current_owned = owned.to_vec();
    let mut total = 0u32;
    for city in order {
        total = total.saturating_add(map.connection_cost_to(&current_owned, city).unwrap_or(0));
        current_owned.push(city.clone());
    }
    total
}

fn optimal_build_order(map: &Map, owned: &[String], selected: &[String]) -> Vec<String> {
    if selected.is_empty() {
        return Vec::new();
    }
    if selected.len() == 1 {
        return selected.to_vec();
    }

    if selected.len() <= 7 {
        // Brute-force all permutations (Heap's algorithm) to find the cheapest order.
        let mut arr: Vec<String> = selected.to_vec();
        let n = arr.len();
        let mut best_cost = u32::MAX;
        let mut best_order: Vec<String> = arr.clone();
        heap_permutations(&mut arr, n, &mut |perm: &[String]| {
            let cost = simulate_per_city_route_cost(map, owned, perm);
            if cost < best_cost {
                best_cost = cost;
                best_order = perm.to_vec();
            }
        });
        best_order
    } else {
        // Greedy: always pick the currently cheapest city to extend the network.
        let mut remaining: Vec<String> = selected.to_vec();
        let mut current_owned: Vec<String> = owned.to_vec();
        let mut order = Vec::new();
        while !remaining.is_empty() {
            let best_idx = remaining
                .iter()
                .enumerate()
                .min_by_key(|(_, city)| {
                    map.connection_cost_to(&current_owned, city)
                        .unwrap_or(u32::MAX)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let city = remaining.remove(best_idx);
            current_owned.push(city.clone());
            order.push(city);
        }
        order
    }
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    name: Option<String>,
    color: Option<PlayerColor>,
    url: Option<String>,
}

impl CliArgs {
    fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut name = None;
        let mut color = None;
        let mut url = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--name" => name = args.next(),
                "--color" => {
                    color = args.next().and_then(|s| match s.to_lowercase().as_str() {
                        "red" => Some(PlayerColor::Red),
                        "blue" => Some(PlayerColor::Blue),
                        "green" => Some(PlayerColor::Green),
                        "yellow" => Some(PlayerColor::Yellow),
                        "purple" => Some(PlayerColor::Purple),
                        "black" => Some(PlayerColor::Black),
                        _ => {
                            eprintln!("Unknown color '{}'. Valid: red, blue, green, yellow, purple, black", s);
                            None
                        }
                    });
                }
                "--url" => url = args.next(),
                other => eprintln!("Unknown argument: {other}"),
            }
        }

        Self { name, color, url }
    }
}

/// Heap's algorithm: generate all permutations of `arr[0..k]`, calling `callback` for each.
fn heap_permutations<T: Clone>(arr: &mut Vec<T>, k: usize, callback: &mut impl FnMut(&[T])) {
    if k == 1 {
        callback(arr);
        return;
    }
    for i in 0..k {
        heap_permutations(arr, k - 1, callback);
        if k.is_multiple_of(2) {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
    }
}
