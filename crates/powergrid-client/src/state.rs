use bevy::prelude::*;
use crossbeam_channel::Sender;
use powergrid_core::{
    actions::Action,
    connection_cost,
    map::{City, Map},
    types::{Phase, PlayerColor, PlayerId, Resource},
    GameState,
};
use std::collections::{HashMap, HashSet};

/// A snapshot of every player's city count at the end of a round.
pub type CitySnapshot = Vec<(PlayerId, usize)>;

// ---------------------------------------------------------------------------
// AppState resource
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Screen {
    Connect,
    Game,
}

#[derive(Resource)]
pub struct AppState {
    pub screen: Screen,

    // Connect fields
    pub connect_url: String,
    pub player_name: String,
    pub selected_color: PlayerColor,

    // Connection state
    pub connected: bool,
    pub my_id: Option<PlayerId>,
    /// Name + color to send on first Welcome.
    pub pending_join: Option<(String, PlayerColor)>,

    // Game state
    pub game_state: Option<GameState>,
    pub error_message: Option<String>,

    // Map viewport
    pub map_zoom: f32,
    pub map_offset: Vec2,

    // Build phase
    pub selected_build_cities: Vec<String>,
    pub build_preview: BuildPreview,

    // Buy resources
    pub resource_cart: HashMap<Resource, u8>,
    pub resource_cart_cost: Option<u32>,

    // Auction
    pub bid_amount: u32,

    // City count history: one CitySnapshot per round recorded so far.
    pub city_history: Vec<CitySnapshot>,
    last_recorded_round: u32,
}

impl AppState {
    pub fn new(cli: CliArgs) -> Self {
        let connect_url = cli
            .url
            .unwrap_or_else(|| "ws://localhost:3000/ws".to_string());
        let player_name = cli.name.unwrap_or_default();
        let selected_color = cli.color.unwrap_or(PlayerColor::Red);

        // Auto-connect when all three args provided.
        let (screen, pending_join) =
            if !player_name.is_empty() && cli.color.is_some() && !connect_url.is_empty() {
                (Screen::Connect, Some((player_name.clone(), selected_color)))
            } else {
                (Screen::Connect, None)
            };

        Self {
            screen,
            connect_url,
            player_name,
            selected_color,
            connected: false,
            my_id: None,
            pending_join,
            game_state: None,
            error_message: None,
            map_zoom: 1.0,
            map_offset: Vec2::ZERO,
            selected_build_cities: Vec::new(),
            build_preview: BuildPreview::default(),
            resource_cart: HashMap::new(),
            resource_cart_cost: None,
            bid_amount: 0,
            city_history: Vec::new(),
            last_recorded_round: 0,
        }
    }

    /// Called every time a StateUpdate arrives.
    pub fn handle_state_update(&mut self, gs: GameState, action_tx: &Sender<Action>) {
        // Clear build selection once it's no longer our build turn.
        let still_my_build = self
            .my_id
            .map(|id| {
                matches!(&gs.phase, Phase::BuildCities { remaining }
                    if remaining.first() == Some(&id))
            })
            .unwrap_or(false);
        if !still_my_build {
            self.selected_build_cities.clear();
            self.build_preview = BuildPreview::default();
        }

        // Clear resource cart once it's no longer our buy turn.
        let still_my_buy = self
            .my_id
            .map(|id| {
                matches!(&gs.phase, Phase::BuyResources { remaining }
                    if remaining.first() == Some(&id))
            })
            .unwrap_or(false);
        if !still_my_buy {
            self.resource_cart.clear();
            self.resource_cart_cost = None;
        }

        // Record city counts when the round number advances (or on first state).
        if self.city_history.is_empty() || gs.round > self.last_recorded_round {
            let snapshot: CitySnapshot =
                gs.players.iter().map(|p| (p.id, p.city_count())).collect();
            self.city_history.push(snapshot);
            self.last_recorded_round = gs.round;
        }

        // Move to game screen on first state.
        self.screen = Screen::Game;
        self.game_state = Some(gs);
        self.error_message = None;

        // Keep build preview fresh after state update.
        self.refresh_build_preview();
        self.refresh_resource_preview();

        let _ = action_tx; // reserved for future auto-actions
    }

    // -----------------------------------------------------------------------
    // Resource cart
    // -----------------------------------------------------------------------

    pub fn add_to_cart(&mut self, resource: Resource) {
        let Some(state) = &self.game_state else {
            return;
        };
        let Some(my_id) = self.my_id else { return };
        let Some(player) = state.player(my_id) else {
            return;
        };

        let cart_count = self.resource_cart.get(&resource).copied().unwrap_or(0);
        let new_count = cart_count + 1;
        if state.resources.available(resource) < new_count {
            return;
        }
        let mut sim = player.clone();
        for (&r, &amt) in &self.resource_cart {
            sim.resources.add(r, amt);
        }
        if !sim.can_add_resource(resource, 1) {
            return;
        }
        *self.resource_cart.entry(resource).or_insert(0) += 1;
        self.refresh_resource_preview();
    }

    pub fn remove_from_cart(&mut self, resource: Resource) {
        let count = self.resource_cart.entry(resource).or_insert(0);
        if *count > 0 {
            *count -= 1;
        }
        self.refresh_resource_preview();
    }

    pub fn clear_cart(&mut self) {
        self.resource_cart.clear();
        self.resource_cart_cost = None;
    }

    pub fn cart_purchases(&self) -> Vec<(Resource, u8)> {
        [
            Resource::Coal,
            Resource::Oil,
            Resource::Garbage,
            Resource::Uranium,
        ]
        .iter()
        .filter_map(|&r| {
            let amt = self.resource_cart.get(&r).copied().unwrap_or(0);
            (amt > 0).then_some((r, amt))
        })
        .collect()
    }

    fn refresh_resource_preview(&mut self) {
        let Some(state) = &self.game_state else {
            self.resource_cart_cost = None;
            return;
        };
        let purchases = self.cart_purchases();
        self.resource_cart_cost = if purchases.is_empty() {
            None
        } else {
            state.resources.batch_price(&purchases)
        };
    }

    // -----------------------------------------------------------------------
    // Build preview
    // -----------------------------------------------------------------------

    pub fn toggle_build_city(&mut self, city_id: String) {
        let Some(state) = &self.game_state else {
            return;
        };
        let Some(my_id) = self.my_id else { return };

        if let Some(city) = state.map.cities.get(&city_id) {
            if city.owners.contains(&my_id) || city.owners.len() >= 3 {
                return;
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

    pub fn clear_build_selection(&mut self) {
        self.selected_build_cities.clear();
        self.build_preview = BuildPreview::default();
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
// BuildPreview
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct BuildPreview {
    pub ordered: Vec<String>,
    pub total_route_cost: u32,
    pub total_slot_cost: u32,
    pub total_cost: u32,
    pub edges: HashSet<(String, String)>,
}

fn compute_build_preview(
    map: &Map,
    owned: &[String],
    selected: &[String],
    cities: &HashMap<String, City>,
) -> BuildPreview {
    if selected.is_empty() {
        return BuildPreview::default();
    }

    let ordered = optimal_build_order(map, owned, selected);
    let mut current_owned = owned.to_vec();
    let mut total_route_cost = 0u32;
    let mut total_slot_cost = 0u32;
    let mut edges = HashSet::new();

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

    BuildPreview {
        ordered,
        total_route_cost,
        total_slot_cost,
        total_cost: total_route_cost.saturating_add(total_slot_cost),
        edges,
    }
}

fn simulate_route_cost(map: &Map, owned: &[String], order: &[String]) -> u32 {
    let mut current = owned.to_vec();
    let mut total = 0u32;
    for city in order {
        total = total.saturating_add(map.connection_cost_to(&current, city).unwrap_or(0));
        current.push(city.clone());
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
        let mut arr = selected.to_vec();
        let n = arr.len();
        let mut best_cost = u32::MAX;
        let mut best = arr.clone();
        heap_permutations(&mut arr, n, &mut |perm: &[String]| {
            let cost = simulate_route_cost(map, owned, perm);
            if cost < best_cost {
                best_cost = cost;
                best = perm.to_vec();
            }
        });
        best
    } else {
        let mut remaining = selected.to_vec();
        let mut current = owned.to_vec();
        let mut order = Vec::new();
        while !remaining.is_empty() {
            let best_idx = remaining
                .iter()
                .enumerate()
                .min_by_key(|(_, city)| map.connection_cost_to(&current, city).unwrap_or(u32::MAX))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let city = remaining.remove(best_idx);
            current.push(city.clone());
            order.push(city);
        }
        order
    }
}

fn heap_permutations<T: Clone>(arr: &mut Vec<T>, k: usize, cb: &mut impl FnMut(&[T])) {
    if k == 1 {
        cb(arr);
        return;
    }
    for i in 0..k {
        heap_permutations(arr, k - 1, cb);
        if k.is_multiple_of(2) {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

pub struct CliArgs {
    pub name: Option<String>,
    pub color: Option<PlayerColor>,
    pub url: Option<String>,
    pub auto_connect: bool,
    pub windowed: bool,
}

impl CliArgs {
    pub fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut name = None;
        let mut color = None;
        let mut url = None;
        let mut windowed = false;

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
                        "white" => Some(PlayerColor::White),
                        other => {
                            eprintln!("Unknown color '{other}'");
                            None
                        }
                    });
                }
                "--url" => url = args.next(),
                "-w" | "--windowed" => windowed = true,
                other => eprintln!("Unknown argument: {other}"),
            }
        }

        let auto_connect = name.is_some() && color.is_some() && url.is_some();
        Self {
            name,
            color,
            url,
            auto_connect,
            windowed,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: map PlayerColor to an egui Color32
// ---------------------------------------------------------------------------

pub fn player_color_to_egui(color: PlayerColor) -> egui::Color32 {
    match color {
        PlayerColor::Red => egui::Color32::from_rgb(220, 40, 40),
        PlayerColor::Blue => egui::Color32::from_rgb(40, 80, 220),
        PlayerColor::Green => egui::Color32::from_rgb(40, 180, 60),
        PlayerColor::Yellow => egui::Color32::from_rgb(240, 200, 20),
        PlayerColor::Purple => egui::Color32::from_rgb(150, 30, 200),
        PlayerColor::White => egui::Color32::from_rgb(240, 240, 240),
    }
}
