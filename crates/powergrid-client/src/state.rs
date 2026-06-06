use crate::{
    auth::{AuthPendingSlot, SavedCredentials},
    peer_hints::{LocalHintTracker, PeerHints},
};
use egui::Vec2;
use powergrid_core::{
    actions::RoomSummary,
    connection_cost,
    map::Map,
    types::{BotDifficulty, Phase, PlantKind, PlayerColor, PlayerId, Resource, ResourceMarket},
    GameStateView,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// A snapshot of every player's city count at the end of a round.
pub type CitySnapshot = Vec<(PlayerId, usize)>;

/// Which tab is selected in the bottom-right info panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottomTab {
    EventLog,
    CityGraph,
    Replenish,
    Payout,
}

impl BottomTab {
    pub fn label(self) -> &'static str {
        match self {
            BottomTab::EventLog => "EVENTS",
            BottomTab::CityGraph => "CITIES",
            BottomTab::Replenish => "REPLENISH",
            BottomTab::Payout => "PAYOUT",
        }
    }
}

// ---------------------------------------------------------------------------
// AppState resource
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Screen {
    MainMenu,
    LocalSetup,
    Login,
    Register,
    RoomBrowser,
    Game,
}

pub struct AppState {
    pub screen: Screen,

    // Auth fields
    pub auth_token: Option<String>,
    pub auth_username: Option<String>,
    pub auth_user_id: Option<PlayerId>,
    pub login_identifier: String,
    pub login_password: String,
    pub register_email: String,
    pub register_username: String,
    pub register_password: String,
    pub auth_error: Option<String>,
    pub auth_in_flight: bool,
    /// Shared slot written by auth background threads, read each frame.
    pub auth_pending: AuthPendingSlot,

    // Connect fields
    pub server_name: String,
    pub port: u16,
    pub selected_color: PlayerColor,

    // Connection state
    pub connected: bool,
    pub pending_connect: bool,
    pub my_id: Option<PlayerId>,

    // Lobby / room state
    pub current_room: Option<String>,
    pub room_list: Vec<RoomSummary>,
    /// Input field used in the room browser to create or join a room.
    pub room_name_input: String,
    /// If set, auto-create this room name after authentication (CLI arg).
    pub auto_room: Option<String>,
    /// Bot name + color + difficulty inputs for the Add Bot form (online lobby).
    pub bot_name_input: String,
    pub bot_color_input: PlayerColor,
    pub bot_difficulty_input: BotDifficulty,

    // Game state
    pub game_state: Option<GameStateView>,
    /// Static map received once on RoomJoined. City owners kept current via
    /// Arc::make_mut in handle_state_update.
    pub map: Option<Arc<Map>>,
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
    pub bid_plant_number: Option<u8>,
    /// Last committed bid amount per player for the currently auctioned plant.
    /// Derived from sequential StateUpdates — when active_bid.highest_bidder changes,
    /// we record (highest_bidder, amount). Cleared on plant change or auction end.
    pub auction_last_bids: HashMap<PlayerId, u32>,
    /// Tracks which plant the `auction_last_bids` map applies to.
    pub auction_last_plant: Option<u8>,

    // Resource-discard phase (hybrid shared-slot overflow)
    pub discard_gas: u8,
    pub discard_oil: u8,

    // PowerCitiesFuel phase (hybrid fuel split during bureaucracy)
    pub power_fuel_gas: u8,

    // Bureaucracy plant-selection scratch state
    pub power_selected_plants: HashSet<u8>,
    pub power_selected_initialised: bool,

    // City count history: one CitySnapshot per round recorded so far.
    pub city_history: Vec<CitySnapshot>,
    last_recorded_round: u32,

    /// Elapsed seconds when the last ListRooms was sent.
    pub room_list_last_refresh: f64,

    // ESC menu overlay
    pub menu_open: bool,
    // Bottom-right info panel (Space toggles)
    pub bottom_panel_open: bool,
    pub bottom_panel_tab: BottomTab,

    // Floating action panel positioning (captured from top-bar column rects each frame)
    // Indices: 0=Auction, 1=BuyResources, 2=BuildCities, 3=Bureaucracy
    pub phase_column_rects: [Option<egui::Rect>; 4],
    pub top_panel_bottom: f32,
    pub left_panel_width: f32,

    // Window mode (kept in sync with the actual viewport)
    pub fullscreen: bool,

    // When true, skip all disk reads/writes for credentials and preferences.
    pub no_preferences: bool,

    // Local play setup
    pub local_name: String,
    pub local_color: PlayerColor,
    /// Per-bot difficulty for local play. Length = number of bots (1–5).
    pub local_bots: Vec<BotDifficulty>,

    // Peer hint display (received from other clients)
    pub peer_hints: PeerHints,
    // Tracks local selection changes for outgoing hint emission
    pub hint_tracker: LocalHintTracker,

    /// Set to `true` when it transitions to being the local player's turn.
    /// Consumed (and gated on online-only) by `PowerGridApp::update` in main.rs.
    pub play_turn_sound: bool,
}

impl AppState {
    pub fn new(cli: CliArgs) -> Self {
        let server_name = cli
            .server
            .unwrap_or_else(|| "powergrid.onyxoryx.net".to_string());
        let port = cli.port.unwrap_or(3000);
        let selected_color = cli.color.unwrap_or(PlayerColor::Red);

        // Load saved credentials; always start on MainMenu but seed auth fields for Online play.
        let saved = if cli.no_preferences {
            None
        } else {
            crate::auth::load_credentials(&server_name, port)
        };
        let (auth_token, auth_username, auth_user_id) = if let Some(ref c) = saved {
            (
                Some(c.token.clone()),
                Some(c.username.clone()),
                Some(c.user_id),
            )
        } else {
            (None, None, None)
        };
        let screen = Screen::MainMenu;

        // CLI --windowed overrides saved preference; otherwise use saved (default: fullscreen).
        let fullscreen = if cli.windowed {
            false
        } else if cli.no_preferences {
            crate::auth::UserPreferences::default().fullscreen
        } else {
            crate::auth::load_preferences().fullscreen
        };

        Self {
            screen,
            auth_token,
            auth_username,
            auth_user_id,
            login_identifier: String::new(),
            login_password: String::new(),
            register_email: String::new(),
            register_username: String::new(),
            register_password: String::new(),
            auth_error: None,
            auth_in_flight: false,
            auth_pending: AuthPendingSlot::new(),
            server_name,
            port,
            selected_color,
            connected: false,
            pending_connect: false,
            my_id: None,
            current_room: None,
            room_list: Vec::new(),
            room_name_input: String::new(),
            auto_room: cli.room,
            bot_name_input: String::new(),
            bot_color_input: PlayerColor::Blue,
            bot_difficulty_input: BotDifficulty::Normal,
            game_state: None,
            map: None,
            error_message: None,
            map_zoom: 1.0,
            map_offset: Vec2::ZERO,
            selected_build_cities: Vec::new(),
            build_preview: BuildPreview::default(),
            resource_cart: HashMap::new(),
            resource_cart_cost: None,
            bid_amount: 0,
            bid_plant_number: None,
            auction_last_bids: HashMap::new(),
            auction_last_plant: None,
            discard_gas: 0,
            discard_oil: 0,
            power_fuel_gas: 0,
            power_selected_plants: HashSet::new(),
            power_selected_initialised: false,
            city_history: Vec::new(),
            last_recorded_round: 0,
            room_list_last_refresh: f64::NEG_INFINITY,
            menu_open: false,
            bottom_panel_open: false,
            bottom_panel_tab: BottomTab::EventLog,
            fullscreen,
            no_preferences: cli.no_preferences,
            local_name: "Human".to_string(),
            local_color: PlayerColor::Red,
            local_bots: vec![
                BotDifficulty::Normal,
                BotDifficulty::Normal,
                BotDifficulty::Normal,
            ],
            peer_hints: PeerHints::default(),
            hint_tracker: LocalHintTracker::new(),
            phase_column_rects: [None; 4],
            top_panel_bottom: 0.0,
            left_panel_width: 220.0,
            play_turn_sound: false,
        }
    }

    pub fn ws_url(&self) -> String {
        format!("ws://{}:{}/ws", self.server_name, self.port)
    }

    /// Apply a successful auth result: store credentials and advance to Connect screen.
    pub fn apply_auth_success(&mut self, creds: SavedCredentials) {
        if !self.no_preferences {
            if let Err(e) = crate::auth::save_credentials(&creds) {
                tracing::warn!("Failed to save credentials: {e}");
            }
        }
        self.auth_token = Some(creds.token);
        self.auth_username = Some(creds.username);
        self.auth_user_id = Some(creds.user_id);
        self.server_name = creds.server;
        self.port = creds.port;
        self.auth_error = None;
        self.auth_in_flight = false;
        self.pending_connect = true;
        self.screen = Screen::RoomBrowser;
    }

    /// Fire-and-forget logout: spawns a background thread to hit the auth endpoint,
    /// then updates state via the shared auth_pending slot.
    pub fn trigger_logout(&mut self) {
        use crate::auth::{do_logout, AuthEvent};
        if let (Some(token), server, port) =
            (self.auth_token.clone(), self.server_name.clone(), self.port)
        {
            let slot = self.auth_pending.0.clone();
            std::thread::spawn(move || {
                do_logout(&server, port, &token);
                *slot.lock().unwrap() = Some(AuthEvent::LoggedOut);
            });
            self.auth_in_flight = true;
        } else {
            self.logout();
        }
    }

    /// Clear all auth state and return to Login screen.
    pub fn logout(&mut self) {
        if !self.no_preferences {
            if let Err(e) = crate::auth::clear_credentials(&self.server_name, self.port) {
                tracing::warn!("Failed to clear credentials: {e}");
            }
        }
        self.auth_token = None;
        self.auth_username = None;
        self.auth_user_id = None;
        self.auth_error = None;
        self.auth_in_flight = false;
        self.connected = false;
        self.pending_connect = false;
        self.my_id = None;
        self.current_room = None;
        self.game_state = None;
        self.map = None;
        self.screen = Screen::Login;
    }

    /// Called every time a StateUpdate arrives.
    pub fn handle_state_update(&mut self, view: GameStateView) {
        // Update city owners in the local map from the view.
        if let Some(map_arc) = &mut self.map {
            let map = Arc::make_mut(map_arc);
            for city in map.cities.values_mut() {
                city.owners.clear();
            }
            for (city_id, owners) in &view.city_owners {
                if let Some(city) = map.cities.get_mut(city_id) {
                    city.owners = owners.clone();
                }
            }
        }

        // Clear build selection once it's no longer our build turn.
        let still_my_build = self
            .my_id
            .map(|id| {
                matches!(&view.phase, Phase::BuildCities { remaining }
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
                matches!(&view.phase, Phase::BuyResources { remaining }
                    if remaining.first() == Some(&id))
            })
            .unwrap_or(false);
        if !still_my_buy {
            self.resource_cart.clear();
            self.resource_cart_cost = None;
        }

        // Clear discard-resource counters once we leave that phase.
        let still_my_discard = self
            .my_id
            .map(|id| matches!(&view.phase, Phase::DiscardResource { player, .. } if *player == id))
            .unwrap_or(false);
        if !still_my_discard {
            self.discard_gas = 0;
            self.discard_oil = 0;
        }

        // Clear power-fuel counter once we leave that phase.
        let still_my_fuel = self
            .my_id
            .map(|id| matches!(&view.phase, Phase::PowerCitiesFuel { player, .. } if *player == id))
            .unwrap_or(false);
        if !still_my_fuel {
            self.power_fuel_gas = 0;
        }

        // Clear plant-selection scratch state when not in Bureaucracy.
        let still_bureaucracy = matches!(&view.phase, Phase::Bureaucracy { .. });
        if !still_bureaucracy {
            self.power_selected_plants.clear();
            self.power_selected_initialised = false;
        }

        // Track per-player committed bid amounts for the current auction.
        // Only the highest_bidder changes each round, so we accumulate entries over time.
        match &view.phase {
            Phase::Auction {
                active_bid: Some(bid),
                ..
            } => {
                if self.auction_last_plant != Some(bid.plant_number) {
                    self.auction_last_bids.clear();
                    self.auction_last_plant = Some(bid.plant_number);
                }
                self.auction_last_bids
                    .insert(bid.highest_bidder, bid.amount);
            }
            _ => {
                self.auction_last_bids.clear();
                self.auction_last_plant = None;
            }
        }

        // Clear peer hints when the phase changes (stale selections from the previous phase).
        let phase_changed = self
            .game_state
            .as_ref()
            .map(|gs| std::mem::discriminant(&gs.phase) != std::mem::discriminant(&view.phase))
            .unwrap_or(true);
        if phase_changed {
            self.peer_hints.clear();
        }

        // Record city counts when the round number advances (or on first state).
        if self.city_history.is_empty() || view.round > self.last_recorded_round {
            let snapshot: CitySnapshot = view
                .players
                .iter()
                .map(|p| (p.id, view.player_city_count(p.id)))
                .collect();
            self.city_history.push(snapshot);
            self.last_recorded_round = view.round;
        }

        // Detect transition into "it's now my turn" for the turn-notification sound.
        if let Some(my_id) = self.my_id {
            let was_active = self
                .game_state
                .as_ref()
                .map(|gs| crate::ui::is_active_player(gs, my_id))
                .unwrap_or(false);
            if !was_active && crate::ui::is_active_player(&view, my_id) {
                self.play_turn_sound = true;
            }
        }

        // Move to game screen on first state.
        self.screen = Screen::Game;
        self.game_state = Some(view);
        self.error_message = None;

        // Keep build preview fresh after state update.
        self.refresh_build_preview();
        self.refresh_resource_preview();
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

    pub fn clear_cart(&mut self) {
        self.resource_cart.clear();
        self.resource_cart_cost = None;
    }

    pub fn cart_purchases(&self) -> Vec<(Resource, u8)> {
        [
            Resource::Coal,
            Resource::Oil,
            Resource::Gas,
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

    /// Set the cart amount for a single resource to exactly `target`, replacing any
    /// existing amount for that resource. `add_to_cart` is reused so market availability
    /// and storage capacity are still enforced; the cart will be capped accordingly.
    pub fn set_cart_amount(&mut self, resource: Resource, target: u8) {
        self.resource_cart.remove(&resource);
        for _ in 0..target {
            let before = self.resource_cart.get(&resource).copied().unwrap_or(0);
            self.add_to_cart(resource);
            let after = self.resource_cart.get(&resource).copied().unwrap_or(0);
            if after == before {
                break;
            }
        }
        self.refresh_resource_preview();
    }

    /// Replace the cart with enough resources to fire all fuel-burning plants `sets` times.
    /// Accounts for already-owned resources. Cheaper resource wins for hybrid plants.
    pub fn fill_cart_for_sets(&mut self, sets: u8) {
        let Some(state) = &self.game_state else {
            return;
        };
        let Some(my_id) = self.my_id else { return };
        let Some(player) = state.player(my_id) else {
            return;
        };
        let targets = compute_set_cart(player, &state.resources, sets);
        self.clear_cart();
        for (resource, amount) in targets {
            for _ in 0..amount {
                self.add_to_cart(resource);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Build preview
    // -----------------------------------------------------------------------

    pub fn toggle_build_city(&mut self, city_id: String) {
        // Deselect unconditionally so the user can always undo a selection.
        if let Some(pos) = self
            .selected_build_cities
            .iter()
            .position(|c| c == &city_id)
        {
            self.selected_build_cities.remove(pos);
            self.refresh_build_preview();
            return;
        }

        let Some(my_id) = self.my_id else { return };
        let Some(gs) = self.game_state.as_ref() else {
            return;
        };
        let Some(map) = self.map.as_deref() else {
            return;
        };

        if !gs.is_city_active(&city_id, map) {
            return;
        }

        let Some(city) = map.cities.get(&city_id) else {
            return;
        };

        if city.owners.contains(&my_id) || city.owners.len() >= gs.step as usize {
            return;
        }

        let owned: Vec<String> = gs.player_cities(my_id).into_iter().cloned().collect();
        let mut combined: Vec<String> = owned;
        combined.extend(self.selected_build_cities.iter().cloned());
        if map.connection_cost_to(&combined, &city_id).is_none() {
            return;
        }

        self.selected_build_cities.push(city_id);
        self.refresh_build_preview();
    }

    pub fn clear_build_selection(&mut self) {
        self.selected_build_cities.clear();
        self.build_preview = BuildPreview::default();
    }

    fn refresh_build_preview(&mut self) {
        let (Some(gs), Some(my_id), Some(map)) =
            (self.game_state.as_ref(), self.my_id, self.map.as_deref())
        else {
            self.build_preview = BuildPreview::default();
            return;
        };
        let owned: Vec<String> = gs.player_cities(my_id).into_iter().cloned().collect();
        let city_owners = &gs.city_owners;
        let selected = self.selected_build_cities.clone();
        self.build_preview = compute_build_preview(map, &owned, &selected, city_owners);
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
    city_owners: &HashMap<String, Vec<PlayerId>>,
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
        let slot_cost = city_owners
            .get(city_id)
            .map(|owners| connection_cost(owners.len()))
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
// Resource set computation
// ---------------------------------------------------------------------------

/// Compute how many of each resource to add to the cart to fire all plants `sets` times.
///
/// Algorithm:
/// 1. Dedicated plants (Coal/Oil/Gas/Uranium): buy `sets * cost - owned`, capped by market.
/// 2. Hybrid (GasOrOil) plants: after dedicated buys are simulated, pick the cheaper of gas
///    or oil for each remaining unit, tie-breaking to oil.
///
/// Returns a deterministic `Vec<(Resource, u8)>` in Coal/Oil/Gas/Uranium order.
/// Amounts are advisory — the caller should still run them through `add_to_cart` to
/// enforce storage capacity and market constraints.
fn compute_set_cart(
    player: &powergrid_core::types::Player,
    market: &ResourceMarket,
    sets: u8,
) -> Vec<(Resource, u8)> {
    let coal_only_need: u8 = player
        .plants
        .iter()
        .filter(|p| p.kind == PlantKind::Coal)
        .map(|p| p.cost.saturating_mul(sets))
        .fold(0u8, |a, b| a.saturating_add(b));
    let oil_only_need: u8 = player
        .plants
        .iter()
        .filter(|p| p.kind == PlantKind::Oil)
        .map(|p| p.cost.saturating_mul(sets))
        .fold(0u8, |a, b| a.saturating_add(b));
    let gas_need: u8 = player
        .plants
        .iter()
        .filter(|p| p.kind == PlantKind::Gas)
        .map(|p| p.cost.saturating_mul(sets))
        .fold(0u8, |a, b| a.saturating_add(b));
    let uranium_need: u8 = player
        .plants
        .iter()
        .filter(|p| p.kind == PlantKind::Uranium)
        .map(|p| p.cost.saturating_mul(sets))
        .fold(0u8, |a, b| a.saturating_add(b));
    let hybrid_need: u8 = player
        .plants
        .iter()
        .filter(|p| p.kind == PlantKind::GasOrOil)
        .map(|p| p.cost.saturating_mul(sets))
        .fold(0u8, |a, b| a.saturating_add(b));

    let mut sim_market = market.clone();
    let mut sim_player = player.clone();
    let mut result: HashMap<Resource, u8> = HashMap::new();

    // Buy the hard-requirement amount for each dedicated resource, capped by what the
    // simulated market + storage allow.
    for (resource, need) in [
        (Resource::Coal, coal_only_need),
        (Resource::Oil, oil_only_need),
        (Resource::Gas, gas_need),
        (Resource::Uranium, uranium_need),
    ] {
        let owned = sim_player.resources.get(resource);
        let want = need.saturating_sub(owned);
        if want == 0 {
            continue;
        }
        let cap = want.min(sim_market.available(resource));
        // Walk down from cap to find the largest block the player can store.
        for n in (1..=cap).rev() {
            if sim_player.can_add_resource(resource, n) {
                *result.entry(resource).or_insert(0) += n;
                sim_market.take(resource, n);
                sim_player.resources.add(resource, n);
                break;
            }
        }
    }

    // Determine how much hybrid fuel is still needed after dedicated plants are covered.
    let leftover_gas = sim_player.resources.gas.saturating_sub(gas_need);
    let leftover_oil = sim_player.resources.oil.saturating_sub(oil_only_need);
    let hybrid_remaining = hybrid_need.saturating_sub(leftover_gas.saturating_add(leftover_oil));

    // For each remaining hybrid unit, pick the cheaper of gas or oil.
    for _ in 0..hybrid_remaining {
        let gas_ok = sim_market.available(Resource::Gas) >= 1
            && sim_player.can_add_resource(Resource::Gas, 1);
        let oil_ok = sim_market.available(Resource::Oil) >= 1
            && sim_player.can_add_resource(Resource::Oil, 1);
        if !gas_ok && !oil_ok {
            break;
        }
        let gas_price = if gas_ok {
            sim_market.price(Resource::Gas, 1).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        let oil_price = if oil_ok {
            sim_market.price(Resource::Oil, 1).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        // Tie-break to oil (matches the bot strategy convention).
        let pick = if oil_price <= gas_price {
            Resource::Oil
        } else {
            Resource::Gas
        };
        *result.entry(pick).or_insert(0) += 1;
        sim_market.take(pick, 1);
        sim_player.resources.add(pick, 1);
    }

    // Return in canonical Coal/Oil/Gas/Uranium order.
    [
        Resource::Coal,
        Resource::Oil,
        Resource::Gas,
        Resource::Uranium,
    ]
    .iter()
    .filter_map(|&r| {
        let amt = result.get(&r).copied().unwrap_or(0);
        (amt > 0).then_some((r, amt))
    })
    .collect()
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

pub struct CliArgs {
    pub color: Option<PlayerColor>,
    pub server: Option<String>,
    pub port: Option<u16>,
    /// Auto-create/join this room name on connect (for CLI-driven testing).
    pub room: Option<String>,
    pub windowed: bool,
    pub no_preferences: bool,
}

impl CliArgs {
    pub fn parse() -> Self {
        let mut args = std::env::args().skip(1);
        let mut color = None;
        let mut server = None;
        let mut port = None;
        let mut room = None;
        let mut windowed = false;
        let mut no_preferences = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    println!(
                        "Usage: powergrid-client [options]

Options:
  --color <color>       Auto-select player color on connect
                          Choices: red, blue, green, yellow, purple, white
  --server <host>       Server hostname to connect to
  --port <port>         Server port
  --room <name>         Auto-create/join this room on connect
  -w, --windowed        Run in windowed mode (default: borderless fullscreen)
  -n, --no-preferences  Don't load or save credentials/preferences (for testing)
  -h, --help            Show this help message"
                    );
                    std::process::exit(0);
                }
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
                "--server" => server = args.next(),
                "--port" => {
                    port = args.next().and_then(|s| {
                        s.parse::<u16>().ok().or_else(|| {
                            eprintln!("Invalid port '{s}'");
                            None
                        })
                    });
                }
                "--room" => room = args.next(),
                "-w" | "--windowed" => windowed = true,
                "-n" | "--no-preferences" => no_preferences = true,
                other => eprintln!("Unknown argument: {other}"),
            }
        }

        Self {
            color,
            server,
            port,
            room,
            windowed,
            no_preferences,
        }
    }
}

// ---------------------------------------------------------------------------
// Drain auth results written by background threads
// ---------------------------------------------------------------------------

pub fn process_auth_events(state: &mut AppState) {
    if !state.auth_in_flight {
        return;
    }
    let slot = state.auth_pending.clone();
    if let Some(event) = slot.take() {
        match event {
            crate::auth::AuthEvent::Success(creds) => {
                state.apply_auth_success(creds);
            }
            crate::auth::AuthEvent::Failure(msg) => {
                state.auth_error = Some(msg);
                state.auth_in_flight = false;
            }
            crate::auth::AuthEvent::LoggedOut => {
                state.logout();
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_core::types::{PlantKind, Player, PlayerColor, PlayerResources, PowerPlant};

    fn make_plant(kind: PlantKind, cost: u8) -> PowerPlant {
        PowerPlant {
            number: 10,
            kind,
            cost,
            cities: 2,
        }
    }

    fn make_player(plants: Vec<PowerPlant>, resources: PlayerResources) -> Player {
        let mut p = Player::new("test".to_string(), PlayerColor::Red);
        p.plants = plants;
        p.resources = resources;
        p
    }

    #[test]
    fn single_coal_plant_no_stock() {
        let player = make_player(
            vec![make_plant(PlantKind::Coal, 3)],
            PlayerResources::default(),
        );
        let market = ResourceMarket::initial();
        let result = compute_set_cart(&player, &market, 1);
        assert_eq!(result, vec![(Resource::Coal, 3)]);
    }

    #[test]
    fn single_coal_plant_two_sets() {
        let player = make_player(
            vec![make_plant(PlantKind::Coal, 2)],
            PlayerResources::default(),
        );
        let market = ResourceMarket::initial();
        let result = compute_set_cart(&player, &market, 2);
        // 2 sets = plant.cost * 2 = 4
        assert_eq!(result, vec![(Resource::Coal, 4)]);
    }

    #[test]
    fn coal_plant_with_partial_stock_subtracts_owned() {
        let resources = PlayerResources {
            coal: 2,
            ..Default::default()
        };
        let player = make_player(vec![make_plant(PlantKind::Coal, 3)], resources);
        let market = ResourceMarket::initial();
        let result = compute_set_cart(&player, &market, 1);
        // Needs 3, has 2, so buy 1.
        assert_eq!(result, vec![(Resource::Coal, 1)]);
    }

    #[test]
    fn hybrid_plant_picks_cheaper_resource() {
        // No dedicated gas/oil plants; one GasOrOil hybrid.
        // Initial market: gas=6 (cheapest slot price ~$7), oil=18 (cheapest slot price $3/unit).
        // Oil is cheaper, so both units should come from oil.
        let player = make_player(
            vec![make_plant(PlantKind::GasOrOil, 2)],
            PlayerResources::default(),
        );
        let market = ResourceMarket::initial();
        let result = compute_set_cart(&player, &market, 1);
        assert_eq!(result, vec![(Resource::Oil, 2)]);
    }

    #[test]
    fn hybrid_plant_picks_gas_when_oil_scarce() {
        // Make oil scarce (expensive) so gas wins.
        let player = make_player(
            vec![make_plant(PlantKind::GasOrOil, 2)],
            PlayerResources::default(),
        );
        let market = ResourceMarket {
            coal: 24,
            oil: 3,  // slots 0-2, price $8/unit
            gas: 24, // slots 0-23, price $1/unit at slot 23
            uranium: 2,
        };
        let result = compute_set_cart(&player, &market, 1);
        assert_eq!(result, vec![(Resource::Gas, 2)]);
    }

    #[test]
    fn hybrid_plant_falls_back_to_gas_when_oil_depleted() {
        let player = make_player(
            vec![make_plant(PlantKind::GasOrOil, 2)],
            PlayerResources::default(),
        );
        let market = ResourceMarket {
            coal: 24,
            oil: 0, // exhausted
            gas: 6,
            uranium: 2,
        };
        let result = compute_set_cart(&player, &market, 1);
        assert_eq!(result, vec![(Resource::Gas, 2)]);
    }

    #[test]
    fn dedicated_plants_satisfied_first_then_hybrid_buys_oil() {
        // Coal plant (cost 2) + GasOrOil hybrid (cost 1). Player has 3 coal.
        // Coal-only need = 2, covered by stock. Hybrid needs 1 gas-or-oil;
        // with initial market oil is cheaper, so we buy 1 oil.
        let resources = PlayerResources {
            coal: 3,
            ..Default::default()
        };
        let player = make_player(
            vec![
                make_plant(PlantKind::Coal, 2),
                make_plant(PlantKind::GasOrOil, 1),
            ],
            resources,
        );
        let market = ResourceMarket::initial();
        let result = compute_set_cart(&player, &market, 1);
        assert_eq!(result, vec![(Resource::Oil, 1)]);
    }
}
