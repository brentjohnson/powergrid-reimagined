use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type PlayerId = Uuid;
pub type CityId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Coal,
    Oil,
    Garbage,
    Uranium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayerColor {
    Red,
    Blue,
    Green,
    Yellow,
    Purple,
    White,
}

/// A power plant card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerPlant {
    /// Minimum bid / market cost.
    pub number: u8,
    pub kind: PlantKind,
    /// Resources consumed per firing.
    pub cost: u8,
    /// Cities powered per firing.
    pub cities: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlantKind {
    Coal,
    Oil,
    CoalOrOil,
    Garbage,
    Uranium,
    Wind,   // no resource cost
    Fusion, // no resource cost (Step 3 era)
}

impl PlantKind {
    pub fn resources(&self) -> Vec<Resource> {
        match self {
            PlantKind::Coal => vec![Resource::Coal],
            PlantKind::Oil => vec![Resource::Oil],
            PlantKind::CoalOrOil => vec![Resource::Coal, Resource::Oil],
            PlantKind::Garbage => vec![Resource::Garbage],
            PlantKind::Uranium => vec![Resource::Uranium],
            PlantKind::Wind | PlantKind::Fusion => vec![],
        }
    }

    pub fn needs_resources(&self) -> bool {
        !matches!(self, PlantKind::Wind | PlantKind::Fusion)
    }
}

/// The resource market tracks available supply and current prices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMarket {
    pub coal: u8,
    pub oil: u8,
    pub garbage: u8,
    pub uranium: u8,
}

impl ResourceMarket {
    /// Standard starting supply for 2–6 players (we use max supply for simplicity).
    pub fn initial() -> Self {
        Self {
            coal: 24,
            oil: 18,
            garbage: 6,
            uranium: 2,
        }
    }

    pub fn available(&self, resource: Resource) -> u8 {
        match resource {
            Resource::Coal => self.coal,
            Resource::Oil => self.oil,
            Resource::Garbage => self.garbage,
            Resource::Uranium => self.uranium,
        }
    }

    pub fn take(&mut self, resource: Resource, amount: u8) -> bool {
        let avail = self.available(resource);
        if avail < amount {
            return false;
        }
        match resource {
            Resource::Coal => self.coal -= amount,
            Resource::Oil => self.oil -= amount,
            Resource::Garbage => self.garbage -= amount,
            Resource::Uranium => self.uranium -= amount,
        }
        true
    }

    pub fn replenish(&mut self, resource: Resource, amount: u8) {
        let max = match resource {
            Resource::Coal => 24,
            Resource::Oil => 24,
            Resource::Garbage => 24,
            Resource::Uranium => 12,
        };
        let field = match resource {
            Resource::Coal => &mut self.coal,
            Resource::Oil => &mut self.oil,
            Resource::Garbage => &mut self.garbage,
            Resource::Uranium => &mut self.uranium,
        };
        *field = (*field + amount).min(max);
    }

    /// Cost to buy `amount` units of `resource`.
    /// Prices are on a sliding scale — cheaper when plentiful, pricier when scarce.
    pub fn price(&self, resource: Resource, amount: u8) -> Option<u32> {
        let current = self.available(resource);
        if current < amount {
            return None;
        }
        let table = price_table(resource);
        let mut total = 0u32;
        // Slot index 0 = scarcest (most expensive). Resources occupy slots 0..(current-1).
        // Buy from cheapest available (index current-1) upward toward the expensive end.
        let last_occupied = (current - 1) as usize;
        for i in 0..amount as usize {
            let slot = last_occupied - i;
            total += table[slot] as u32;
        }
        Some(total)
    }

    /// Cost to buy all items in `purchases`, simulating sequential market depletion.
    /// Returns `None` if any resource is unavailable in the required quantity.
    pub fn batch_price(&self, purchases: &[(Resource, u8)]) -> Option<u32> {
        let mut scratch = self.clone();
        let mut total = 0u32;
        for &(resource, amount) in purchases {
            let cost = scratch.price(resource, amount)?;
            total += cost;
            scratch.take(resource, amount);
        }
        Some(total)
    }
}

/// Returns the price per unit at each market slot (index 0 = most expensive / scarce).
fn price_table(resource: Resource) -> &'static [u8] {
    match resource {
        Resource::Coal => &[
            8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1,
        ],
        Resource::Oil => &[
            8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1,
        ],
        Resource::Garbage => &[
            8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1,
        ],
        Resource::Uranium => &[16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1],
    }
}

/// A player's stored resources (on their power plants).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerResources {
    pub coal: u8,
    pub oil: u8,
    pub garbage: u8,
    pub uranium: u8,
}

impl PlayerResources {
    pub fn get(&self, resource: Resource) -> u8 {
        match resource {
            Resource::Coal => self.coal,
            Resource::Oil => self.oil,
            Resource::Garbage => self.garbage,
            Resource::Uranium => self.uranium,
        }
    }

    pub fn add(&mut self, resource: Resource, amount: u8) {
        match resource {
            Resource::Coal => self.coal += amount,
            Resource::Oil => self.oil += amount,
            Resource::Garbage => self.garbage += amount,
            Resource::Uranium => self.uranium += amount,
        }
    }

    pub fn remove(&mut self, resource: Resource, amount: u8) -> bool {
        let field = match resource {
            Resource::Coal => &mut self.coal,
            Resource::Oil => &mut self.oil,
            Resource::Garbage => &mut self.garbage,
            Resource::Uranium => &mut self.uranium,
        };
        if *field < amount {
            return false;
        }
        *field -= amount;
        true
    }
}

/// A player in the game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: PlayerId,
    pub name: String,
    pub color: PlayerColor,
    pub money: u32,
    pub cities: Vec<CityId>,
    pub plants: Vec<PowerPlant>,
    pub resources: PlayerResources,
    pub passed_auction: bool,
    pub last_cities_powered: u8,
}

impl Player {
    pub fn new(name: String, color: PlayerColor) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            color,
            money: 50,
            cities: Vec::new(),
            plants: Vec::new(),
            resources: PlayerResources::default(),
            passed_auction: false,
            last_cities_powered: 0,
        }
    }

    pub fn city_count(&self) -> usize {
        self.cities.len()
    }

    /// Max resources this player can store across all their plants.
    pub fn resource_capacity(&self, resource: Resource) -> u8 {
        self.plants
            .iter()
            .map(|p| {
                let accepts = p.kind.resources().contains(&resource)
                    || (resource == Resource::Coal && p.kind == PlantKind::CoalOrOil)
                    || (resource == Resource::Oil && p.kind == PlantKind::CoalOrOil);
                if accepts {
                    p.cost * 2
                } else {
                    0
                }
            })
            .sum()
    }

    /// Whether the player can store `amount` more of `resource`, respecting
    /// the shared-slot constraint on CoalOrOil hybrid plants.
    pub fn can_add_resource(&self, resource: Resource, amount: u8) -> bool {
        match resource {
            Resource::Coal | Resource::Oil => {
                let coal_only: u8 = self
                    .plants
                    .iter()
                    .filter(|p| p.kind == PlantKind::Coal)
                    .map(|p| p.cost * 2)
                    .sum();
                let oil_only: u8 = self
                    .plants
                    .iter()
                    .filter(|p| p.kind == PlantKind::Oil)
                    .map(|p| p.cost * 2)
                    .sum();
                let hybrid: u8 = self
                    .plants
                    .iter()
                    .filter(|p| p.kind == PlantKind::CoalOrOil)
                    .map(|p| p.cost * 2)
                    .sum();

                let (new_coal, new_oil) = if resource == Resource::Coal {
                    (self.resources.coal + amount, self.resources.oil)
                } else {
                    (self.resources.coal, self.resources.oil + amount)
                };

                // Each resource must fit in its dedicated + hybrid slots.
                if new_coal > coal_only + hybrid {
                    return false;
                }
                if new_oil > oil_only + hybrid {
                    return false;
                }
                // Coal and oil together cannot exceed the shared hybrid slots.
                new_coal.saturating_sub(coal_only) + new_oil.saturating_sub(oil_only) <= hybrid
            }
            _ => self.resources.get(resource) + amount <= self.resource_capacity(resource),
        }
    }

    /// How many coal+oil units violate the hybrid shared-slot constraint.
    ///
    /// Returns 0 when the current coal/oil storage is within capacity.  A nonzero
    /// value means the player must drop that many total units of coal or oil (or a
    /// combination), and the split is only unambiguous if they hold zero of one of
    /// the two resources.
    pub fn shared_slot_overflow(&self) -> u8 {
        let coal_only: u8 = self
            .plants
            .iter()
            .filter(|p| p.kind == PlantKind::Coal)
            .map(|p| p.cost * 2)
            .sum();
        let oil_only: u8 = self
            .plants
            .iter()
            .filter(|p| p.kind == PlantKind::Oil)
            .map(|p| p.cost * 2)
            .sum();
        let hybrid: u8 = self
            .plants
            .iter()
            .filter(|p| p.kind == PlantKind::CoalOrOil)
            .map(|p| p.cost * 2)
            .sum();
        let coal = self.resources.coal;
        let oil = self.resources.oil;
        let coal_into_hybrid = coal.saturating_sub(coal_only);
        let oil_into_hybrid = oil.saturating_sub(oil_only);
        (coal_into_hybrid + oil_into_hybrid).saturating_sub(hybrid)
    }

    /// Number of cities this player can power given their plants and stored resources.
    pub fn cities_powerable(&self) -> u8 {
        let n = self.plants.len();
        let mut best = 0u8;
        for mask in 0u8..(1u8 << n) {
            let subset: Vec<&PowerPlant> = self
                .plants
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, p)| p)
                .collect();
            if let Some((powered, _)) = check_plant_feasibility(&subset, &self.resources) {
                if powered > best {
                    best = powered;
                }
            }
        }
        best
    }
}

/// Check whether a set of plants can fire with the given resources using a
/// two-pass allocation: pure-fuel plants are satisfied first, then CoalOrOil
/// hybrids consume whatever coal+oil remains.  Hybrids prefer oil when
/// possible to conserve coal for future pure-Coal plants.
///
/// Returns `Some((cities_powered, remaining_resources))` if feasible, or
/// `None` if the resources are insufficient.
pub fn check_plant_feasibility(
    plants: &[&PowerPlant],
    resources: &PlayerResources,
) -> Option<(u8, PlayerResources)> {
    let mut coal = resources.coal;
    let mut oil = resources.oil;
    let mut garbage = resources.garbage;
    let mut uranium = resources.uranium;
    let mut powered = 0u8;
    let mut pure_coal_cost: u8 = 0;
    let mut pure_oil_cost: u8 = 0;
    let mut hybrid_cost: u8 = 0;

    for plant in plants {
        match plant.kind {
            PlantKind::Coal => pure_coal_cost += plant.cost,
            PlantKind::Oil => pure_oil_cost += plant.cost,
            PlantKind::CoalOrOil => hybrid_cost += plant.cost,
            PlantKind::Garbage => {
                if garbage < plant.cost {
                    return None;
                }
                garbage -= plant.cost;
            }
            PlantKind::Uranium => {
                if uranium < plant.cost {
                    return None;
                }
                uranium -= plant.cost;
            }
            PlantKind::Wind | PlantKind::Fusion => {}
        }
        powered += plant.cities;
    }

    // Satisfy pure plants first.
    if pure_coal_cost > coal || pure_oil_cost > oil {
        return None;
    }
    coal -= pure_coal_cost;
    oil -= pure_oil_cost;

    // Satisfy hybrids with remaining coal+oil pool; prefer oil to preserve coal.
    if hybrid_cost > coal + oil {
        return None;
    }
    let from_oil = hybrid_cost.min(oil);
    oil -= from_oil;
    coal -= hybrid_cost - from_oil;

    Some((
        powered,
        PlayerResources {
            coal,
            oil,
            garbage,
            uranium,
        },
    ))
}

/// Income table: indexed by number of cities powered.
pub fn income_for(cities_powered: u8) -> u32 {
    match cities_powered {
        0 => 10,
        1 => 22,
        2 => 33,
        3 => 44,
        4 => 54,
        5 => 64,
        6 => 73,
        7 => 82,
        8 => 90,
        9 => 98,
        10 => 105,
        11 => 112,
        12 => 118,
        13 => 124,
        14 => 129,
        15 => 134,
        16 => 138,
        17 => 142,
        _ => 150,
    }
}

/// Connection cost to build in a city that already has n other players' connections.
pub fn connection_cost(existing_connections: usize) -> u32 {
    match existing_connections {
        0 => 10,
        1 => 15,
        2 => 20,
        _ => 20, // max 3 players per city in base game
    }
}

/// The phase the game is currently in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Players are joining; host hasn't started yet.
    Lobby,
    /// Determine player order for this round.
    PlayerOrder,
    /// Auction phase: players bid on power plants in reverse order.
    Auction {
        /// Index into `game.player_order` — whose turn to select a plant.
        current_bidder_idx: usize,
        /// Active bid on a specific plant, if any.
        active_bid: Option<ActiveBid>,
        /// Players who have already bought a plant this round (by player id index).
        bought: Vec<PlayerId>,
        /// Players who have passed this auction round.
        passed: Vec<PlayerId>,
    },
    /// Waiting for a player to choose which of their existing plants to discard after winning a 4th.
    DiscardPlant {
        /// The player who must discard.
        player: PlayerId,
        /// The newly-won plant (not yet in player's hand).
        new_plant: PowerPlant,
        /// Auction bought list (to resume after discard).
        bought: Vec<PlayerId>,
        /// Auction passed list (to resume after discard).
        passed: Vec<PlayerId>,
    },
    /// Waiting for a player to choose which coal/oil to discard when hybrid-slot overflow is
    /// ambiguous (i.e. neither resource alone exceeds its per-resource cap, but they jointly
    /// exceed the available shared slots on CoalOrOil plants).
    DiscardResource {
        /// The player who must choose.
        player: PlayerId,
        /// Total units of coal+oil the player must drop.
        drop_total: u8,
        /// Auction bought list (to resume after discard).
        bought: Vec<PlayerId>,
        /// Auction passed list (to resume after discard).
        passed: Vec<PlayerId>,
    },
    /// Buy resources phase: players buy in reverse order.
    BuyResources {
        remaining: Vec<PlayerId>, // players yet to act, in order
    },
    /// Build cities phase: players build in reverse order.
    BuildCities { remaining: Vec<PlayerId> },
    /// Bureaucracy: power cities, collect income, restock market.
    Bureaucracy { remaining: Vec<PlayerId> },
    /// Waiting for a player to choose the coal/oil split when firing hybrid (CoalOrOil)
    /// plants — only entered when the split is genuinely ambiguous (the player has both
    /// coal and oil available beyond what their pure-fuel plants need, with slack to
    /// spend either).
    PowerCitiesFuel {
        /// The player who must choose.
        player: PlayerId,
        /// The chosen subset of plants to fire (the optimal one selected server-side).
        plant_numbers: Vec<u8>,
        /// Total fuel needed by the hybrid plants in `plant_numbers`.
        hybrid_cost: u8,
        /// Bureaucracy `remaining` list to restore once the split is applied.
        remaining: Vec<PlayerId>,
    },
    /// Game over.
    GameOver { winner: PlayerId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveBid {
    pub plant_number: u8,
    pub highest_bidder: PlayerId,
    pub amount: u32,
    /// Players still in the bidding (haven't passed on this plant).
    pub remaining_bidders: Vec<PlayerId>,
}

/// The power plant market has an "actual" (lower 4) and "future" (upper 4) section.
/// In Step 3, `future` is always empty and `actual` holds all 6 available plants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantMarket {
    /// The plants available for auction (sorted by number).
    /// Steps 1/2: lower 4. Step 3: all 6.
    pub actual: Vec<PowerPlant>,
    /// The future market plants (sorted by number). Always empty in Step 3.
    pub future: Vec<PowerPlant>,
    /// Draw deck (face down). Index 0 is the bottom; `pop()` draws from the top.
    pub deck: Vec<PowerPlant>,
    /// Plant 13, held aside until placed on top of the deck at game start.
    #[serde(default)]
    pub plant_13: Option<PowerPlant>,
    /// Cards cycled below the Step 3 card during Steps 1/2. `Some` means the Step 3
    /// card is in play (between `deck` and this pile). When the main deck is exhausted
    /// and a draw is attempted, the Step 3 card is "drawn" — these cards are then
    /// shuffled to form the Step 3 draw deck. `None` before setup or after Step 3 triggers.
    #[serde(default)]
    pub below_step3: Option<Vec<PowerPlant>>,
    /// Set to true by `refill()` when the Step 3 card is drawn. Cleared by rules.rs
    /// after the Step 3 transition is applied.
    #[serde(default)]
    pub step3_triggered: bool,
    /// True once Step 3 is active. Changes market fill target to 6 and removes
    /// the actual/future split.
    #[serde(default)]
    pub in_step3: bool,
}

impl PlantMarket {
    /// Remove a plant from the actual market by its number. Returns it if found.
    pub fn take_from_actual(&mut self, number: u8) -> Option<PowerPlant> {
        if let Some(pos) = self.actual.iter().position(|p| p.number == number) {
            let plant = self.actual.remove(pos);
            self.refill();
            Some(plant)
        } else {
            None
        }
    }

    /// Draw from deck into market until full.
    /// Steps 1/2: fill to 8, split 4 actual / 4 future.
    /// Step 3: fill to 6, all in actual, future empty.
    /// Sets `step3_triggered` if the Step 3 card is drawn (i.e. a draw is attempted
    /// while `deck` is empty and `below_step3` is `Some`). Caller must handle the
    /// transition.
    pub fn refill(&mut self) {
        let target = if self.in_step3 { 6 } else { 8 };
        let mut all: Vec<PowerPlant> = self.actual.drain(..).chain(self.future.drain(..)).collect();
        while all.len() < target {
            if self.deck.is_empty() {
                // Deck exhausted. If the Step 3 card is in play, we just "drew" it.
                if self.below_step3.is_some() {
                    self.step3_triggered = true;
                }
                break;
            }
            all.push(self.deck.pop().unwrap());
        }
        all.sort_by_key(|p| p.number);
        if self.in_step3 {
            self.actual = all;
            self.future = Vec::new();
        } else {
            self.actual = all.iter().take(4).cloned().collect();
            self.future = all.iter().skip(4).cloned().collect();
        }
    }

    /// Remove the lowest-numbered plant from the actual market (used at end of round).
    pub fn remove_lowest(&mut self) {
        if !self.actual.is_empty() {
            self.actual.remove(0);
            self.refill();
        }
    }

    /// Remove all plants from the actual market whose number is ≤ `max_cities`.
    /// Used after city building to clear obsolete plants.
    pub fn remove_obsolete(&mut self, max_cities: usize) {
        loop {
            if let Some(lowest) = self.actual.first() {
                if (lowest.number as usize) <= max_cities {
                    self.actual.remove(0);
                    self.refill();
                    continue;
                }
            }
            break;
        }
    }

    /// Move the highest-numbered plant from the future market to below the Step 3 card,
    /// then refill. Used at end of Bureaucracy in Steps 1 and 2.
    pub fn cycle_highest_to_bottom(&mut self) {
        if let Some(plant) = self.future.pop() {
            if let Some(below) = self.below_step3.as_mut() {
                below.push(plant);
            } else {
                // Step 3 already triggered or not in play; put in main deck.
                self.deck.insert(0, plant);
            }
            self.refill();
        }
    }

    /// Remove the highest-numbered plant from the market entirely.
    /// Used at end of Bureaucracy in Step 3 (replaces cycling to deck bottom).
    pub fn remove_highest_from_game(&mut self) {
        if !self.actual.is_empty() {
            self.actual.pop(); // actual is sorted ascending, last = highest
            self.refill();
        }
    }

    /// Shuffle the draw deck, remove plants based on player count,
    /// then place plant 13 on top and the Step 3 card on the bottom.
    /// Called once at game start.
    pub fn setup_deck(&mut self, rng: &mut impl rand::Rng, player_count: usize) {
        // 1. Shuffle the deck.
        self.deck.shuffle(rng);

        // 2. Remove random plants (face-down) based on player count.
        let remove_count = match player_count {
            2 => 8,
            3 => 8,
            4 => 4,
            _ => 0, // 5-6 players: remove none
        };
        self.deck
            .truncate(self.deck.len().saturating_sub(remove_count));

        // 3. Place plant 13 on top (end of vec, drawn first via pop()).
        if let Some(plant_13) = self.plant_13.take() {
            self.deck.push(plant_13);
        }

        // 4. Step 3 card sits between the main deck and the below-step3 pile.
        self.below_step3 = Some(Vec::new());
    }
}
