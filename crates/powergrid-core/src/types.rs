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
    Gas,
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

/// Bot difficulty level, carried in `LobbyAction::AddBot` and `Session::add_bot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BotDifficulty {
    Easy,
    #[default]
    Normal,
    Hard,
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
    GasOrOil,
    Gas,
    Uranium,
    Wind, // no resource cost
}

impl PlantKind {
    pub fn resources(&self) -> Vec<Resource> {
        match self {
            PlantKind::Coal => vec![Resource::Coal],
            PlantKind::Oil => vec![Resource::Oil],
            PlantKind::GasOrOil => vec![Resource::Gas, Resource::Oil],
            PlantKind::Gas => vec![Resource::Gas],
            PlantKind::Uranium => vec![Resource::Uranium],
            PlantKind::Wind => vec![],
        }
    }

    pub fn needs_resources(&self) -> bool {
        !matches!(self, PlantKind::Wind)
    }
}

/// The resource market tracks available supply and current prices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMarket {
    pub coal: u8,
    pub oil: u8,
    pub gas: u8,
    pub uranium: u8,
}

impl ResourceMarket {
    /// Standard starting supply for 2–6 players (we use max supply for simplicity).
    pub fn initial() -> Self {
        Self {
            coal: 23,
            gas: 18,
            oil: 14,
            uranium: 2,
        }
    }

    pub fn available(&self, resource: Resource) -> u8 {
        match resource {
            Resource::Coal => self.coal,
            Resource::Oil => self.oil,
            Resource::Gas => self.gas,
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
            Resource::Gas => self.gas -= amount,
            Resource::Uranium => self.uranium -= amount,
        }
        true
    }

    pub fn replenish(&mut self, resource: Resource, amount: u8) {
        let max = price_table(resource).len() as u8;
        let field = match resource {
            Resource::Coal => &mut self.coal,
            Resource::Oil => &mut self.oil,
            Resource::Gas => &mut self.gas,
            Resource::Uranium => &mut self.uranium,
        };
        *field = (*field + amount).min(max);
    }

    /// Cost to buy `amount` units of `resource`.
    /// Prices are on a sliding scale — cheaper when plentiful, pricier when scarce.
    pub fn price(&self, resource: Resource, amount: u8) -> Option<u32> {
        if amount == 0 {
            return Some(0);
        }
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
pub fn price_table(resource: Resource) -> &'static [u8] {
    match resource {
        Resource::Coal => &[
            9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1,
        ],
        Resource::Gas => &[
            8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1,
        ],
        Resource::Oil => &[9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
        Resource::Uranium => &[9, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2, 1],
    }
}

/// A player's stored resources (on their power plants).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerResources {
    pub coal: u8,
    pub oil: u8,
    pub gas: u8,
    pub uranium: u8,
}

impl PlayerResources {
    pub fn get(&self, resource: Resource) -> u8 {
        match resource {
            Resource::Coal => self.coal,
            Resource::Oil => self.oil,
            Resource::Gas => self.gas,
            Resource::Uranium => self.uranium,
        }
    }

    pub fn add(&mut self, resource: Resource, amount: u8) {
        match resource {
            Resource::Coal => self.coal += amount,
            Resource::Oil => self.oil += amount,
            Resource::Gas => self.gas += amount,
            Resource::Uranium => self.uranium += amount,
        }
    }

    pub fn remove(&mut self, resource: Resource, amount: u8) -> bool {
        let field = match resource {
            Resource::Coal => &mut self.coal,
            Resource::Oil => &mut self.oil,
            Resource::Gas => &mut self.gas,
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
    pub plants: Vec<PowerPlant>,
    pub resources: PlayerResources,
    pub last_cities_powered: u8,
}

impl Player {
    pub fn new(name: String, color: PlayerColor) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            color,
            money: 50,
            plants: Vec::new(),
            resources: PlayerResources::default(),
            last_cities_powered: 0,
        }
    }

    /// Max resources this player can store across all their plants.
    pub fn resource_capacity(&self, resource: Resource) -> u8 {
        self.plants
            .iter()
            .map(|p| {
                let accepts = p.kind.resources().contains(&resource)
                    || (resource == Resource::Gas && p.kind == PlantKind::GasOrOil)
                    || (resource == Resource::Oil && p.kind == PlantKind::GasOrOil);
                if accepts {
                    p.cost * 2
                } else {
                    0
                }
            })
            .sum()
    }

    /// Whether the player can store `amount` more of `resource`, respecting
    /// the shared-slot constraint on GasOrOil hybrid plants.
    pub fn can_add_resource(&self, resource: Resource, amount: u8) -> bool {
        match resource {
            Resource::Gas | Resource::Oil => {
                let gas_only: u8 = self
                    .plants
                    .iter()
                    .filter(|p| p.kind == PlantKind::Gas)
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
                    .filter(|p| p.kind == PlantKind::GasOrOil)
                    .map(|p| p.cost * 2)
                    .sum();

                let (new_gas, new_oil) = if resource == Resource::Gas {
                    (self.resources.gas + amount, self.resources.oil)
                } else {
                    (self.resources.gas, self.resources.oil + amount)
                };

                // Each resource must fit in its dedicated + hybrid slots.
                if new_gas > gas_only + hybrid {
                    return false;
                }
                if new_oil > oil_only + hybrid {
                    return false;
                }
                // Gas and oil together cannot exceed the shared hybrid slots.
                new_gas.saturating_sub(gas_only) + new_oil.saturating_sub(oil_only) <= hybrid
            }
            _ => self.resources.get(resource) + amount <= self.resource_capacity(resource),
        }
    }

    /// How many gas+oil units violate the hybrid shared-slot constraint.
    ///
    /// Returns 0 when the current gas/oil storage is within capacity. A nonzero
    /// value means the player must drop that many total units of gas or oil (or a
    /// combination), and the split is only unambiguous if they hold zero of one of
    /// the two resources.
    pub fn shared_slot_overflow(&self) -> u8 {
        let gas_only: u8 = self
            .plants
            .iter()
            .filter(|p| p.kind == PlantKind::Gas)
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
            .filter(|p| p.kind == PlantKind::GasOrOil)
            .map(|p| p.cost * 2)
            .sum();
        let gas = self.resources.gas;
        let oil = self.resources.oil;
        let gas_into_hybrid = gas.saturating_sub(gas_only);
        let oil_into_hybrid = oil.saturating_sub(oil_only);
        (gas_into_hybrid + oil_into_hybrid).saturating_sub(hybrid)
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

    /// Optimal feasible subset of plants to fire in the Bureaucracy phase.
    /// Returns (chosen plant numbers, cities powered capped at cities owned, remaining resources).
    /// Used by the bot and client to compute the recommended default selection.
    pub fn optimal_firing_subset(&self, city_count: u8) -> (Vec<u8>, u8, PlayerResources) {
        let cities_owned = city_count;
        let n = self.plants.len();
        let mut best_powered = 0u8;
        let mut best_res = self.resources.clone();
        let mut best_subset: Vec<u8> = Vec::new();

        for mask in 1u32..(1u32 << n) {
            let subset: Vec<&PowerPlant> = self
                .plants
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, p)| p)
                .collect();

            if let Some((powered, res)) = check_plant_feasibility(&subset, &self.resources) {
                let capped = powered.min(cities_owned);
                let leftover =
                    res.coal as u16 + res.oil as u16 + res.gas as u16 + res.uranium as u16;
                let best_leftover = best_res.coal as u16
                    + best_res.oil as u16
                    + best_res.gas as u16
                    + best_res.uranium as u16;
                if capped > best_powered || (capped == best_powered && leftover > best_leftover) {
                    best_powered = capped;
                    best_res = res;
                    best_subset = self
                        .plants
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, p)| p.number)
                        .collect();
                }
            }
        }

        (best_subset, best_powered, best_res)
    }
}

/// Check whether a set of plants can fire with the given resources using a
/// two-pass allocation: pure-fuel plants are satisfied first, then GasOrOil
/// hybrids consume whatever gas+oil remains.  Hybrids prefer oil when
/// possible to conserve gas for future pure-Gas plants.
///
/// Returns `Some((cities_powered, remaining_resources))` if feasible, or
/// `None` if the resources are insufficient.
pub fn check_plant_feasibility(
    plants: &[&PowerPlant],
    resources: &PlayerResources,
) -> Option<(u8, PlayerResources)> {
    let mut coal = resources.coal;
    let mut oil = resources.oil;
    let mut gas = resources.gas;
    let mut uranium = resources.uranium;
    let mut powered = 0u8;
    let mut pure_coal_cost: u8 = 0;
    let mut pure_oil_cost: u8 = 0;
    let mut pure_gas_cost: u8 = 0;
    let mut hybrid_cost: u8 = 0;

    for plant in plants {
        match plant.kind {
            PlantKind::Coal => pure_coal_cost += plant.cost,
            PlantKind::Oil => pure_oil_cost += plant.cost,
            PlantKind::Gas => pure_gas_cost += plant.cost,
            PlantKind::GasOrOil => hybrid_cost += plant.cost,
            PlantKind::Uranium => {
                if uranium < plant.cost {
                    return None;
                }
                uranium -= plant.cost;
            }
            PlantKind::Wind => {}
        }
        powered += plant.cities;
    }

    // Satisfy pure plants first.
    if pure_coal_cost > coal || pure_oil_cost > oil || pure_gas_cost > gas {
        return None;
    }
    coal -= pure_coal_cost;
    oil -= pure_oil_cost;
    gas -= pure_gas_cost;

    // Satisfy hybrids with remaining gas+oil pool; prefer oil to preserve gas.
    if hybrid_cost > gas + oil {
        return None;
    }
    let from_oil = hybrid_cost.min(oil);
    oil -= from_oil;
    gas -= hybrid_cost - from_oil;

    Some((
        powered,
        PlayerResources {
            coal,
            oil,
            gas,
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
    /// Waiting for a player to choose which gas/oil to discard when hybrid-slot overflow is
    /// ambiguous (i.e. neither resource alone exceeds its per-resource cap, but they jointly
    /// exceed the available shared slots on GasOrOil plants).
    DiscardResource {
        /// The player who must choose.
        player: PlayerId,
        /// Total units of gas+oil the player must drop.
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
    /// Waiting for a player to choose the gas/oil split when firing hybrid (GasOrOil)
    /// plants — only entered when the split is genuinely ambiguous (the player has both
    /// gas and oil available beyond what their pure-fuel plants need, with slack to
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
    /// The plant number that holds the discount token (min bid = 1 Elektro), if any.
    /// Set at the start of Phase 2; cleared at Phase 2 exit or when a smaller plant
    /// is drawn into `actual` during Phase 2.
    #[serde(default)]
    pub discount_token: Option<u8>,
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

    /// Remove all plants whose number is ≤ `max_cities` from the actual market,
    /// the draw deck, and the below-Step-3 pile. Used after city building so
    /// that no obsolete plant remains anywhere a future draw could surface it.
    pub fn remove_obsolete(&mut self, max_cities: usize) {
        // Purge unseen piles first so the subsequent refill() draws from a clean deck.
        self.deck.retain(|p| (p.number as usize) > max_cities);
        if let Some(below) = self.below_step3.as_mut() {
            below.retain(|p| (p.number as usize) > max_cities);
        }
        // Then drop obsolete plants from the visible market, refilling from the
        // (now safe) deck after each removal.
        while let Some(lowest) = self.actual.first() {
            if (lowest.number as usize) <= max_cities {
                self.actual.remove(0);
                self.refill();
            } else {
                break;
            }
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

    /// Set up the deck for game start (Deluxe rules). Called once after `build_plant_deck`.
    ///
    /// 1. Partition all plants into low (≤15) and high (>15) pools.
    /// 2. Draw 8 random low cards for the initial market (4 actual + 4 future, sorted).
    /// 3. Discard a number of remaining low/high cards based on player count.
    /// 4. Shuffle the rest together; Step 3 triggers when this deck empties.
    pub fn setup_deck(&mut self, rng: &mut impl rand::Rng, player_count: usize) {
        // 1. Partition deck into low (≤15) and high (>15).
        let mut low: Vec<PowerPlant> = Vec::new();
        let mut high: Vec<PowerPlant> = Vec::new();
        for p in self.deck.drain(..) {
            if p.number <= 15 {
                low.push(p);
            } else {
                high.push(p);
            }
        }

        // 2. Shuffle low, draw 8 for initial market.
        low.shuffle(rng);
        let mut market_sorted: Vec<PowerPlant> = low.drain(..8.min(low.len())).collect();
        market_sorted.sort_by_key(|p| p.number);
        self.actual = market_sorted.iter().take(4).cloned().collect();
        self.future = market_sorted.iter().skip(4).cloned().collect();

        // 3. Remove random plants based on player count.
        let (low_remove, high_remove) = match player_count {
            2 | 3 => (2usize, 6usize),
            4 => (1, 3),
            _ => (0, 0), // 5–6 players: remove nothing
        };
        low.shuffle(rng);
        low.truncate(low.len().saturating_sub(low_remove));
        high.shuffle(rng);
        high.truncate(high.len().saturating_sub(high_remove));

        // 4. Combine, shuffle, assign to deck. The Step 3 card triggers when this
        // deck is exhausted (below_step3 = Some signals the card is "in play").
        let mut combined: Vec<PowerPlant> = low.into_iter().chain(high).collect();
        combined.shuffle(rng);
        self.deck = combined;
        self.below_step3 = Some(Vec::new());
    }
}
