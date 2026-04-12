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
    Black,
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
            Resource::Oil => 18,
            Resource::Garbage => 6,
            Resource::Uranium => 2,
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
        // Resources are bought from the most expensive available slot downward.
        // Slot index 0 = scarcest (most expensive).
        let slots = table.len() as u8;
        // Occupied slots count from the right (cheapest side).
        // We buy from the leftmost occupied slots (most expensive available).
        let first_occupied = slots - current; // index of cheapest available slot
        for i in 0..amount {
            let slot = first_occupied + i;
            if slot as usize >= table.len() {
                return None;
            }
            total += table[slot as usize] as u32;
        }
        Some(total)
    }
}

/// Returns the price per unit at each market slot (index 0 = most expensive / scarce).
fn price_table(resource: Resource) -> &'static [u8] {
    match resource {
        Resource::Coal => &[
            8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        Resource::Oil => &[8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1],
        Resource::Garbage => &[8, 7, 6, 5, 4, 3, 2, 1, 1, 1], // padded to match actual board
        Resource::Uranium => &[16, 14, 12, 10, 8, 6, 4, 2, 1, 1],
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

    /// Number of cities this player can power given their plants and stored resources.
    pub fn cities_powerable(&self) -> u8 {
        let mut coal = self.resources.coal;
        let mut oil = self.resources.oil;
        let mut garbage = self.resources.garbage;
        let mut uranium = self.resources.uranium;
        let mut powered = 0u8;

        for plant in &self.plants {
            let can_fire = match plant.kind {
                PlantKind::Coal => coal >= plant.cost,
                PlantKind::Oil => oil >= plant.cost,
                PlantKind::CoalOrOil => coal + oil >= plant.cost,
                PlantKind::Garbage => garbage >= plant.cost,
                PlantKind::Uranium => uranium >= plant.cost,
                PlantKind::Wind | PlantKind::Fusion => true,
            };
            if can_fire {
                match plant.kind {
                    PlantKind::Coal => coal -= plant.cost,
                    PlantKind::Oil => oil -= plant.cost,
                    PlantKind::CoalOrOil => {
                        let from_coal = plant.cost.min(coal);
                        coal -= from_coal;
                        oil -= plant.cost - from_coal;
                    }
                    PlantKind::Garbage => garbage -= plant.cost,
                    PlantKind::Uranium => uranium -= plant.cost,
                    PlantKind::Wind | PlantKind::Fusion => {}
                }
                powered += plant.cities;
            }
        }
        powered
    }
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
    /// Buy resources phase: players buy in reverse order.
    BuyResources {
        remaining: Vec<PlayerId>, // players yet to act, in order
    },
    /// Build cities phase: players build in reverse order.
    BuildCities { remaining: Vec<PlayerId> },
    /// Bureaucracy: power cities, collect income, restock market.
    Bureaucracy { remaining: Vec<PlayerId> },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantMarket {
    /// The 4 plants available for auction this round (sorted by number).
    pub actual: Vec<PowerPlant>,
    /// The 4 plants in the future market (sorted by number).
    pub future: Vec<PowerPlant>,
    /// Draw deck (face down).
    pub deck: Vec<PowerPlant>,
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

    /// Draw from deck into actual+future until both are full (4 each).
    pub fn refill(&mut self) {
        let mut all: Vec<PowerPlant> = self.actual.drain(..).chain(self.future.drain(..)).collect();
        while all.len() < 8 {
            if let Some(card) = self.deck.pop() {
                all.push(card);
            } else {
                break;
            }
        }
        all.sort_by_key(|p| p.number);
        self.actual = all.iter().take(4).cloned().collect();
        self.future = all.iter().skip(4).cloned().collect();
    }

    /// Remove the lowest-numbered plant from the actual market (used at end of round).
    pub fn remove_lowest(&mut self) {
        if !self.actual.is_empty() {
            self.actual.remove(0);
            self.refill();
        }
    }
}
