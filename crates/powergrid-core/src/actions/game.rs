use crate::types::{CityId, PlayerColor, Resource};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Actions that a client can send to the server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Join the game lobby.
    JoinGame { name: String, color: PlayerColor },
    /// Host starts the game (transitions Lobby → PlayerOrder → Auction).
    StartGame,
    /// During auction: select a plant to put up for bid.
    SelectPlant { plant_number: u8 },
    /// During auction: place or raise a bid on the current plant.
    PlaceBid { amount: u32 },
    /// During auction: pass on the current bid or skip selecting a plant.
    PassAuction,
    /// During buy resources: purchase resources from the market.
    BuyResources { resource: Resource, amount: u8 },
    /// During buy resources: purchase a batch of resources atomically and end turn.
    /// An empty purchases list is equivalent to DoneBuying (skip buying).
    BuyResourceBatch { purchases: Vec<(Resource, u8)> },
    /// During buy resources: done buying (pass).
    DoneBuying,
    /// During build cities: build in a city.
    BuildCity { city_id: CityId },
    /// During build cities: build in multiple cities (in order) and end turn atomically.
    BuildCities { city_ids: Vec<CityId> },
    /// During build cities: done building (pass).
    DoneBuilding,
    /// During bureaucracy: declare which plants to fire.
    PowerCities {
        /// Numbers of plants the player is firing this round.
        plant_numbers: Vec<u8>,
    },
    /// During discard phase: choose which existing plant to discard after winning a 4th.
    DiscardPlant { plant_number: u8 },
    /// During resource-discard phase: choose how many gas and oil to drop to resolve
    /// hybrid shared-slot overflow.  `gas + oil` must equal `Phase::DiscardResource::drop_total`.
    DiscardResource { gas: u8, oil: u8 },
    /// During power-cities fuel phase: choose how to split a hybrid plant's fuel cost
    /// between gas and oil.  `gas + oil` must equal `Phase::PowerCitiesFuel::hybrid_cost`.
    PowerCitiesFuel { gas: u8, oil: u8 },
}

/// Errors returned when an action is invalid.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ActionError {
    #[error("game is full")]
    GameFull,
    #[error("name must not be empty")]
    NameEmpty,
    #[error("name exceeds maximum length")]
    NameTooLong,
    #[error("name already taken")]
    NameTaken,
    #[error("color already taken")]
    ColorTaken,
    #[error("only the host can start the game")]
    NotHost,
    #[error("need at least 2 players to start")]
    NotEnoughPlayers,
    #[error("action not allowed in current phase")]
    WrongPhase,
    #[error("it is not your turn")]
    NotYourTurn,
    #[error("plant {0} is not in the market")]
    PlantNotInMarket(u8),
    #[error("bid of {0} is too low; minimum is {1}")]
    BidTooLow(u32, u32),
    #[error("you cannot afford that")]
    CannotAfford,
    #[error("resource not available in that quantity")]
    ResourceUnavailable,
    #[error("you do not have capacity for that many resources")]
    OverCapacity,
    #[error("city {0} does not exist")]
    CityNotFound(String),
    #[error("city {0} is already full")]
    CityFull(String),
    #[error("you already have a city there")]
    AlreadyBuiltThere,
    #[error("you cannot afford to build there")]
    CannotAffordCity,
    #[error("build list must not be empty; use DoneBuilding to skip")]
    EmptyBuildList,
    #[error("duplicate city in build list")]
    DuplicateCityInBuild,
    #[error("city {0} is in an inactive region")]
    CityRegionInactive(String),
    #[error("you do not own plant {0}")]
    PlantNotOwned(u8),
    #[error("unknown player")]
    UnknownPlayer,
    #[error("you must buy a power plant in the first round")]
    MustBuyPlantInRoundOne,
    #[error("cannot discard the plant you just acquired")]
    CannotDiscardNewPlant,
    #[error("coal + oil must equal the required drop total, and neither may exceed what you hold")]
    InvalidDiscardSplit,
    #[error("coal + oil must equal the hybrid fuel cost, and neither may exceed what you hold after pure-fuel plants are paid")]
    InvalidFuelSplit,
    #[error("selected plants cannot all be fired with available resources")]
    InfeasiblePlantSelection,
}
