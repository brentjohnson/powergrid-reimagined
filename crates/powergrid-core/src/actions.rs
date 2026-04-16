use crate::types::{CityId, PlayerColor, Resource};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Actions that a client can send to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// During buy resources: done buying (pass).
    DoneBuying,
    /// During build cities: build in a city.
    BuildCity { city_id: CityId },
    /// During build cities: done building (pass).
    DoneBuilding,
    /// During bureaucracy: declare which plants to fire.
    PowerCities {
        /// Numbers of plants the player is firing this round.
        plant_numbers: Vec<u8>,
    },
}

/// Errors returned when an action is invalid.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ActionError {
    #[error("game is full")]
    GameFull,
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
    #[error("you do not own plant {0}")]
    PlantNotOwned(u8),
    #[error("unknown player")]
    UnknownPlayer,
}

/// Messages sent from the server to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Sent immediately on connection so the client knows its own player ID.
    Welcome { your_id: crate::types::PlayerId },
    /// Full game state broadcast after every valid action.
    StateUpdate(Box<crate::state::GameState>),
    /// Sent only to the client whose action was rejected.
    ActionError { message: String },
    /// Informational event (e.g. "Hamburg was built by Red").
    Event { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_error_serde_roundtrip() {
        let msg = ServerMessage::ActionError {
            message: "it is not your turn".to_string(),
        };
        let json = serde_json::to_string(&msg).expect("serialization should succeed");
        let parsed: ServerMessage =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert!(
            matches!(parsed, ServerMessage::ActionError { message } if message == "it is not your turn")
        );
    }
}
