use crate::types::{CityId, PlayerColor, PlayerId, Resource};
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
    /// During resource-discard phase: choose how many coal and oil to drop to resolve
    /// hybrid shared-slot overflow.  `coal + oil` must equal `Phase::DiscardResource::drop_total`.
    DiscardResource { coal: u8, oil: u8 },
    /// During power-cities fuel phase: choose how to split a hybrid plant's fuel cost
    /// between coal and oil.  `coal + oil` must equal `Phase::PowerCitiesFuel::hybrid_cost`.
    PowerCitiesFuel { coal: u8, oil: u8 },
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
    /// Lobby-level error (room not found, name taken, etc.).
    LobbyError { message: String },
    /// Current list of rooms (response to ListRooms).
    RoomList { rooms: Vec<RoomSummary> },
    /// Sent to a client when they successfully join or create a room.
    RoomJoined { room: String, your_id: PlayerId },
    /// Sent to a client when they leave a room.
    RoomLeft { room: String },
}

// ---------------------------------------------------------------------------
// Lobby protocol
// ---------------------------------------------------------------------------

/// Top-level envelope for all client→server messages in the lobby server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Lobby-level actions (room management, bot management).
    Lobby(LobbyAction),
    /// In-game action, scoped to a named room.
    Room { room: String, action: Action },
}

/// Lobby-level actions not routed through `apply_action`.
/// Uses `"action"` as the tag field (not `"type"`) so it can be inlined
/// into the parent `ClientMessage` object without a duplicate `"type"` key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LobbyAction {
    /// List all current rooms.
    ListRooms,
    /// Create a new room with the given name.
    CreateRoom { name: String },
    /// Join an existing room.
    JoinRoom { name: String },
    /// Leave the current room.
    LeaveRoom,
    /// Add an in-process bot to the current room (host only, lobby phase only).
    AddBot {
        bot_name: String,
        color: PlayerColor,
    },
    /// Remove a bot from the current room (host only, lobby phase only).
    RemoveBot { bot_id: PlayerId },
}

/// Summary of a room for the room-list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSummary {
    pub name: String,
    pub player_count: u8,
    pub max_players: u8,
    pub in_lobby: bool,
    pub has_started: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

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

    #[test]
    fn test_lobby_error_serde_roundtrip() {
        let msg = ServerMessage::LobbyError {
            message: "room not found".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ServerMessage::LobbyError { message } if message == "room not found")
        );
    }

    #[test]
    fn test_room_joined_serde_roundtrip() {
        let id = Uuid::new_v4();
        let msg = ServerMessage::RoomJoined {
            room: "alpha".to_string(),
            your_id: id,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ServerMessage::RoomJoined { room, your_id } if room == "alpha" && your_id == id)
        );
    }

    #[test]
    fn test_room_left_serde_roundtrip() {
        let msg = ServerMessage::RoomLeft {
            room: "alpha".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ServerMessage::RoomLeft { room } if room == "alpha"));
    }

    #[test]
    fn test_room_list_serde_roundtrip() {
        let msg = ServerMessage::RoomList {
            rooms: vec![RoomSummary {
                name: "friday".to_string(),
                player_count: 2,
                max_players: 6,
                in_lobby: true,
                has_started: false,
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ServerMessage::RoomList { rooms } if rooms.len() == 1));
    }

    #[test]
    fn test_client_message_lobby_serde_roundtrip() {
        let msg = ClientMessage::Lobby(LobbyAction::CreateRoom {
            name: "test-room".to_string(),
        });
        let json = serde_json::to_string(&msg).unwrap();
        // Verify the wire format: type=lobby, action=create_room, name=test-room all in one object.
        assert!(json.contains("\"type\":\"lobby\""), "json: {json}");
        assert!(json.contains("\"action\":\"create_room\""), "json: {json}");
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ClientMessage::Lobby(LobbyAction::CreateRoom { name }) if name == "test-room")
        );
    }

    #[test]
    fn test_client_message_room_serde_roundtrip() {
        let msg = ClientMessage::Room {
            room: "my-room".to_string(),
            action: Action::StartGame,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ClientMessage::Room { room, action: Action::StartGame } if room == "my-room")
        );
    }
}
