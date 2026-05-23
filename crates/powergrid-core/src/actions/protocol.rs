use crate::types::{BotDifficulty, PlayerId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::game::Action;
use super::hints::HintPayload;

// ---------------------------------------------------------------------------
// Structured error types
// ---------------------------------------------------------------------------

/// Authentication failure reasons, sent before the connection is closed.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthError {
    #[error("authentication timeout")]
    Timeout,
    #[error("invalid or expired token")]
    InvalidToken,
    #[error("authenticate first")]
    AuthRequired,
    #[error(
        "protocol version mismatch: server requires {server_version}, client sent {client_version}"
    )]
    VersionMismatch {
        server_version: u32,
        client_version: u32,
    },
}

/// Lobby-level errors (room management, bot management, membership checks).
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LobbyError {
    #[error("room '{name}' not found")]
    RoomNotFound { name: String },
    #[error("a room named '{name}' already exists")]
    RoomAlreadyExists { name: String },
    #[error("room name is invalid")]
    InvalidRoomName,
    #[error("leave your current room before joining another")]
    AlreadyInRoom,
    #[error("not in any room")]
    NotInRoom,
    #[error("you are not in this room")]
    NotMember,
    #[error("only the room host can do that")]
    NotHost,
    #[error("cannot do that after the game has started")]
    GameAlreadyStarted,
    #[error("bot name must not be empty")]
    BotNameEmpty,
    #[error("bot name is too long")]
    BotNameTooLong,
    #[error("bot not found")]
    BotNotFound,
    #[error("invalid message: {message}")]
    InvalidMessage { message: String },
    #[error("failed to add bot: {error}")]
    BotAddFailed { error: super::game::ActionError },
}

// ---------------------------------------------------------------------------
// Server → client messages
// ---------------------------------------------------------------------------

/// Messages sent from the server to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Sent after a successful Authenticate handshake.
    Authenticated { user_id: PlayerId, username: String },
    /// Sent when authentication fails; connection will be closed.
    AuthError { error: AuthError },
    /// Wire-safe game state broadcast after every valid action (no hidden deck, no map).
    StateUpdate(Box<crate::state::GameStateView>),
    /// Sent only to the client whose in-game action was rejected.
    ActionError { error: super::game::ActionError },
    /// Incremental event message (e.g. "Hamburg was built by Red").
    Event { message: String },
    /// Lobby-level error (room not found, name taken, etc.).
    LobbyError { error: LobbyError },
    /// Current list of rooms (response to ListRooms).
    RoomList { rooms: Vec<RoomSummary> },
    /// Sent to a client when they successfully join or create a room.
    /// Includes the full static map (sent once; subsequent StateUpdates omit it).
    RoomJoined {
        room: String,
        your_id: PlayerId,
        map: Box<crate::map::Map>,
    },
    /// Sent to a client when they leave a room.
    RoomLeft { room: String },
    /// Ephemeral peer selection hint relayed from another client in the same room.
    PeerHint {
        player_id: PlayerId,
        hint: HintPayload,
    },
}

// ---------------------------------------------------------------------------
// Client → server messages
// ---------------------------------------------------------------------------

/// Top-level envelope for all client→server messages in the lobby server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Must be the first message sent after connecting; carries the session token
    /// and the client's protocol version (must match `PROTOCOL_VERSION`).
    Authenticate {
        token: String,
        protocol_version: u32,
    },
    /// Lobby-level actions (room management, bot management).
    Lobby { action: LobbyAction },
    /// In-game action, scoped to a named room.
    Room { room: String, action: Action },
    /// Ephemeral selection hint (cart, city picks, etc.) — not a game action.
    RoomHint { room: String, hint: HintPayload },
}

/// Lobby-level actions not routed through `apply_action`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
        color: crate::types::PlayerColor,
        difficulty: BotDifficulty,
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
    pub has_started: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::PROTOCOL_VERSION;
    use uuid::Uuid;

    #[test]
    fn test_auth_error_serde_roundtrip() {
        let msg = ServerMessage::AuthError {
            error: AuthError::Timeout,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            ServerMessage::AuthError {
                error: AuthError::Timeout
            }
        ));
    }

    #[test]
    fn test_auth_error_version_mismatch_serde() {
        let msg = ServerMessage::AuthError {
            error: AuthError::VersionMismatch {
                server_version: 2,
                client_version: 1,
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"auth_error\""));
        assert!(json.contains("\"type\":\"version_mismatch\""));
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            ServerMessage::AuthError {
                error: AuthError::VersionMismatch {
                    server_version: 2,
                    client_version: 1,
                }
            }
        ));
    }

    #[test]
    fn test_action_error_serde_roundtrip() {
        let msg = ServerMessage::ActionError {
            error: crate::actions::ActionError::WrongPhase,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            ServerMessage::ActionError {
                error: crate::actions::ActionError::WrongPhase
            }
        ));
    }

    #[test]
    fn test_lobby_error_serde_roundtrip() {
        let msg = ServerMessage::LobbyError {
            error: LobbyError::RoomNotFound {
                name: "test".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            ServerMessage::LobbyError {
                error: LobbyError::RoomNotFound { name }
            } if name == "test"
        ));
    }

    #[test]
    fn test_room_joined_serde_roundtrip() {
        let id = Uuid::new_v4();
        let map = crate::map::default_map();
        let msg = ServerMessage::RoomJoined {
            room: "alpha".to_string(),
            your_id: id,
            map: Box::new(map),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ServerMessage::RoomJoined { room, your_id, .. } if room == "alpha" && your_id == id)
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
                has_started: false,
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ServerMessage::RoomList { rooms } if rooms.len() == 1));
    }

    #[test]
    fn test_authenticate_carries_protocol_version() {
        let msg = ClientMessage::Authenticate {
            token: "abc".to_string(),
            protocol_version: PROTOCOL_VERSION,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(
            json.contains("\"protocol_version\""),
            "json must contain protocol_version: {json}"
        );
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ClientMessage::Authenticate { protocol_version, .. } if protocol_version == PROTOCOL_VERSION)
        );
    }

    #[test]
    fn test_client_message_lobby_nested_type_tag() {
        let msg = ClientMessage::Lobby {
            action: LobbyAction::CreateRoom {
                name: "test-room".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        // Both levels use "type" as the tag key — no duplicate keys at one level.
        assert!(json.contains("\"type\":\"lobby\""), "json: {json}");
        assert!(json.contains("\"type\":\"create_room\""), "json: {json}");
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(parsed, ClientMessage::Lobby { action: LobbyAction::CreateRoom { name } } if name == "test-room")
        );
    }

    #[test]
    fn test_add_bot_difficulty_required() {
        let msg = ClientMessage::Lobby {
            action: LobbyAction::AddBot {
                bot_name: "BotA".to_string(),
                color: crate::types::PlayerColor::Red,
                difficulty: BotDifficulty::Hard,
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            ClientMessage::Lobby {
                action: LobbyAction::AddBot {
                    difficulty: BotDifficulty::Hard,
                    ..
                }
            }
        ));
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
