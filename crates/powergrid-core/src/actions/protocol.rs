use crate::types::PlayerId;
use serde::{Deserialize, Serialize};

use super::game::Action;

/// Messages sent from the server to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Sent after a successful Authenticate handshake.
    Authenticated { user_id: PlayerId, username: String },
    /// Sent when authentication fails; connection will be closed.
    AuthError { message: String },
    /// Sent immediately on connection so the client knows its own player ID (legacy server only).
    Welcome { your_id: PlayerId },
    /// Wire-safe game state broadcast after every valid action (no hidden deck, no map).
    StateUpdate(Box<crate::state::GameStateView>),
    /// Sent only to the client whose action was rejected.
    ActionError { message: String },
    /// Incremental event message (e.g. "Hamburg was built by Red").
    Event { message: String },
    /// Lobby-level error (room not found, name taken, etc.).
    LobbyError { message: String },
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
}

// ---------------------------------------------------------------------------
// Lobby protocol
// ---------------------------------------------------------------------------

/// Top-level envelope for all client→server messages in the lobby server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Must be the first message sent after connecting; carries the session token.
    Authenticate { token: String },
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
        color: crate::types::PlayerColor,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
