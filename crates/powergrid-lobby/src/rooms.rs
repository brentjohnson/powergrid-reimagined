use powergrid_core::{
    actions::{Action, RoomSummary, ServerMessage},
    map::Map,
    rules::apply_action,
    types::{Phase, PlayerColor, PlayerId},
    GameState,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::info;
use uuid::Uuid;

#[allow(dead_code)]
pub struct BotSlot {
    pub id: PlayerId,
    pub name: String,
    pub color: PlayerColor,
}

pub struct Room {
    pub name: String,
    pub game: GameState,
    pub humans: Vec<(PlayerId, mpsc::UnboundedSender<String>)>,
    pub bots: Vec<BotSlot>,
    /// socket_id of the first human who created the room (for permission checks on lobby actions)
    pub creator_socket: PlayerId,
}

impl Room {
    pub fn new(name: String, game: GameState, creator_socket: PlayerId) -> Self {
        Self {
            name,
            game,
            humans: Vec::new(),
            bots: Vec::new(),
            creator_socket,
        }
    }

    pub fn broadcast(&mut self, json: &str) {
        self.humans
            .retain(|(_, tx)| tx.send(json.to_string()).is_ok());
    }

    pub fn broadcast_msg(&mut self, msg: &ServerMessage) {
        let json = serde_json::to_string(msg).unwrap();
        self.broadcast(&json);
    }

    /// Add an in-process bot to the room. Only valid in Phase::Lobby.
    pub fn add_bot(&mut self, bot_name: String, color: PlayerColor) -> Result<PlayerId, String> {
        let bot_id = Uuid::new_v4();
        apply_action(
            &mut self.game,
            bot_id,
            Action::JoinGame {
                name: bot_name.clone(),
                color,
            },
        )
        .map_err(|e| e.to_string())?;
        info!(
            "Bot '{}' ({:?}) added to room '{}'",
            bot_name, color, self.name
        );
        self.bots.push(BotSlot {
            id: bot_id,
            name: bot_name,
            color,
        });
        Ok(bot_id)
    }

    /// Remove a bot from the room. Only valid in Phase::Lobby.
    pub fn remove_bot(&mut self, bot_id: PlayerId) -> Result<(), String> {
        if !matches!(self.game.phase, Phase::Lobby) {
            return Err("cannot remove bot after game has started".to_string());
        }
        let idx = self
            .bots
            .iter()
            .position(|b| b.id == bot_id)
            .ok_or_else(|| "bot not found in this room".to_string())?;
        self.bots.remove(idx);
        self.game.players.retain(|p| p.id != bot_id);
        self.game.player_order.retain(|id| *id != bot_id);
        info!("Bot {} removed from room '{}'", bot_id, self.name);
        Ok(())
    }

    pub fn summary(&self) -> RoomSummary {
        RoomSummary {
            name: self.name.clone(),
            player_count: self.game.players.len() as u8,
            max_players: 6,
            in_lobby: matches!(self.game.phase, Phase::Lobby),
            has_started: !matches!(self.game.phase, Phase::Lobby),
        }
    }

    pub fn human_count(&self) -> usize {
        self.humans.len()
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.game.phase, Phase::GameOver { .. })
    }
}

pub struct RoomManager {
    rooms: RwLock<HashMap<String, Arc<Mutex<Room>>>>,
    default_map: Arc<Map>,
}

impl RoomManager {
    pub fn new(default_map: Map) -> Self {
        Self {
            rooms: RwLock::new(HashMap::new()),
            default_map: Arc::new(default_map),
        }
    }

    pub async fn list(&self) -> Vec<RoomSummary> {
        let rooms = self.rooms.read().await;
        let mut summaries = Vec::new();
        for room_arc in rooms.values() {
            let room = room_arc.lock().await;
            summaries.push(room.summary());
        }
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn create(
        &self,
        name: String,
        creator_socket: PlayerId,
    ) -> Result<Arc<Mutex<Room>>, String> {
        validate_room_name(&name)?;
        let key = name.to_lowercase();
        let mut rooms = self.rooms.write().await;
        if rooms.contains_key(&key) {
            return Err(format!("a room named '{}' already exists", name));
        }
        let map = (*self.default_map).clone();
        let game = GameState::new(map, 6);
        let room = Arc::new(Mutex::new(Room::new(name, game, creator_socket)));
        rooms.insert(key, Arc::clone(&room));
        Ok(room)
    }

    pub async fn get(&self, name: &str) -> Option<Arc<Mutex<Room>>> {
        let rooms = self.rooms.read().await;
        rooms.get(&name.to_lowercase()).cloned()
    }

    /// Drop the room if game is over and no humans remain.
    pub async fn drop_if_finished(&self, name: &str) {
        let key = name.to_lowercase();
        let should_drop = {
            let rooms = self.rooms.read().await;
            if let Some(room_arc) = rooms.get(&key) {
                let room = room_arc.lock().await;
                room.is_game_over() && room.human_count() == 0
            } else {
                false
            }
        };
        if should_drop {
            let mut rooms = self.rooms.write().await;
            rooms.remove(&key);
            info!(
                "Room '{}' dropped after game over with no human connections",
                name
            );
        }
    }
}

fn validate_room_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 32 {
        return Err("room name must be 1–32 characters".to_string());
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(
            "room name may only contain letters, digits, hyphens, and underscores".to_string(),
        );
    }
    Ok(())
}
