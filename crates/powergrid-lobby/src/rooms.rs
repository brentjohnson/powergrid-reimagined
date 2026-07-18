use powergrid_core::{
    actions::{LobbyError, RoomSummary, ServerMessage},
    limits::MAX_ROOM_NAME,
    types::{BotDifficulty, Phase, PlayerColor, PlayerId},
};
use powergrid_session::{Map, Session, Subscriber, MAX_PLAYERS};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::info;

pub struct Room {
    pub name: String,
    pub session: Session,
    /// WS senders keyed by user_id for targeted per-player messages.
    pub humans: Vec<(PlayerId, mpsc::UnboundedSender<String>)>,
    /// user_id of the human who created the room (survives reconnects).
    pub creator_user_id: PlayerId,
    /// When the game left the Lobby phase (for game-result persistence).
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Once-only guard so a finished game is recorded to the DB exactly once.
    pub results_recorded: bool,
}

impl Room {
    pub fn new(name: String, map: Map, creator_user_id: PlayerId) -> Self {
        Self {
            name,
            session: Session::new(map, MAX_PLAYERS),
            humans: Vec::new(),
            creator_user_id,
            started_at: None,
            results_recorded: false,
        }
    }

    pub fn broadcast(&mut self, json: &str) {
        self.session.broadcast_json(json);
        // Also prune our humans list to match live subscribers.
        self.humans.retain(|(_, tx)| !tx.is_closed());
    }

    pub fn broadcast_msg(&mut self, msg: &ServerMessage) {
        self.session.broadcast(msg);
        self.humans.retain(|(_, tx)| !tx.is_closed());
    }

    pub fn add_human(&mut self, user_id: PlayerId, tx: mpsc::UnboundedSender<String>) {
        self.session
            .add_subscriber(Subscriber::mpsc(Some(user_id), tx.clone()));
        self.humans.push((user_id, tx));
    }

    pub fn replace_human(&mut self, user_id: PlayerId, tx: mpsc::UnboundedSender<String>) {
        // Update the WS sender in our tracking list.
        if let Some(slot) = self.humans.iter_mut().find(|(id, _)| *id == user_id) {
            slot.1 = tx.clone();
        } else {
            self.humans.push((user_id, tx.clone()));
        }
        // Add a fresh subscriber (stale ones are pruned on next broadcast).
        self.session
            .add_subscriber(Subscriber::mpsc(Some(user_id), tx));
    }

    pub fn add_bot(
        &mut self,
        bot_name: String,
        color: PlayerColor,
        difficulty: BotDifficulty,
    ) -> Result<PlayerId, LobbyError> {
        self.session
            .add_bot(bot_name, color, difficulty)
            .map_err(|e| LobbyError::BotAddFailed { error: e })
    }

    pub fn remove_bot(&mut self, bot_id: PlayerId) -> Result<(), LobbyError> {
        self.session.remove_bot(bot_id).map_err(|e| {
            if e.contains("game has started") {
                LobbyError::GameAlreadyStarted
            } else {
                LobbyError::BotNotFound
            }
        })
    }

    pub fn summary(&self) -> RoomSummary {
        RoomSummary {
            name: self.name.clone(),
            player_count: self.session.game.players.len() as u8,
            max_players: MAX_PLAYERS,
            has_started: !matches!(self.session.game.phase, Phase::Lobby),
        }
    }

    pub fn human_count(&self) -> usize {
        self.humans.len()
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.session.game.phase, Phase::GameOver { .. })
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
        creator_user_id: PlayerId,
    ) -> Result<Arc<Mutex<Room>>, LobbyError> {
        validate_room_name(&name)?;
        let key = name.to_lowercase();
        let mut rooms = self.rooms.write().await;
        if rooms.contains_key(&key) {
            return Err(LobbyError::RoomAlreadyExists { name });
        }
        let map = (*self.default_map).clone();
        let room = Arc::new(Mutex::new(Room::new(name, map, creator_user_id)));
        rooms.insert(key, Arc::clone(&room));
        Ok(room)
    }

    pub async fn get(&self, name: &str) -> Option<Arc<Mutex<Room>>> {
        let rooms = self.rooms.read().await;
        rooms.get(&name.to_lowercase()).cloned()
    }

    /// Drop the room if no humans remain and it's either finished or still an
    /// unstarted (Lobby-phase) room. Dropping empty Lobby rooms keeps a stale
    /// pre-game room from lingering and blocking re-creation with the same name
    /// (e.g. after a disconnect during setup).
    pub async fn drop_if_finished(&self, name: &str) {
        let key = name.to_lowercase();
        let should_drop = {
            let rooms = self.rooms.read().await;
            if let Some(room_arc) = rooms.get(&key) {
                let room = room_arc.lock().await;
                room.human_count() == 0
                    && (room.is_game_over() || matches!(room.session.game.phase, Phase::Lobby))
            } else {
                false
            }
        };
        if should_drop {
            let mut rooms = self.rooms.write().await;
            rooms.remove(&key);
            info!("Room '{}' dropped: no human connections remain", name);
        }
    }
}

fn validate_room_name(name: &str) -> Result<(), LobbyError> {
    let len = name.chars().count();
    if len == 0 || len > MAX_ROOM_NAME {
        return Err(LobbyError::InvalidRoomName);
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(LobbyError::InvalidRoomName);
    }
    Ok(())
}
