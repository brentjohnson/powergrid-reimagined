use iced::{futures::channel::mpsc, Element, Subscription};
use powergrid_core::{
    actions::Action,
    types::{PlayerColor, Resource},
    GameState,
};
use uuid::Uuid;

use crate::connection::{self, WsEvent};
use crate::screens::{self, ConnectScreen};

#[derive(Debug, Clone)]
pub enum Message {
    // Connect screen
    ServerUrlChanged(String),
    NameChanged(String),
    ColorSelected(PlayerColor),
    Connect,

    // WebSocket events
    WsEvent(WsEvent),

    // Lobby
    StartGame,

    // Auction
    SelectPlant(u8),
    BidAmountChanged(String),
    PlaceBid,
    PassAuction,

    // Buy resources
    BuyResource(Resource),
    DoneBuying,

    // Build
    #[allow(dead_code)]
    BuildCity(String),
    DoneBuilding,

    // Bureaucracy
    PowerCities,
}

pub enum Screen {
    Connect(ConnectScreen),
    Game,
}

pub struct App {
    screen: Screen,
    game_state: Option<GameState>,
    my_id: Option<Uuid>,
    ws_sender: Option<mpsc::Sender<Action>>,
    /// Set when the user clicks Connect; drives the subscription.
    connect_url: Option<String>,
    /// Name + color saved at Connect time, sent to server after Welcome arrives.
    pending_join: Option<(String, PlayerColor)>,
    /// Current text in the bid amount input field.
    bid_amount: String,
    /// Last action error received from the server; cleared on next successful state update.
    error_message: Option<String>,
}

impl App {
    pub fn new() -> (Self, iced::Task<Message>) {
        (
            Self {
                screen: Screen::Connect(ConnectScreen::new()),
                game_state: None,
                my_id: None,
                ws_sender: None,
                connect_url: None,
                pending_join: None,
                bid_amount: String::new(),
                error_message: None,
            },
            iced::Task::none(),
        )
    }

    pub fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::ServerUrlChanged(url) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.server_url = url;
                }
            }
            Message::NameChanged(name) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.player_name = name;
                }
            }
            Message::ColorSelected(color) => {
                if let Screen::Connect(s) = &mut self.screen {
                    s.selected_color = color;
                }
            }
            Message::Connect => {
                if let Screen::Connect(s) = &self.screen {
                    self.connect_url = Some(s.server_url.clone());
                    self.pending_join = Some((s.player_name.clone(), s.selected_color));
                }
            }

            Message::WsEvent(event) => match event {
                WsEvent::Connected(sender) => {
                    self.ws_sender = Some(sender);
                    // my_id will be set when we receive the Welcome message.
                    // Switch to Game screen to show "Waiting for server..." while
                    // Welcome + JoinGame round-trip completes.
                    self.screen = Screen::Game;
                }
                WsEvent::MessageReceived(msg) => {
                    use powergrid_core::actions::ServerMessage;
                    match msg {
                        ServerMessage::Welcome { your_id } => {
                            self.my_id = Some(your_id);
                            // Now that we know our ID, send JoinGame.
                            // We need the name/color from the connect screen, which is
                            // stored in connect_url's companion — save them before switching.
                            if let Some((name, color)) = self.pending_join.take() {
                                self.send(Action::JoinGame { name, color });
                            }
                        }
                        ServerMessage::StateUpdate(state) => {
                            self.game_state = Some(state);
                            self.error_message = None;
                        }
                        ServerMessage::ActionError { message } => {
                            self.error_message = Some(message);
                        }
                        ServerMessage::Event { .. } => {}
                    }
                }
                WsEvent::Disconnected => {
                    self.ws_sender = None;
                    if matches!(self.screen, Screen::Game) {
                        // Stay on game screen; reconnect will be attempted automatically.
                    }
                }
            },

            Message::StartGame => {
                self.send(Action::StartGame);
            }
            Message::SelectPlant(num) => {
                self.send(Action::SelectPlant { plant_number: num });
            }
            Message::BidAmountChanged(val) => {
                self.bid_amount = val;
            }
            Message::PlaceBid => {
                if let Ok(amount) = self.bid_amount.trim().parse::<u32>() {
                    self.send(Action::PlaceBid { amount });
                    self.bid_amount = String::new();
                }
            }
            Message::PassAuction => {
                self.send(Action::PassAuction);
            }
            Message::BuyResource(resource) => {
                self.send(Action::BuyResources { resource, amount: 1 });
            }
            Message::DoneBuying => {
                self.send(Action::DoneBuying);
            }
            Message::BuildCity(city_id) => {
                self.send(Action::BuildCity { city_id });
            }
            Message::DoneBuilding => {
                self.send(Action::DoneBuilding);
            }
            Message::PowerCities => {
                // Fire all plants by default.
                if let Some(state) = &self.game_state {
                    if let Some(id) = self.my_id {
                        if let Some(player) = state.player(id) {
                            let plant_numbers: Vec<u8> = player.plants.iter().map(|p| p.number).collect();
                            self.send(Action::PowerCities { plant_numbers });
                        }
                    }
                }
            }
        }
        iced::Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        match &self.screen {
            Screen::Connect(s) => s.view(),
            Screen::Game => {
                if let Some(state) = &self.game_state {
                    let my_id = self.my_id.unwrap_or(Uuid::nil());
                    let is_host = state.host_id() == Some(my_id);
                    if matches!(state.phase, powergrid_core::types::Phase::Lobby) {
                        screens::lobby_view(state, is_host, self.error_message.as_deref())
                    } else {
                        screens::game_view(state, my_id, &self.bid_amount, self.error_message.as_deref())
                    }
                } else {
                    iced::widget::text("Connecting...").into()
                }
            }
        }
    }

    pub fn subscription(&self) -> Subscription<Message> {
        match &self.connect_url {
            Some(url) => connection::connect(url.clone()).map(Message::WsEvent),
            None => Subscription::none(),
        }
    }

    fn send(&mut self, action: Action) {
        if let Some(tx) = &mut self.ws_sender {
            let _ = tx.try_send(action);
        }
    }
}
