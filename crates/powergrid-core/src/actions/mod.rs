mod game;
mod hints;
mod protocol;

pub use game::{Action, ActionError};
pub use hints::HintPayload;
pub use protocol::{AuthError, ClientMessage, LobbyAction, LobbyError, RoomSummary, ServerMessage};

/// Current wire protocol version. Increment this on every breaking protocol change.
pub const PROTOCOL_VERSION: u32 = 2;
