use rodio::Source;

/// Audio subsystem for the client.
///
/// Holds the rodio `OutputStream` alive (dropping it stops all playback) and
/// exposes a single `play_turn_sound()` helper.  Creating an `Audio` may fail
/// silently on headless machines or systems with no audio device; in that case
/// `Audio::new()` returns `None` and the rest of the game continues normally.
pub struct Audio {
    /// Kept alive so the audio device stays open.
    _stream: rodio::OutputStream,
    handle: rodio::OutputStreamHandle,
}

const TURN_SOUND: &[u8] = include_bytes!("../assets/confirm_style_2_echo_001.ogg");

impl Audio {
    /// Try to open the default audio output device.  Returns `None` on failure.
    pub fn new() -> Option<Self> {
        match rodio::OutputStream::try_default() {
            Ok((_stream, handle)) => Some(Self { _stream, handle }),
            Err(e) => {
                tracing::warn!("Audio device unavailable, sounds disabled: {e}");
                None
            }
        }
    }

    /// Play the turn-notification sound once (non-blocking).
    pub fn play_turn_sound(&self) {
        let cursor = std::io::Cursor::new(TURN_SOUND);
        match rodio::Decoder::new(cursor) {
            Ok(source) => {
                let _ = self.handle.play_raw(source.convert_samples());
            }
            Err(e) => {
                tracing::warn!("Failed to decode turn sound: {e}");
            }
        }
    }
}
