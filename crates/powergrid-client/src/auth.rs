use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Saved credentials (persisted to disk)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedCredentials {
    pub token: String,
    pub user_id: Uuid,
    pub username: String,
    pub server: String,
    pub port: u16,
}

fn credentials_path() -> Option<PathBuf> {
    ProjectDirs::from("net", "onyxoryx", "powergrid")
        .map(|d| d.config_dir().join("credentials.json"))
}

pub fn load_credentials() -> Option<SavedCredentials> {
    let path = credentials_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn save_credentials(c: &SavedCredentials) -> std::io::Result<()> {
    let path =
        credentials_path().ok_or_else(|| std::io::Error::other("could not resolve config dir"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(c).map_err(std::io::Error::other)?;
    std::fs::write(&path, data)?;
    // Restrict permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn clear_credentials() -> std::io::Result<()> {
    if let Some(path) = credentials_path() {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Auth channel — sends requests from Bevy to a background thread
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AuthEvent {
    Success(SavedCredentials),
    Failure(String),
    LoggedOut,
}

/// Shared result slot: background thread writes here, Bevy reads it each frame.
#[derive(Clone)]
pub struct AuthPendingSlot(pub Arc<Mutex<Option<AuthEvent>>>);

impl AuthPendingSlot {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    pub fn take(&self) -> Option<AuthEvent> {
        self.0.lock().unwrap().take()
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers (blocking, called from std::thread::spawn)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct RegisterBody<'a> {
    email: &'a str,
    username: &'a str,
    password: &'a str,
}

#[derive(Serialize)]
struct LoginBody<'a> {
    identifier: &'a str,
    password: &'a str,
}

#[derive(Deserialize)]
struct AuthRespBody {
    token: String,
    user_id: Uuid,
    username: String,
}

#[derive(Deserialize)]
struct ErrRespBody {
    error: String,
}

fn http_base(server: &str, port: u16) -> String {
    format!("http://{}:{}", server, port)
}

pub fn do_register(
    server: &str,
    port: u16,
    email: &str,
    username: &str,
    password: &str,
) -> Result<SavedCredentials, String> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/auth/register", http_base(server, port)))
        .json(&RegisterBody {
            email,
            username,
            password,
        })
        .send()
        .map_err(|e| format!("connection error: {e}"))?;

    let status = resp.status();
    if status.is_success() {
        let body: AuthRespBody = resp.json().map_err(|e| e.to_string())?;
        Ok(SavedCredentials {
            token: body.token,
            user_id: body.user_id,
            username: body.username,
            server: server.to_string(),
            port,
        })
    } else {
        let body: ErrRespBody = resp.json().unwrap_or(ErrRespBody {
            error: status.to_string(),
        });
        Err(body.error)
    }
}

pub fn do_login(
    server: &str,
    port: u16,
    identifier: &str,
    password: &str,
) -> Result<SavedCredentials, String> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/auth/login", http_base(server, port)))
        .json(&LoginBody {
            identifier,
            password,
        })
        .send()
        .map_err(|e| format!("connection error: {e}"))?;

    let status = resp.status();
    if status.is_success() {
        let body: AuthRespBody = resp.json().map_err(|e| e.to_string())?;
        Ok(SavedCredentials {
            token: body.token,
            user_id: body.user_id,
            username: body.username,
            server: server.to_string(),
            port,
        })
    } else {
        let body: ErrRespBody = resp.json().unwrap_or(ErrRespBody {
            error: status.to_string(),
        });
        Err(body.error)
    }
}

pub fn do_logout(server: &str, port: u16, token: &str) {
    if let Ok(client) = reqwest::blocking::Client::builder().build() {
        let _ = client
            .post(format!("{}/auth/logout", http_base(server, port)))
            .header("Authorization", format!("Bearer {}", token))
            .send();
    }
}
