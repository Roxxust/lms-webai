use axum::{
    extract::{Json, State, ConnectInfo},
    http::{header, Method, StatusCode, HeaderValue, Request as HttpRequest, Response as HttpResponse},
    response::Html,
    routing::{get, post},
    Router, body::Body, middleware::Next,
};
use bcrypt::{hash, verify, DEFAULT_COST};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, net::{SocketAddr, IpAddr}, sync::Arc, time::{Duration, SystemTime, UNIX_EPOCH}};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

#[derive(Serialize)]
struct PromptResponse {
    response: String,
}

#[derive(Deserialize)]
struct AuthRequest {
    password: String,
}

#[derive(Serialize)]
struct AuthResponse {
    token: String,
    expires_in: u64,
}

#[derive(Clone)]
struct AppState {
    lmstudio_url: String,
    hashed_password: String,
    active_tokens: Arc<RwLock<HashMap<String, TokenInfo>>>,
}

#[derive(Debug, Clone)]
struct TokenInfo {
    expires_at: SystemTime,
    last_used: SystemTime,
}

impl TokenInfo {
    fn new() -> Self {
        let now = SystemTime::now();
        let expires_at = now + Duration::from_secs(7200);
        Self {
            expires_at,
            last_used: now,
        }
    }
    
    fn is_expired(&self) -> bool {
        self.expires_at.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default()
            < SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
    }
    
    fn update_last_used(&mut self) {
        self.last_used = SystemTime::now();
    }
}

const HTML_INTERFACE: &str = include_str!("index.html");

async fn ip_filter_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: HttpRequest<Body>,
    next: Next,
) -> Result<HttpResponse<Body>, StatusCode> {
    let ip = addr.ip();
    
    if ip.is_loopback() || is_private_ip(&ip) {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_private() || 
            ipv4.is_loopback() || 
            ipv4.is_link_local() ||
            (ipv4.octets()[0] == 192 && ipv4.octets()[1] == 168)
        },
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback() || 
            ipv6.segments()[0] == 0xfc00 || 
            ipv6.segments()[0] == 0xfd00
        }
    }
}

async fn root() -> Html<&'static str> {
    Html(HTML_INTERFACE)
}

async fn authenticate(
    State(state): State<AppState>,
    Json(payload): Json<AuthRequest>,
) -> Result<Json<AuthResponse>, StatusCode> {
    match verify(&payload.password, &state.hashed_password) {
        Ok(true) => {
            let token = format!("lmst-{}{}", 
                &Uuid::new_v4().to_string().replace("-", "")[..8],
                &Uuid::new_v4().to_string().replace("-", "")[..8]
            );
            
            let token_info = TokenInfo::new();
            let expires_in = token_info.expires_at.duration_since(SystemTime::now())
                .map(|d| d.as_secs())
                .unwrap_or(7200);
            
            state.active_tokens.write().await.insert(token.clone(), token_info);
            
            Ok(Json(AuthResponse {
                token,
                expires_in,
            }))
        },
        Ok(false) => {
            Err(StatusCode::UNAUTHORIZED)
        },
        Err(_) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn handle_prompt(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<PromptRequest>,
) -> Result<Json<PromptResponse>, (StatusCode, String)> {
    if payload.prompt.len() > 2000 {
        return Err((StatusCode::BAD_REQUEST, "Prompt too long".to_string()));
    }

    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("Bearer "))
        .ok_or((StatusCode::UNAUTHORIZED, "Missing authorization".to_string()))?;

    let mut tokens = state.active_tokens.write().await;
    if let Some(token_info) = tokens.get_mut(auth_header) {
        if token_info.is_expired() {
            tokens.remove(auth_header);
            return Err((StatusCode::UNAUTHORIZED, "Token expired".to_string()));
        }
        token_info.update_last_used();
    } else {
        return Err((StatusCode::UNAUTHORIZED, "Invalid token".to_string()));
    }
    drop(tokens);

    let client = reqwest::Client::new();
    
    let lm_request = serde_json::json!({
        "prompt": payload.prompt,
        "temperature": 0.7,
        "max_tokens": 512
    });

    let response = client
        .post(&state.lmstudio_url)
        .json(&lm_request)
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| {
            eprintln!("Failed to send request to LM Studio: {}", e);
            if e.is_timeout() {
                (StatusCode::GATEWAY_TIMEOUT, "LM Studio timeout".to_string())
            } else {
                (StatusCode::BAD_GATEWAY, "Failed to connect to LM Studio".to_string())
            }
        })?;

    if response.status().is_success() {
        let lm_response: serde_json::Value = response
            .json()
            .await
            .map_err(|e| {
                eprintln!("Failed to parse LM Studio response: {}", e);
                (StatusCode::BAD_GATEWAY, "Invalid response from LM Studio".to_string())
            })?;

        eprintln!("LM Studio full response: {:?}", lm_response);

        let text_response = if state.lmstudio_url.contains("/chat/completions") {
            lm_response["choices"]
                .as_array()
                .and_then(|choices| choices.first())
                .and_then(|choice| choice["message"]["content"].as_str())
                .unwrap_or("No response content")
                .to_string()
        } else {
            lm_response["choices"]
                .as_array()
                .and_then(|choices| choices.first())
                .and_then(|choice| choice["text"].as_str())
                .unwrap_or("No response content")
                .to_string()
        };

        Ok(Json(PromptResponse {
            response: text_response,
        }))
    } else {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        eprintln!("LM Studio API error {}: {}", status, error_text);
        Err((StatusCode::BAD_GATEWAY, format!("LM Studio error: {}", error_text)))
    }
}

async fn logout(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> Result<StatusCode, StatusCode> {
    let auth_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("Bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    state.active_tokens.write().await.remove(auth_header);
    Ok(StatusCode::OK)
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn cleanup_expired_tokens(state: AppState) {
    let mut tokens = state.active_tokens.write().await;
    let now = SystemTime::now();
    tokens.retain(|_, token_info| {
        token_info.expires_at > now
    });
}

fn get_local_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let local_addr = socket.local_addr().ok()?;
    Some(local_addr.ip().to_string())
}

#[tokio::main]
async fn main() {
    dotenvy::from_filename(".env").ok();
    
    let password = env::var("PASSWORD").expect("PASSWORD must be set in .env");
    let lmstudio_url = env::var("LMSTUDIO_URL").unwrap_or("http://localhost:1234/v1/completions".to_string());
    let server_port: u16 = env::var("PORT").unwrap_or("3000".to_string()).parse().unwrap();
    let max_requests_per_minute: u32 = env::var("MAX_REQUESTS_PER_MINUTE").unwrap_or("5".to_string()).parse().unwrap();
    
    let hashed_password = hash(password, DEFAULT_COST).expect("Failed to hash password");

    let shared_state = AppState {
        lmstudio_url,
        hashed_password,
        active_tokens: Arc::new(RwLock::new(HashMap::new())),
    };

    let cleanup_state = shared_state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(300)).await;
            cleanup_expired_tokens(cleanup_state.clone()).await;
        }
    });

    let allowed_origins: Vec<HeaderValue> = env::var("ALLOWED_ORIGINS")
        .unwrap_or("http://localhost:3000,http://127.0.0.1:3000".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    let app = Router::new()
        .route("/", get(root))
        .route("/auth", post(authenticate))
        .route("/prompt", post(handle_prompt))
        .route("/logout", post(logout))
        .route("/health", get(health_check))
        .with_state(shared_state)
        .layer(axum::middleware::from_fn(ip_filter_middleware))
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], server_port));
    println!("Server running on http://{}", addr);
    
    if let Some(local_ip) = get_local_ip() {
        println!("Access from other devices at: http://{}:{}", local_ip, server_port);
    }
    println!("Token expiration: 2 hours");
    println!("Rate limit: {} requests per minute", max_requests_per_minute);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap();
}