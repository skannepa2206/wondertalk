import streamlit as st
import pyttsx3
import wikipedia
import uuid
import os
import base64
import html
from string import Template
import threading
import logging
from logging.handlers import RotatingFileHandler
import speech_recognition as sr
from dotenv import load_dotenv
import chromadb
import requests

# UI config must be the first Streamlit command
st.set_page_config(page_title="WonderTalk", layout="wide")

# Load environment
load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(APP_DIR, "chroma_db")
LOG_DIR = os.path.join(APP_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")
ASSETS_DIR = os.path.join(APP_DIR, "assets")
MATRICS_FONT_PATH = os.path.join(ASSETS_DIR, "matrics.ttf")

logger = logging.getLogger("wondertalk")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False


def log_exception(context: str, exc: Exception) -> None:
    logger.exception("%s: %s", context, exc)


def _load_matrics_font_css() -> tuple[str, str]:
    if not os.path.exists(MATRICS_FONT_PATH):
        return "", "Ndot57"
    try:
        with open(MATRICS_FONT_PATH, "rb") as font_file:
            b64 = base64.b64encode(font_file.read()).decode("utf-8")
        css = (
            "@font-face {"
            "font-family: 'Matrics';"
            f"src: url('data:font/ttf;base64,{b64}') format('truetype');"
            "font-weight: 400;"
            "font-style: normal;"
            "font-display: swap;"
            "}"
        )
        return css, "Matrics"
    except Exception as exc:
        log_exception("Failed to load Matrics font", exc)
        return "", "Ndot57"


def _normalize_endpoint(url: str) -> str:
    if not url:
        return ""
    cleaned = url.strip().rstrip("/")
    suffix = "/openai/v1"
    if cleaned.endswith(suffix):
        cleaned = cleaned[: -len(suffix)]
    return cleaned


AZURE_OPENAI_ENDPOINT = _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", "").strip()
if not AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_BASE_URL:
    AZURE_OPENAI_ENDPOINT = _normalize_endpoint(AZURE_OPENAI_BASE_URL)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01").strip()

AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_OPENAI_DEPLOYMENT_NAME_MAP = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "").strip()
if not AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_DEPLOYMENT_NAME_MAP:
    if "=" in AZURE_OPENAI_DEPLOYMENT_NAME_MAP:
        right = AZURE_OPENAI_DEPLOYMENT_NAME_MAP.split("=", 1)[1].strip()
        left = AZURE_OPENAI_DEPLOYMENT_NAME_MAP.split("=", 1)[0].strip()
        AZURE_OPENAI_DEPLOYMENT = right or left
    else:
        AZURE_OPENAI_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT_NAME_MAP

AZURE_OPENAI_API_MODE = os.getenv("AZURE_OPENAI_API_MODE", "").strip().lower()
AZURE_OPENAI_RESPONSES_URL = os.getenv("AZURE_OPENAI_RESPONSES_URL", "").strip()
AZURE_OPENAI_TEMPERATURE = os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7").strip()
AZURE_OPENAI_MAX_TOKENS = os.getenv("AZURE_OPENAI_MAX_TOKENS", "400").strip()
AZURE_OPENAI_MAX_OUTPUT_TOKENS = os.getenv("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "400").strip()
AZURE_OPENAI_RESPONSES_TEMPERATURE = os.getenv("AZURE_OPENAI_RESPONSES_TEMPERATURE", "").strip()

_PLACEHOLDER_KEYS = {
    "your_key_here",
    "your_azure_openai_key",
    "your_azure_openai_key_here",
    "azure_openai_key",
    "azure_api_key",
    "changeme",
}


def using_responses_api() -> bool:
    if AZURE_OPENAI_API_MODE:
        return AZURE_OPENAI_API_MODE == "responses"
    return bool(AZURE_OPENAI_RESPONSES_URL)


def _build_responses_url() -> str:
    if AZURE_OPENAI_RESPONSES_URL:
        return AZURE_OPENAI_RESPONSES_URL
    if not AZURE_OPENAI_ENDPOINT:
        return ""
    base = AZURE_OPENAI_ENDPOINT.rstrip("/")
    if base.endswith("/openai/v1"):
        return f"{base}/responses"
    if AZURE_OPENAI_API_VERSION:
        return f"{base}/openai/responses?api-version={AZURE_OPENAI_API_VERSION}"
    return f"{base}/openai/responses"


def azure_config_ready() -> bool:
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        return False
    if AZURE_OPENAI_API_KEY.lower() in _PLACEHOLDER_KEYS:
        return False
    if "<" in AZURE_OPENAI_DEPLOYMENT:
        return False
    if using_responses_api():
        url = _build_responses_url()
        if not url or "<" in url:
            return False
        return True
    if not AZURE_OPENAI_ENDPOINT:
        return False
    if "<" in AZURE_OPENAI_ENDPOINT:
        return False
    return True


def _raise_for_error(resp: requests.Response) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json()
    except Exception:
        detail = resp.text
    message = _format_azure_error(detail)
    if "Unsupported parameter" in message and "temperature" in message:
        message = f"{message} (Tip: remove temperature for this model.)"
    logger.error("Azure OpenAI error %s: %s", resp.status_code, message)
    raise RuntimeError(f"Azure OpenAI error {resp.status_code}: {message}")


def _extract_response_text(data: dict) -> str:
    if isinstance(data, dict):
        if isinstance(data.get("output_text"), str):
            return data["output_text"].strip()
        output = data.get("output")
        if isinstance(output, list):
            texts = []
            for item in output:
                content = item.get("content", []) if isinstance(item, dict) else []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                    elif isinstance(part, str):
                        texts.append(part)
            if texts:
                return "\n".join(texts).strip()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                return str(msg["content"]).strip()
    return str(data)


def _format_azure_error(detail) -> str:
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code")
            if msg:
                return str(msg)
    return str(detail)


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def call_azure_chat(prompt_text: str) -> str:
    logger.info("Azure chat request start (deployment=%s)", AZURE_OPENAI_DEPLOYMENT)
    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}"
        f"/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": _to_float(AZURE_OPENAI_TEMPERATURE, 0.7),
        "max_tokens": _to_int(AZURE_OPENAI_MAX_TOKENS, 400),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    _raise_for_error(resp)
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_azure_responses(prompt_text: str) -> str:
    logger.info("Azure responses request start (model=%s)", AZURE_OPENAI_DEPLOYMENT)
    url = _build_responses_url()
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "input": prompt_text,
        "max_output_tokens": _to_int(AZURE_OPENAI_MAX_OUTPUT_TOKENS, 400),
    }
    if AZURE_OPENAI_RESPONSES_TEMPERATURE:
        payload["temperature"] = _to_float(AZURE_OPENAI_RESPONSES_TEMPERATURE, 0.7)
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    _raise_for_error(resp)
    data = resp.json()
    return _extract_response_text(data)


def call_azure_openai(prompt_text: str) -> str:
    if using_responses_api():
        return call_azure_responses(prompt_text)
    return call_azure_chat(prompt_text)

# ChromaDB setup (persistent cache)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="prompt_store")

# TTS setup
engine = pyttsx3.init()
engine.setProperty("rate", 150)
voices = engine.getProperty("voices")
if voices:
    engine.setProperty("voice", voices[1].id if len(voices) > 1 else voices[0].id)
speech_lock = threading.Lock()

# Session state init
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "voice_active" not in st.session_state:
    st.session_state.voice_active = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_chats" not in st.session_state:
    st.session_state.recent_chats = []

if "show_pricing" not in st.session_state:
    st.session_state.show_pricing = False

if "plan_view" not in st.session_state:
    st.session_state.plan_view = "Personal"

if "mode" not in st.session_state:
    st.session_state.mode = "Kid-friendly"

if "clear_prompt" not in st.session_state:
    st.session_state.clear_prompt = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_status" not in st.session_state:
    st.session_state.last_status = ""
if "last_source" not in st.session_state:
    st.session_state.last_source = ""

# Stop flag
stop_speech_flag = threading.Event()
speech_thread = None


def speak_text(text):
    global speech_thread
    stop_speech_flag.clear()

    def _speak():
        try:
            with speech_lock:
                if stop_speech_flag.is_set():
                    return
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:
            log_exception("TTS failed", exc)

    if speech_thread and speech_thread.is_alive():
        try:
            stop_speech_flag.set()
            engine.stop()
        except Exception as exc:
            log_exception("TTS stop failed", exc)

    speech_thread = threading.Thread(target=_speak, daemon=True)
    speech_thread.start()


def stop_speech():
    stop_speech_flag.set()
    try:
        engine.stop()
    except Exception as exc:
        log_exception("TTS stop failed", exc)
    st.session_state.voice_active = False


def start_speech(text):
    st.session_state.voice_active = True
    speak_text(text)


# Prompt formatter

def smart_prompt_builder(user_prompt):
    user_prompt = user_prompt.lower()
    if user_prompt.startswith("why"):
        return f"Explain this kindly to a 6-year-old: Why {user_prompt[4:]}"
    if user_prompt.startswith("how"):
        return f"Explain how this works in a fun way for a kid: {user_prompt}"
    if "story" in user_prompt or "tell me" in user_prompt:
        return f"Tell a short, fun story for a child: {user_prompt}"
    return f"Answer this simply and kindly like to a 6-year-old: {user_prompt}"


def build_prompt(user_prompt, mode, context=""):
    if mode == "Homework":
        base = f"Explain step-by-step in a simple, encouraging way for a student: {user_prompt}"
    elif mode == "Story":
        base = f"Tell a short, fun story for a child: {user_prompt}"
    elif mode == "Explore":
        base = f"Give a curious, easy-to-follow explanation with fun facts: {user_prompt}"
    else:
        base = smart_prompt_builder(user_prompt)
    if context:
        return f"{base}\n\nConversation so far:\n{context}"
    return base


# Wikipedia fallback

def get_wikipedia_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "Sorry, I couldn't find anything on Wikipedia."


# Mic input

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.session_state.pending_prompt = query
            st.success(f"You said: {query}")
            st.rerun()
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
            logger.warning("Speech recognition: unknown value")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")
            logger.warning("Speech recognition service unavailable")


def mic_available() -> bool:
    try:
        import pyaudio  # noqa: F401
        return True
    except Exception as exc:
        if not st.session_state.get("mic_missing_logged", False):
            log_exception("PyAudio missing", exc)
            st.session_state.mic_missing_logged = True
        return False


def build_context(messages, max_turns=4):
    history = [m for m in messages if m.get("role") in ("user", "assistant")]
    history = history[-max_turns * 2 :]
    lines = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def add_message(role, content, status="", source=""):
    message = {"role": role, "content": content}
    if status:
        message["status"] = status
    if source:
        message["source"] = source
    st.session_state.messages.append(message)


def render_message(message):
    role = message.get("role", "assistant")
    content = html.escape(message.get("content", "")).replace("\n", "<br>")
    status = message.get("status", "")
    source = message.get("source", "")
    meta = ""
    if role == "assistant" and status:
        meta = f'<div class="msg-meta {source}">{status}</div>'
    st.markdown(
        f'<div class="msg {role}">{meta}<div class="bubble">{content}</div></div>',
        unsafe_allow_html=True,
    )


# UI
matrics_css, display_font = _load_matrics_font_css()

css = Template(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    @font-face {
        font-family: 'Ndot57';
        src: url('https://cdn.shopify.com/s/files/1/0581/8096/5855/files/ndot57.ttf') format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    $MATRICS_CSS

    :root {
        --bg: #0b0b0b;
        --panel: #141414;
        --panel-2: #1d1d1d;
        --text: #f5f5f5;
        --muted: #a8a8a8;
        --accent: #ff2d2d;
        --accent-2: #ff4d4d;
        --glow: rgba(255, 45, 45, 0.2);
        --font-display: '$DISPLAY_FONT', 'Space Grotesk', sans-serif;
        --font-body: 'Space Grotesk', sans-serif;
    }

    html, body, .stApp {
        font-family: var(--font-body);
        color: var(--text);
    }

    .stApp * {
        font-family: inherit;
    }

    [class*="material-icons"], [class*="material-symbols"] {
        font-family: "Material Symbols Outlined", "Material Symbols Rounded", "Material Symbols Sharp", "Material Icons", sans-serif !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }

    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stSidebarCollapsedControl"] button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 36px;
        height: 36px;
        display: grid;
        place-items: center;
    }

    div[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
    div[data-testid="stSidebarCollapsedControl"] span[data-testid="stIconMaterial"] {
        font-family: "Material Icons", "Material Symbols Outlined", sans-serif !important;
        font-size: 22px !important;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        color: var(--text);
    }

    button[data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {
        font-family: "Material Icons", "Material Symbols Outlined", sans-serif !important;
        font-size: 22px !important;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(900px 360px at 60% 0%, #1a1a1a 0%, var(--bg) 65%),
            radial-gradient(600px 260px at 20% 20%, #121212 0%, transparent 60%),
            radial-gradient(circle at 1px 1px, rgba(255,255,255,0.06) 1px, transparent 0);
        background-size: auto, auto, 20px 20px;
    }

    .main .block-container {
        padding-top: 1.6rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid #1f2430;
    }

    .hero {
        text-align: center;
        margin: 0 0 0.8rem 0;
    }

    .hero-title {
        font-family: var(--font-display);
        font-size: clamp(2.4rem, 4vw, 3.2rem);
        font-weight: 400;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        margin: 0 auto 1rem;
        max-width: 620px;
    }

    .hero-accent {
        width: 220px;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        margin: 0.4rem auto 0.9rem;
        opacity: 0.8;
    }

    .panel-shell {
        background: var(--panel);
        border: 1px solid #262626;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    .panel-anchor {
        height: 0;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlock"]:has(.panel-anchor) {
        background: var(--panel);
        border: 1px solid #262626;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }

    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
    }

    .panel-meta {
        font-size: 0.85rem;
        color: var(--muted);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 45, 45, 0.35);
        background: rgba(255, 45, 45, 0.1);
        color: var(--accent);
        font-size: 0.75rem;
    }

    .input-label {
        font-size: 0.92rem;
        color: var(--muted);
        margin: 0.6rem 0 0.35rem;
    }

    .stTextArea textarea {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid #303030;
        border-radius: 12px;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px var(--glow);
    }

    .stTextInput input {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid #303030;
        border-radius: 10px;
    }

    .stButton > button {
        background: transparent;
        color: var(--text);
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    button[kind="primary"] {
        background: #f5f5f5;
        color: #0a0a0a !important;
        border: none;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    }

    button[kind="primary"]:hover {
        background: #ffffff;
        color: #0a0a0a !important;
    }

    button[kind="primary"] * {
        color: #0a0a0a !important;
        font-weight: 600;
    }

    .status-line {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
    }

    .status-dot.cache { background: #6dd5a5; }
    .status-dot.wikipedia { background: #6aa7ff; }

    .answer-placeholder {
        color: var(--muted);
        font-size: 0.95rem;
        padding: 0.8rem 0;
    }

    div[data-testid="stNotification"] > div {
        border-radius: 12px;
    }

    div[data-testid="stAlert"] > div {
        border-radius: 12px;
    }

    .side-card {
        background: #111111;
        border: 1px solid #242424;
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.9rem;
    }

    .side-title {
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .side-list {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 0;
        padding-left: 1.1rem;
    }

    .side-list li {
        margin-bottom: 0.35rem;
    }

    .side-meta {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.35rem;
    }

    .sidebar-brand {
        font-family: var(--font-display);
        font-size: 1.2rem;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }

    .nav-section {
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
        color: var(--muted);
        margin: 1rem 0 0.4rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        color: var(--text);
        border: 1px solid transparent;
        background: transparent;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }

    .nav-item .material-icons {
        font-size: 1.05rem;
        color: var(--muted);
    }

    .nav-item.active {
        border-color: #2a2a2a;
        background: rgba(255, 255, 255, 0.04);
    }

    .nav-item:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    section[data-testid="stSidebar"] > div:first-child {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .account-card {
        margin-top: auto;
        background: #101010;
        border: 1px solid #242424;
        border-radius: 14px;
        padding: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }

    .account-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        color: var(--text);
        font-size: 0.9rem;
    }

    .account-pill {
        font-size: 0.72rem;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: var(--muted);
    }

    .chat-anchor,
    .composer-anchor {
        height: 0;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlock"]:has(.chat-anchor) {
        padding-bottom: 5.5rem;
        min-height: 220px;
    }

    .msg {
        display: flex;
        flex-direction: column;
        margin: 0.75rem 0;
    }

    .msg.user {
        align-items: flex-end;
    }

    .msg.assistant {
        align-items: flex-start;
    }

    .msg .bubble {
        background: #111111;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        max-width: 780px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    }

    .msg.user .bubble {
        background: rgba(255, 255, 255, 0.06);
        border-color: #303030;
    }

    .msg-meta {
        font-size: 0.75rem;
        color: var(--muted);
        margin: 0 0 0.35rem;
    }

    .msg-meta.cache { color: #6dd5a5; }
    .msg-meta.wikipedia { color: #6aa7ff; }

    div[data-testid="stVerticalBlock"]:has(.composer-anchor) {
        position: sticky;
        bottom: 1.2rem;
        background: rgba(12, 12, 12, 0.92);
        border: 1px solid #2a2a2a;
        border-radius: 18px;
        padding: 0.9rem 1rem 0.6rem;
        box-shadow: 0 22px 50px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(16px);
        z-index: 10;
    }

    .composer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    .composer-title {
        font-weight: 600;
        font-size: 1rem;
    }

    .composer-meta {
        color: var(--muted);
        font-size: 0.85rem;
    }

    .pricing-shell {
        background: #111111;
        border: 1px solid #262626;
        border-radius: 18px;
        padding: 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    }

    .pricing-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .pricing-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
    }

    .plan-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.2rem;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .plan-card.featured {
        background: linear-gradient(140deg, #242424 0%, #1a1a1a 40%, #151515 100%);
        border-color: #3a3a3a;
    }

    .plan-price {
        font-size: 2rem;
        font-weight: 600;
    }

    .plan-cta {
        background: #f5f5f5;
        color: #0a0a0a;
        border-radius: 999px;
        padding: 0.55rem 1rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .plan-list {
        list-style: none;
        padding: 0;
        margin: 0;
        color: var(--muted);
        font-size: 0.88rem;
    }

    .plan-list li {
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    </style>
    """
).substitute(MATRICS_CSS=matrics_css, DISPLAY_FONT=display_font)

st.markdown(css, unsafe_allow_html=True)

api_label = "Responses API" if using_responses_api() else "Chat Completions"
deployment_label = AZURE_OPENAI_DEPLOYMENT or "Not set"

with st.sidebar:
    st.markdown('<div class="sidebar-brand">WonderTalk</div>', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.clear_prompt = True
        st.session_state.pending_prompt = ""
        st.session_state.last_answer = ""
        st.session_state.last_status = ""
        st.session_state.last_source = ""
        st.session_state.show_pricing = False
        st.rerun()

    st.text_input("Search chats", key="chat_search", placeholder="Search chats", label_visibility="collapsed")

    st.markdown(
        """
        <div class="nav-section">Create</div>
        <div class="nav-item active"><span class="material-icons">chat_bubble_outline</span>Chat</div>
        <div class="nav-item"><span class="material-icons">image</span>Images</div>
        <div class="nav-item"><span class="material-icons">apps</span>Apps</div>
        <div class="nav-item"><span class="material-icons">travel_explore</span>Deep research</div>
        <div class="nav-item"><span class="material-icons">code</span>Codex</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.recent_chats:
        recent_items = []
        for title in st.session_state.recent_chats[:6]:
            safe_title = html.escape(title)
            recent_items.append(
                f'<div class="nav-item"><span class="material-icons">history</span>{safe_title}</div>'
            )
        st.markdown('<div class="nav-section">Recent</div>' + "".join(recent_items), unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="side-card">
            <div class="side-title">Status</div>
            <div class="pill">{api_label}</div>
            <div class="side-meta">Deployment: {deployment_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not azure_config_ready():
        st.warning(
            "Azure OpenAI is not configured. Update .env with "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT. "
            "Use AZURE_OPENAI_ENDPOINT for Chat Completions or "
            "AZURE_OPENAI_RESPONSES_URL for Responses API."
        )

    if st.button("Test Azure Connection", use_container_width=True):
        if not azure_config_ready():
            st.error("Azure OpenAI is not configured. Update .env and try again.")
        else:
            with st.spinner("Testing Azure OpenAI..."):
                try:
                    _ = call_azure_openai("Reply with OK.")
                    st.success("Azure OpenAI connection is working.")
                except Exception as exc:
                    log_exception("Test Azure Connection failed", exc)
                    st.error(f"Azure OpenAI test failed: {exc}")

    if st.button("Upgrade plan", use_container_width=True):
        st.session_state.show_pricing = True

    st.markdown(
        """
        <div class="account-card">
            <div class="account-row">
                <span class="material-icons">account_circle</span>
                <div>Guest</div>
                <span class="account-pill">Free</span>
            </div>
            <div class="account-row"><span class="material-icons">upgrade</span>Upgrade plan</div>
            <div class="account-row"><span class="material-icons">tune</span>Personalization</div>
            <div class="account-row"><span class="material-icons">settings</span>Settings</div>
            <div class="account-row"><span class="material-icons">help_outline</span>Help</div>
            <div class="account-row"><span class="material-icons">logout</span>Log out</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

main = st.container()
with main:
    if st.session_state.show_pricing:
        st.markdown('<div class="pricing-title">Upgrade your plan</div>', unsafe_allow_html=True)
        plan_view = st.radio(
            "Plan type",
            ["Personal", "Business"],
            horizontal=True,
            key="plan_view",
            label_visibility="collapsed",
        )

        if plan_view == "Business":
            pricing_html = """
            <div class="pricing-shell">
                <div class="pricing-grid">
                    <div class="plan-card">
                        <div class="plan-title">Team</div>
                        <div class="plan-price">$25</div>
                        <div class="plan-sub">per seat / month</div>
                        <div class="plan-cta">Start Team</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>Shared workspaces</li>
                            <li><span class="material-icons">check</span>Team memory</li>
                            <li><span class="material-icons">check</span>File uploads</li>
                            <li><span class="material-icons">check</span>Admin controls</li>
                        </ul>
                    </div>
                    <div class="plan-card featured">
                        <div class="plan-title">Business</div>
                        <div class="plan-price">$60</div>
                        <div class="plan-sub">per seat / month</div>
                        <div class="plan-cta">Add workspace</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>SSO + MFA</li>
                            <li><span class="material-icons">check</span>Priority support</li>
                            <li><span class="material-icons">check</span>Private data controls</li>
                            <li><span class="material-icons">check</span>Advanced analytics</li>
                        </ul>
                    </div>
                    <div class="plan-card">
                        <div class="plan-title">Enterprise</div>
                        <div class="plan-price">Custom</div>
                        <div class="plan-sub">contact sales</div>
                        <div class="plan-cta">Talk to sales</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>Dedicated SLAs</li>
                            <li><span class="material-icons">check</span>On-prem options</li>
                            <li><span class="material-icons">check</span>Compliance reviews</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
        else:
            pricing_html = """
            <div class="pricing-shell">
                <div class="pricing-grid">
                    <div class="plan-card">
                        <div class="plan-title">Free</div>
                        <div class="plan-price">$0</div>
                        <div class="plan-sub">forever</div>
                        <div class="plan-cta">Current plan</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>Everyday chat</li>
                            <li><span class="material-icons">check</span>Kid-safe answers</li>
                            <li><span class="material-icons">check</span>Basic memory</li>
                        </ul>
                    </div>
                    <div class="plan-card">
                        <div class="plan-title">Plus</div>
                        <div class="plan-price">$12</div>
                        <div class="plan-sub">per month</div>
                        <div class="plan-cta">Upgrade to Plus</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>More messages</li>
                            <li><span class="material-icons">check</span>Image tools</li>
                            <li><span class="material-icons">check</span>Longer memory</li>
                        </ul>
                    </div>
                    <div class="plan-card featured">
                        <div class="plan-title">Pro</div>
                        <div class="plan-price">$30</div>
                        <div class="plan-sub">per month</div>
                        <div class="plan-cta">Go Pro</div>
                        <ul class="plan-list">
                            <li><span class="material-icons">check</span>Priority responses</li>
                            <li><span class="material-icons">check</span>Advanced models</li>
                            <li><span class="material-icons">check</span>Workspace projects</li>
                        </ul>
                    </div>
                </div>
            </div>
            """

        st.markdown(pricing_html, unsafe_allow_html=True)

        if st.button("Close pricing", use_container_width=True):
            st.session_state.show_pricing = False

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">What are you working on?</div>
                <div class="hero-subtitle">Ask anything. WonderTalk keeps context, remembers your style, and stays kid-friendly by default.</div>
                <div class="hero-accent"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestion_cols = st.columns(3)
        suggestions = [
            "Tell me a short story about space.",
            "Explain photosynthesis simply.",
            "Help me make a study plan.",
        ]
        for col, suggestion in zip(suggestion_cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state.pending_prompt = suggestion

    chat_block = st.container()
    with chat_block:
        st.markdown('<div class="chat-anchor"></div>', unsafe_allow_html=True)
        for message in st.session_state.messages:
            render_message(message)

    composer = st.container()
    with composer:
        st.markdown('<div class="composer-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="composer-header">
                <div class="composer-title">Ask WonderTalk</div>
                <div class="composer-meta">{api_label} | {deployment_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode_col, spacer_col = st.columns([2, 6])
        with mode_col:
            st.selectbox(
                "Mode",
                ["Kid-friendly", "Homework", "Story", "Explore"],
                key="mode",
                label_visibility="collapsed",
            )
        with spacer_col:
            st.markdown("", unsafe_allow_html=True)

        mic_ready = mic_available()
        if st.session_state.clear_prompt:
            st.session_state.prompt = ""
            st.session_state.clear_prompt = False
        if st.session_state.pending_prompt:
            st.session_state.prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = ""

        input_col, mic_col, send_col = st.columns([7, 1.5, 1.5])
        with input_col:
            st.text_area(
                "Your question",
                key="prompt",
                height=90,
                label_visibility="collapsed",
                placeholder="Ask anything...",
            )
        with mic_col:
            if st.button("Use Mic", use_container_width=True, disabled=not mic_ready):
                recognize_speech()
        with send_col:
            send_clicked = st.button("Send", type="primary", use_container_width=True)

        if not mic_ready:
            st.caption("Microphone disabled (PyAudio not installed).")

        if send_clicked:
            prompt = st.session_state.prompt.strip()
            if not prompt:
                st.warning("Please type or speak your question.")
            else:
                history_context = build_context(st.session_state.messages)
                add_message("user", prompt)
                st.session_state.clear_prompt = True

                recent_title = prompt[:48] + ("..." if len(prompt) > 48 else "")
                if recent_title in st.session_state.recent_chats:
                    st.session_state.recent_chats.remove(recent_title)
                st.session_state.recent_chats.insert(0, recent_title)
                st.session_state.recent_chats = st.session_state.recent_chats[:8]

                with st.spinner("Thinking..."):
                    cache_key = f"{st.session_state.mode}::{prompt}"
                    results = collection.query(query_texts=[cache_key], n_results=1)
                    documents = results.get("documents", [[]])[0]
                    distances = results.get("distances", [[]])[0]

                    if documents and distances and distances[0] < 0.2:
                        answer = results.get("metadatas", [[{}]])[0][0].get("answer", "")
                        status = "Cache hit"
                        source = "cache"
                    else:
                        formatted = build_prompt(prompt, st.session_state.mode, history_context)
                        try:
                            if azure_config_ready():
                                answer = call_azure_openai(formatted)
                            else:
                                raise RuntimeError("Azure OpenAI not configured")
                            if len(answer.split()) < 5:
                                raise ValueError("Answer too short")
                            status = "Generated by Azure OpenAI"
                            source = "azure"
                        except Exception:
                            logger.exception("Primary response failed; falling back to Wikipedia")
                            answer = get_wikipedia_summary(prompt)
                            status = "Wikipedia fallback"
                            source = "wikipedia"

                        collection.add(
                            documents=[cache_key],
                            metadatas=[{"answer": answer}],
                            ids=[str(uuid.uuid4())],
                        )

                    st.session_state.last_answer = answer
                    st.session_state.last_status = status
                    st.session_state.last_source = source
                    add_message("assistant", answer, status=status, source=source)

                    if st.session_state.voice_active:
                        speak_text(answer)

                st.rerun()

        voice_col1, voice_col2 = st.columns(2)
        with voice_col1:
            if st.button("Stop Voice", use_container_width=True):
                stop_speech()
        with voice_col2:
            if st.button("Start Voice", use_container_width=True):
                if st.session_state.last_answer.strip():
                    start_speech(st.session_state.last_answer)
                else:
                    st.warning("No answer to speak yet.")
