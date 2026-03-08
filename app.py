import streamlit as st
import streamlit.components.v1 as components
import pyttsx3
import wikipedia
import uuid
import hashlib
import os
import base64
import html
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
import tempfile
from string import Template
import threading
import logging
from logging.handlers import RotatingFileHandler
import speech_recognition as sr
from dotenv import load_dotenv
import chromadb
import requests

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception:
    speechsdk = None

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

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
SOURCES_DIR = os.path.join(APP_DIR, "sources")
os.makedirs(SOURCES_DIR, exist_ok=True)

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
        return "", "Jost"
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
        return "", "Jost"


def _normalize_endpoint(url: str) -> str:
    if not url:
        return ""
    cleaned = url.strip().rstrip("/")
    suffix = "/openai/v1"
    if cleaned.endswith(suffix):
        cleaned = cleaned[: -len(suffix)]
    return cleaned


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


AZURE_OPENAI_ENDPOINT = _normalize_endpoint(_get_secret("AZURE_OPENAI_ENDPOINT", ""))
AZURE_OPENAI_BASE_URL = _get_secret("AZURE_OPENAI_BASE_URL", "").strip()
if not AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_BASE_URL:
    AZURE_OPENAI_ENDPOINT = _normalize_endpoint(AZURE_OPENAI_BASE_URL)

AZURE_OPENAI_API_KEY = _get_secret("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = _get_secret("AZURE_OPENAI_API_VERSION", "2024-06-01").strip()

AZURE_OPENAI_DEPLOYMENT = _get_secret("AZURE_OPENAI_DEPLOYMENT", "").strip()
AZURE_OPENAI_DEPLOYMENT_NAME_MAP = _get_secret("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "").strip()
AZURE_OPENAI_DEPLOYMENT_URL_MAP = _get_secret("AZURE_OPENAI_DEPLOYMENT_URL_MAP", "").strip()

def _parse_deployment_map(raw: str, fallback: str) -> dict:
    mapping = {}
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            if "=" in part:
                label, dep = part.split("=", 1)
                label = label.strip()
                dep = dep.strip()
                if label:
                    mapping[label] = dep or label
            else:
                mapping[part] = part
    if not mapping and fallback:
        mapping[fallback] = fallback
    return mapping


def _parse_url_map(raw: str) -> dict:
    mapping = {}
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            label, url = part.split("=", 1)
            label = label.strip()
            url = url.strip()
            if label and url:
                mapping[label] = url
    return mapping


def _pick_default_deployment(mapping: dict, fallback: str) -> str:
    for label, dep in mapping.items():
        if "gpt-5.2-chat" in (label or "") or "gpt-5.2-chat" in (dep or ""):
            return dep
    if mapping:
        return next(iter(mapping.values()))
    return fallback


DEPLOYMENT_MAP = _parse_deployment_map(AZURE_OPENAI_DEPLOYMENT_NAME_MAP, AZURE_OPENAI_DEPLOYMENT)
DEPLOYMENT_URL_MAP = _parse_url_map(AZURE_OPENAI_DEPLOYMENT_URL_MAP)
AZURE_OPENAI_DEPLOYMENT = _pick_default_deployment(DEPLOYMENT_MAP, AZURE_OPENAI_DEPLOYMENT)


def _deployment_label_for(deployment: str) -> str:
    for label, dep in DEPLOYMENT_MAP.items():
        if dep == deployment:
            return label
    return deployment

AZURE_OPENAI_API_MODE = _get_secret("AZURE_OPENAI_API_MODE", "").strip().lower()
AZURE_OPENAI_RESPONSES_URL = _get_secret("AZURE_OPENAI_RESPONSES_URL", "").strip()
AZURE_OPENAI_TEMPERATURE = _get_secret("AZURE_OPENAI_TEMPERATURE", "0.7").strip()
AZURE_OPENAI_MAX_TOKENS = _get_secret("AZURE_OPENAI_MAX_TOKENS", "400").strip()
AZURE_OPENAI_MAX_OUTPUT_TOKENS = _get_secret("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "400").strip()
AZURE_OPENAI_RESPONSES_TEMPERATURE = _get_secret("AZURE_OPENAI_RESPONSES_TEMPERATURE", "").strip()

AZURE_SPEECH_KEY = _get_secret("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = _get_secret("AZURE_SPEECH_REGION", "").strip()
AZURE_SPEECH_VOICE = _get_secret("AZURE_SPEECH_VOICE", "").strip()

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


def azure_speech_ready() -> bool:
    return bool(speechsdk and AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)


def _azure_speech_config() -> "speechsdk.SpeechConfig":
    config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    if AZURE_SPEECH_VOICE:
        config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE
    return config


def get_active_deployment() -> str:
    selected = ""
    try:
        selected = st.session_state.get("selected_deployment", "")
    except Exception:
        selected = ""
    return selected or AZURE_OPENAI_DEPLOYMENT


def get_active_deployment_url() -> str:
    label = ""
    try:
        label = st.session_state.get("selected_deployment_label", "")
    except Exception:
        label = ""
    deployment = get_active_deployment()
    return (
        DEPLOYMENT_URL_MAP.get(label)
        or DEPLOYMENT_URL_MAP.get(deployment)
        or ""
    )


def _infer_mode_from_url(url: str) -> str:
    lower = (url or "").lower()
    if "responses" in lower:
        return "responses"
    if "chat/completions" in lower:
        return "chat"
    return ""


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
    deployment = get_active_deployment()
    url_override = get_active_deployment_url()
    if not AZURE_OPENAI_API_KEY or not deployment:
        return False
    if AZURE_OPENAI_API_KEY.lower() in _PLACEHOLDER_KEYS:
        return False
    if "<" in deployment:
        return False
    if url_override:
        if "<" in url_override:
            return False
        return True
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


def call_azure_chat(prompt_text: str, url_override: str = "") -> str:
    deployment = get_active_deployment()
    logger.info("Azure chat request start (deployment=%s)", deployment)
    if url_override:
        url = url_override
    else:
        url = (
            f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}"
            f"/openai/deployments/{deployment}/chat/completions"
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
    if url_override and "/deployments/" not in url_override:
        payload["model"] = deployment
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    _raise_for_error(resp)
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_azure_responses(prompt_text: str, url_override: str = "") -> str:
    deployment = get_active_deployment()
    logger.info("Azure responses request start (model=%s)", deployment)
    url = url_override or _build_responses_url()
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    payload = {
        "model": deployment,
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
    url_override = get_active_deployment_url()
    if url_override:
        mode = _infer_mode_from_url(url_override) or ("responses" if using_responses_api() else "chat")
        if mode == "responses":
            return call_azure_responses(prompt_text, url_override)
        return call_azure_chat(prompt_text, url_override)
    if using_responses_api():
        return call_azure_responses(prompt_text)
    return call_azure_chat(prompt_text)

# ChromaDB setup (persistent cache with safe fallback for Streamlit Cloud)
class _InMemoryCacheCollection:
    def __init__(self):
        self._store = {}

    def query(self, query_texts=None, n_results=1):
        key = (query_texts or [""])[0]
        if key in self._store:
            return {
                "documents": [[key]],
                "distances": [[0.0]],
                "metadatas": [[{"answer": self._store[key]}]],
            }
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    def add(self, documents=None, metadatas=None, ids=None):
        for doc, meta in zip(documents or [], metadatas or []):
            if isinstance(meta, dict) and "answer" in meta:
                self._store[doc] = meta["answer"]


class _InMemorySourceCollection:
    def __init__(self):
        self._entries = []

    def add(self, documents=None, metadatas=None, ids=None):
        for doc, meta, item_id in zip(documents or [], metadatas or [], ids or []):
            self._entries.append({"id": item_id, "doc": doc, "meta": meta or {}})

    def delete(self, ids=None):
        if not ids:
            return
        id_set = set(ids)
        self._entries = [e for e in self._entries if e["id"] not in id_set]

    def query(self, query_texts=None, n_results=3):
        query = (query_texts or [""])[0].lower()
        tokens = set(re.findall(r"\w+", query))
        scored = []
        for entry in self._entries:
            text = str(entry["doc"]).lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[:n_results]]
        return {
            "documents": [[e["doc"] for e in top]],
            "metadatas": [[e["meta"] for e in top]],
        }


try:
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR, tenant="default_tenant", database="default_database"
    )
    collection = chroma_client.get_or_create_collection(name="prompt_store")
    source_collection = chroma_client.get_or_create_collection(name="source_store")
except Exception as exc:
    log_exception("ChromaDB init failed; using in-memory cache", exc)
    collection = _InMemoryCacheCollection()
    source_collection = _InMemorySourceCollection()

# TTS setup
TTS_AVAILABLE = True
engine = None
voices = []
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
except Exception as exc:
    log_exception("Local TTS init failed", exc)

TTS_AVAILABLE = bool(engine) or azure_speech_ready()


def _pick_friendly_voice(voice_list):
    preferred = [
        "zira",
        "susan",
        "hazel",
        "aria",
        "jenny",
        "emma",
        "mia",
        "samantha",
        "female",
    ]
    for voice in voice_list:
        name = (getattr(voice, "name", "") or "").lower()
        vid = (getattr(voice, "id", "") or "").lower()
        if any(tag in name or tag in vid for tag in preferred):
            return voice.id
    for voice in voice_list:
        gender = str(getattr(voice, "gender", "")).lower()
        if "female" in gender:
            return voice.id
    return voice_list[0].id if voice_list else None


voice_id = _pick_friendly_voice(voices or [])
if voice_id:
    engine.setProperty("voice", voice_id)
speech_lock = threading.Lock()

# Session state init
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

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

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "selected_deployment_label" not in st.session_state:
    st.session_state.selected_deployment_label = _deployment_label_for(AZURE_OPENAI_DEPLOYMENT) if AZURE_OPENAI_DEPLOYMENT else ""

if "selected_deployment" not in st.session_state:
    st.session_state.selected_deployment = (
        DEPLOYMENT_MAP.get(st.session_state.selected_deployment_label, AZURE_OPENAI_DEPLOYMENT)
    )

if "source_mode" not in st.session_state:
    st.session_state.source_mode = "Curated"

if "source_urls_raw" not in st.session_state:
    st.session_state.source_urls_raw = ""

if "source_indexed_ids" not in st.session_state:
    st.session_state.source_indexed_ids = []

if "source_chunk_count" not in st.session_state:
    st.session_state.source_chunk_count = 0

if "sources_indexed" not in st.session_state:
    st.session_state.sources_indexed = False

if "source_fingerprint" not in st.session_state:
    st.session_state.source_fingerprint = "none"

if "clear_prompt" not in st.session_state:
    st.session_state.clear_prompt = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_tts_audio" not in st.session_state:
    st.session_state.last_tts_audio = b""
if "auto_play_audio" not in st.session_state:
    st.session_state.auto_play_audio = False
if "stop_audio" not in st.session_state:
    st.session_state.stop_audio = False
if "last_status" not in st.session_state:
    st.session_state.last_status = ""
if "last_source" not in st.session_state:
    st.session_state.last_source = ""

# Stop flag
stop_speech_flag = threading.Event()
speech_thread = None


def speak_text(text):
    global speech_thread
    if not TTS_AVAILABLE or engine is None:
        return
    stop_speech_flag.clear()
    speech_text = sanitize_tts_text(text)

    def _speak():
        try:
            with speech_lock:
                if stop_speech_flag.is_set():
                    return
                engine.say(speech_text)
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
    if not TTS_AVAILABLE or engine is None:
        return
    stop_speech_flag.set()
    try:
        engine.stop()
    except Exception as exc:
        log_exception("TTS stop failed", exc)
    st.session_state.voice_active = False


def start_speech(text):
    if azure_speech_ready():
        try:
            audio_bytes = synthesize_speech_azure(text)
            if audio_bytes:
                st.session_state.last_tts_audio = audio_bytes
                st.session_state.auto_play_audio = True
        except Exception as exc:
            log_exception("Azure TTS failed", exc)
    else:
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


def _normalize_urls(raw: str) -> list:
    urls = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if not re.match(r"^https?://", line):
            line = f"https://{line}"
        urls.append(line)
    return urls


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def extract_text_from_file(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed")
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return Path(path).read_text(errors="ignore")


def _wikipedia_api_text(url: str, headers: dict) -> str:
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        return ""
    parts = parsed.path.split("/wiki/")
    if len(parts) < 2 or not parts[1]:
        return ""
    title = unquote(parts[1])
    lang = parsed.netloc.split(".")[0] or "en"
    api_base = f"https://{lang}.wikipedia.org/api/rest_v1/page"
    for endpoint in ("plain", "summary"):
        api_url = f"{api_base}/{endpoint}/{title}"
        resp = requests.get(api_url, headers=headers, timeout=20)
        if resp.ok:
            if endpoint == "plain":
                return resp.text
            data = resp.json()
            return data.get("extract", "")
    return ""


def fetch_url_text(url: str) -> str:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is not installed")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WonderTalk/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = requests.get(url, timeout=20, headers=headers)
    if resp.status_code == 403:
        wiki_text = _wikipedia_api_text(url, headers)
        if wiki_text:
            return wiki_text
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return text


def synthesize_speech_azure(text: str) -> bytes:
    if not azure_speech_ready():
        return b""
    speech_config = _azure_speech_config()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        tmp_path = handle.name
    audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        if result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            raise RuntimeError(details.error_details or "Speech synthesis canceled")
        raise RuntimeError("Speech synthesis failed")
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


def transcribe_audio_azure(audio_bytes: bytes) -> str:
    if not azure_speech_ready():
        return ""
    speech_config = _azure_speech_config()
    speech_config.speech_recognition_language = "en-US"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        handle.write(audio_bytes)
        tmp_path = handle.name
    audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once_async().get()
    Path(tmp_path).unlink(missing_ok=True)
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    if result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails.from_result(result)
        raise RuntimeError(details.error_details or "Speech recognition canceled")
    return ""


def build_source_fingerprint(urls: list, file_labels: list) -> str:
    payload = "||".join(sorted(urls) + sorted(file_labels))
    if not payload:
        return "none"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def clear_indexed_sources():
    ids = st.session_state.source_indexed_ids or []
    if ids:
        try:
            source_collection.delete(ids=ids)
        except Exception as exc:
            log_exception("Source index delete failed", exc)
    st.session_state.source_indexed_ids = []
    st.session_state.source_chunk_count = 0
    st.session_state.sources_indexed = False
    st.session_state.source_fingerprint = "none"


def index_sources(uploaded_files, urls):
    docs = []
    metas = []
    ids = []
    labels = []
    errors = []

    for uploaded in uploaded_files or []:
        suffix = Path(uploaded.name).suffix
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        dest_path = os.path.join(SOURCES_DIR, safe_name)
        with open(dest_path, "wb") as handle:
            handle.write(uploaded.getbuffer())
        labels.append(uploaded.name)
        try:
            text = extract_text_from_file(dest_path)
            for chunk in chunk_text(text):
                ids.append(str(uuid.uuid4()))
                docs.append(chunk)
                metas.append({"label": uploaded.name})
        except Exception as exc:
            log_exception(f"Source file failed: {uploaded.name}", exc)
            errors.append(f"{uploaded.name}: {exc}")

    for url in urls or []:
        labels.append(url)
        try:
            text = fetch_url_text(url)
            for chunk in chunk_text(text):
                ids.append(str(uuid.uuid4()))
                docs.append(chunk)
                metas.append({"label": url})
        except Exception as exc:
            log_exception(f"Source url failed: {url}", exc)
            errors.append(f"{url}: {exc}")

    if docs:
        source_collection.add(documents=docs, metadatas=metas, ids=ids)

    fingerprint = build_source_fingerprint(urls or [], labels)
    return ids, len(docs), fingerprint, errors


def build_source_context(query: str, max_chars: int = 2400) -> str:
    if not st.session_state.sources_indexed:
        return ""
    try:
        results = source_collection.query(query_texts=[query], n_results=3)
    except Exception as exc:
        log_exception("Source retrieval failed", exc)
        return ""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return ""
    lines = []
    for doc, meta in zip(documents, metadatas):
        label = (meta or {}).get("label", "source")
        lines.append(f"[{label}] {doc}")
    if st.session_state.source_mode == "Curated":
        header = "Use only the sources below. If the answer is not present, say you don't know."
    else:
        header = "Prefer the sources below. If needed, you may use general knowledge."
    combined = header + "\n" + "\n\n".join(lines)
    return combined[:max_chars]


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


def sanitize_tts_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[*_`>#\\[\\]()-]", " ", text)
    cleaned = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", " ", cleaned
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    deduped = []
    for word in words:
        if deduped and deduped[-1].lower() == word.lower():
            continue
        deduped.append(word)
    return " ".join(deduped)


# UI
matrics_css, display_font = _load_matrics_font_css()
theme_choice = st.session_state.theme
themes = {
    "Light": {
        "bg": "#f4f1eb",
        "panel": "#ffffff",
        "panel_2": "#fbf9f5",
        "text": "#1b1b1b",
        "muted": "#6c6c6c",
        "accent": "#111111",
        "accent_2": "#2c2c2c",
        "accent_bg": "#111111",
        "accent_text": "#f5f3ef",
        "border": "#e1dbd2",
        "border_strong": "#d6d0c7",
        "glow": "rgba(0, 0, 0, 0.12)",
        "shadow": "rgba(0, 0, 0, 0.08)",
        "shadow_strong": "rgba(0, 0, 0, 0.12)",
        "sidebar_bg": "#f7f5f1",
        "dot": "rgba(0,0,0,0.06)",
        "pill_bg": "#f4f1eb",
        "pill_border": "#dcd6cc",
    },
    "Dark": {
        "bg": "#0b0b0b",
        "panel": "#151515",
        "panel_2": "#1d1d1d",
        "text": "#f5f5f5",
        "muted": "#a8a8a8",
        "accent": "#f5f5f5",
        "accent_2": "#dddddd",
        "accent_bg": "#f5f5f5",
        "accent_text": "#0b0b0b",
        "border": "#2a2a2a",
        "border_strong": "#3a3a3a",
        "glow": "rgba(255, 255, 255, 0.12)",
        "shadow": "rgba(0, 0, 0, 0.45)",
        "shadow_strong": "rgba(0, 0, 0, 0.6)",
        "sidebar_bg": "#111111",
        "dot": "rgba(255,255,255,0.06)",
        "pill_bg": "#1b1b1b",
        "pill_border": "#2a2a2a",
    },
}
theme = themes.get(theme_choice, themes["Light"])
theme_vars = "\n        ".join(
    [
        f"--bg: {theme['bg']};",
        f"--panel: {theme['panel']};",
        f"--panel-2: {theme['panel_2']};",
        f"--text: {theme['text']};",
        f"--muted: {theme['muted']};",
        f"--accent: {theme['accent']};",
        f"--accent-2: {theme['accent_2']};",
        f"--accent-bg: {theme['accent_bg']};",
        f"--accent-text: {theme['accent_text']};",
        f"--border: {theme['border']};",
        f"--border-strong: {theme['border_strong']};",
        f"--glow: {theme['glow']};",
        f"--shadow: {theme['shadow']};",
        f"--shadow-strong: {theme['shadow_strong']};",
        f"--sidebar-bg: {theme['sidebar_bg']};",
        f"--dot: {theme['dot']};",
        f"--pill-bg: {theme['pill_bg']};",
        f"--pill-border: {theme['pill_border']};",
    ]
)

css = Template(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600&family=Sora:wght@300;400;500;600&display=swap');
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
        $THEME_VARS
        --font-display: '$DISPLAY_FONT', 'Jost', sans-serif;
        --font-body: 'Sora', sans-serif;
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

    button[data-testid="stExpandSidebarButton"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.2rem;
        display: inline-flex !important;
        position: fixed;
        top: 12px;
        left: 12px;
        z-index: 2000;
    }

    .stApp {
        background: var(--bg);
        background-image: radial-gradient(circle at 1px 1px, var(--dot) 1px, transparent 0);
        background-size: 22px 22px;
    }

    header[data-testid="stHeader"] {
        background: var(--bg);
        border-bottom: none;
        box-shadow: none;
    }

    header[data-testid="stHeader"] * {
        color: var(--text) !important;
    }

    div[data-testid="stAppViewContainer"] > header {
        background: var(--bg);
    }

    div[data-testid="stAppViewContainer"] {
        padding-top: 0;
    }

    div[data-testid="stToolbar"] {
        display: flex;
        background: transparent;
        box-shadow: none;
    }

    div[data-testid="stToolbar"] a,
    div[data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]),
    div[data-testid="stToolbar"] svg {
        display: none !important;
    }

    div[data-testid="stToolbar"] div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stToolbar"] div[data-testid="stSidebarCollapsedControl"] button,
    div[data-testid="stToolbar"] button[data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
    }

    div[data-testid="stToolbar"] div[data-testid="stSidebarCollapseButton"] svg,
    div[data-testid="stToolbar"] div[data-testid="stSidebarCollapsedControl"] svg,
    div[data-testid="stToolbar"] button[data-testid="stExpandSidebarButton"] svg {
        display: inline-block !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    div[data-testid="stDecoration"] {
        display: none;
    }

    .main .block-container {
        padding-top: 1.6rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        text-align: center;
        margin: 0 0 0.8rem 0;
    }

    .hero-cta .stButton > button {
        border-radius: 999px;
        padding: 0.65rem 1.1rem;
        font-size: 0.95rem;
        white-space: nowrap;
    }

    .hero-cta .stButton > button div {
        display: inline-block;
        white-space: nowrap;
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
        background: transparent;
        border: none;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: none;
    }

    .panel-anchor {
        height: 0;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlock"]:has(.panel-anchor) {
        background: transparent;
        border: none;
        border-radius: 16px;
        padding: 0;
        box-shadow: none;
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
        border: 1px solid var(--pill-border);
        background: var(--pill-bg);
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
        border: 1px solid var(--border-strong);
        border-radius: 12px;
        caret-color: var(--text);
    }

    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px var(--glow);
    }

    .stTextInput input {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        caret-color: var(--text);
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #9a948a;
    }

    div[data-testid="stSelectbox"] > div {
        background: var(--panel-2);
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        color: var(--text);
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
        background: var(--panel-2);
        border: 1px solid var(--border-strong);
        border-radius: 12px;
        min-height: 40px;
        color: var(--text);
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {
        fill: var(--text);
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [role="combobox"] {
        background: var(--panel-2) !important;
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] span,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div {
        color: var(--text) !important;
    }

    .mic-recorder,
    div[data-testid="stMicrophoneRecorder"],
    div[data-testid="stMicrophoneRecorder"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stMicrophoneRecorder"] {
        width: 100% !important;
        display: flex;
        align-items: stretch;
    }

    div[data-testid="stMicrophoneRecorder"] iframe {
        width: 100% !important;
    }

    .mic-recorder button,
    div[data-testid="stMicrophoneRecorder"] button,
    div[data-testid="stMicrophoneRecorder"] .stButton > button {
        background: var(--accent-bg) !important;
        color: var(--accent-text) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.1rem !important;
        font-size: 0.95rem !important;
        box-shadow: 0 12px 28px var(--shadow-strong) !important;
        font-weight: 600 !important;
        width: 100% !important;
        min-height: 44px !important;
    }

    .mic-recorder button:hover,
    div[data-testid="stMicrophoneRecorder"] button:hover,
    div[data-testid="stMicrophoneRecorder"] .stButton > button:hover {
        filter: brightness(0.95) !important;
    }

    .stRadio [role="radiogroup"] label,
    .stRadio [role="radiogroup"] label span,
    .stRadio [role="radiogroup"] label div,
    div[data-testid="stRadio"] label {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    .stRadio input[type="radio"] {
        accent-color: var(--accent-bg);
    }

    @media (max-width: 768px) {
        .composer-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.35rem;
        }
        .composer-meta {
            font-size: 0.75rem;
        }
        div[data-testid="stAudio"] audio {
            width: 100%;
        }
        .stButton > button {
            padding: 0.5rem 0.9rem;
            font-size: 0.9rem;
        }
    }

    .stRadio [role="radiogroup"] {
        gap: 0.7rem;
    }

    .stRadio [role="radiogroup"] label {
        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        padding: 0.45rem 0.7rem;
        font-size: 0.9rem;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div {
        background: var(--panel-2);
        border: 1px solid var(--border-strong);
        border-radius: 14px;
        padding: 0.6rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
        color: var(--text);
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label {
        color: var(--muted) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
        background: transparent;
        color: var(--text);
        border: 1px solid var(--border-strong);
        border-radius: 12px;
        padding: 0.45rem 0.7rem;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover {
        background: var(--accent-bg);
        color: var(--accent-text);
        border-color: var(--accent-bg);
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover * {
        color: var(--accent-text) !important;
    }

    .stButton > button {
        background: transparent;
        color: var(--text);
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent-bg);
        color: var(--accent-text);
        background: var(--accent-bg);
    }

    .stButton > button:hover * {
        color: var(--accent-text) !important;
    }

    .stButton > button:disabled {
        color: var(--muted) !important;
        border-color: var(--border) !important;
        background: transparent !important;
        opacity: 1 !important;
    }

    .stButton > button:disabled * {
        color: var(--muted) !important;
    }

    button[kind="primary"] {
        background: var(--accent-bg);
        color: var(--accent-text) !important;
        border: none;
        box-shadow: 0 12px 28px var(--shadow-strong);
        border-radius: 10px !important;
        min-height: 44px;
        padding: 0.55rem 1.1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    button[kind="primary"]:hover {
        filter: brightness(0.95);
        color: var(--accent-text) !important;
    }

    button[kind="primary"] * {
        color: var(--accent-text) !important;
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
        background: var(--panel-2);
        border: 1px solid var(--border);
        color: var(--text);
    }

    div[data-testid="stAlert"] > div {
        border-radius: 12px;
        background: var(--panel-2);
        border: 1px solid var(--border);
        color: var(--text);
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: var(--text) !important;
    }

    .side-card {
        background: var(--panel);
        border: 1px solid var(--border);
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
        color: var(--text);
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
        border-color: var(--border);
        background: var(--panel-2);
    }

    section[data-testid="stMain"] {
        background: transparent;
    }

    .nav-item:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--panel-2);
    }

    section[data-testid="stSidebar"] > div:first-child {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .account-card {
        margin-top: auto;
        background: var(--panel);
        border: 1px solid var(--border);
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
        padding-bottom: 2.5rem;
        min-height: 0;
        background: transparent;
        border: none;
        box-shadow: none;
    }

    div[data-testid="stVerticalBlock"]:has(div[data-testid="stVerticalBlock"]:has(.chat-anchor)) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
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
        background: var(--panel);
        border: none;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        max-width: 780px;
        box-shadow: none;
    }

    .msg.user .bubble {
        background: var(--panel-2);
        border: none;
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
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
        backdrop-filter: none;
        z-index: 10;
    }

    div[data-testid="stVerticalBlock"]:has(div[data-testid="stVerticalBlock"]:has(.composer-anchor)) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
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
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 40px var(--shadow);
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
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .plan-card.featured {
        background: var(--panel-2);
        border-color: var(--border-strong);
    }

    .plan-price {
        font-size: 2rem;
        font-weight: 600;
    }

    .plan-cta {
        background: var(--accent-bg);
        color: var(--accent-text);
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
).substitute(MATRICS_CSS=matrics_css, DISPLAY_FONT=display_font, THEME_VARS=theme_vars)

st.markdown(css, unsafe_allow_html=True)

_active_url = get_active_deployment_url()
_mode_hint = _infer_mode_from_url(_active_url)
effective_mode = _mode_hint or ("responses" if using_responses_api() else "chat")
api_label = "Responses API" if effective_mode == "responses" else "Chat Completions"
active_deployment = get_active_deployment()
deployment_label = st.session_state.get("selected_deployment_label") or active_deployment or "Not set"
source_mode_label = "Curated" if st.session_state.source_mode == "Curated" else "Web MCP"
if st.session_state.sources_indexed:
    source_meta = f"Sources: {source_mode_label}"
else:
    source_meta = f"Sources: {source_mode_label} (none)"

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

    st.markdown('<div class="nav-section">Theme</div>', unsafe_allow_html=True)
    st.radio(
        "Theme",
        ["Light", "Dark"],
        key="theme",
        horizontal=True,
        label_visibility="collapsed",
    )

    deployment_labels = list(DEPLOYMENT_MAP.keys()) if DEPLOYMENT_MAP else []
    if deployment_labels:
        st.markdown('<div class="nav-section">Model</div>', unsafe_allow_html=True)
        selected_label = st.selectbox(
            "Model",
            deployment_labels,
            key="selected_deployment_label",
            label_visibility="collapsed",
        )
        st.session_state.selected_deployment = DEPLOYMENT_MAP.get(selected_label, selected_label)

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

    st.markdown('<div class="nav-section">Sources</div>', unsafe_allow_html=True)
    st.radio(
        "Sources",
        ["Curated", "Web MCP"],
        key="source_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_sources = st.file_uploader(
        "Upload textbooks or notes",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="source_files",
    )
    st.text_area(
        "Approved websites",
        key="source_urls_raw",
        placeholder="https://example.com\nhttps://school.edu/lesson",
        height=90,
    )

    src_col1, src_col2 = st.columns(2)
    with src_col1:
        if st.button("Index sources", use_container_width=True):
            urls = _normalize_urls(st.session_state.source_urls_raw)
            try:
                clear_indexed_sources()
                ids, count, fingerprint, errors = index_sources(uploaded_sources, urls)
                st.session_state.source_indexed_ids = ids
                st.session_state.source_chunk_count = count
                st.session_state.sources_indexed = count > 0
                st.session_state.source_fingerprint = fingerprint
                if count == 0:
                    st.warning("No source text found. Add files or URLs, then try again.")
                else:
                    st.success(f"Indexed {count} chunks from your sources.")
                if errors:
                    preview = "\n".join(f"- {err}" for err in errors[:3])
                    more = "" if len(errors) <= 3 else f"\n...and {len(errors) - 3} more."
                    st.warning(f"Some sources failed:\n{preview}{more}")
            except Exception as exc:
                log_exception("Index sources failed", exc)
                st.error(f"Indexing failed: {exc}")

    with src_col2:
        if st.button("Clear sources", use_container_width=True):
            clear_indexed_sources()
            st.success("Cleared indexed sources.")

    if st.session_state.source_mode == "Web MCP":
        st.caption("Web MCP fetches only the URLs you list (no open web search).")

    if st.session_state.sources_indexed:
        st.caption(f"Sources indexed: {st.session_state.source_chunk_count} chunks.")
    else:
        st.caption("No sources indexed yet.")

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

    if (not st.session_state.show_pricing) and (len(st.session_state.messages) == 0):
        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">What are you working on?</div>
                <div class="hero-subtitle">
                    Ask anything. WonderTalk keeps context, remembers your style, and stays
                    kid-friendly by default.
                </div>
                <div class="hero-accent"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestions = [
            "Tell me a short story about space.",
            "Explain photosynthesis in simple terms.",
            "Help me make a study plan.",
        ]
        suggestion_cols = st.columns(3)
        for idx, (col, text) in enumerate(zip(suggestion_cols, suggestions)):
            with col:
                st.markdown('<div class="hero-cta">', unsafe_allow_html=True)
                if st.button(text, key=f"hero_suggest_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = text
                st.markdown("</div>", unsafe_allow_html=True)

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
                <div class="composer-meta">{api_label} | {deployment_label} | {source_meta}</div>
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

        browser_mic_ready = azure_speech_ready() and mic_recorder is not None
        mic_ready = mic_available() or browser_mic_ready
        if st.session_state.clear_prompt:
            st.session_state.prompt = ""
            st.session_state.clear_prompt = False
        if st.session_state.pending_prompt:
            st.session_state.prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = ""

        input_col, send_col = st.columns([7.5, 2.5])
        with input_col:
            st.text_area(
                "Your question",
                key="prompt",
                height=90,
                label_visibility="collapsed",
                placeholder="Ask anything...",
            )
        with send_col:
            send_clicked = st.button("Send", type="primary", use_container_width=True)

        mic_row = st.columns([7.5, 2.5])
        with mic_row[0]:
            if browser_mic_ready:
                st.markdown('<div class="mic-recorder">', unsafe_allow_html=True)
                mic_kwargs = dict(
                    start_prompt="Use Mic",
                    stop_prompt="Stop",
                    just_once=True,
                    key="browser_mic",
                )
                try:
                    audio_data = mic_recorder(use_container_width=True, **mic_kwargs)
                except TypeError:
                    audio_data = mic_recorder(**mic_kwargs)
                st.markdown("</div>", unsafe_allow_html=True)
                audio_bytes = None
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes") or audio_data.get("audio")
                elif isinstance(audio_data, (bytes, bytearray)):
                    audio_bytes = audio_data
                if audio_bytes:
                    try:
                        transcript = transcribe_audio_azure(audio_bytes)
                        if transcript:
                            st.session_state.pending_prompt = transcript
                            st.success(f"You said: {transcript}")
                            st.rerun()
                        else:
                            st.warning("Could not understand audio.")
                    except Exception as exc:
                        log_exception("Azure speech recognition failed", exc)
                        st.error(f"Speech recognition failed: {exc}")
            else:
                if st.button("Use Mic", use_container_width=True, disabled=not mic_ready):
                    recognize_speech()
        with mic_row[1]:
            st.markdown("", unsafe_allow_html=True)

        components.html(
            """
            <script>
            (function() {
              const doc = window.parent.document;
              const bind = () => {
                const textarea = doc.querySelector('.stTextArea textarea');
                if (!textarea || textarea.dataset.enterSubmitBound === "1") return;
                textarea.dataset.enterSubmitBound = "1";
                textarea.addEventListener('keydown', (e) => {
                  if (e.key !== 'Enter') return;
                  if (e.altKey) return; // Alt+Enter => newline
                  e.preventDefault(); // Enter => submit
                  const sendBtn = doc.querySelector('button[data-testid="baseButton-primary"]') ||
                                  doc.querySelector('button[kind="primary"]');
                  if (sendBtn) sendBtn.click();
                }, true);
              };

              const styleMicButton = () => {
                const accentBg = getComputedStyle(doc.documentElement).getPropertyValue('--accent-bg') || '#111214';
                const accentText = getComputedStyle(doc.documentElement).getPropertyValue('--accent-text') || '#ffffff';
                const shadowStrong = getComputedStyle(doc.documentElement).getPropertyValue('--shadow-strong') || 'rgba(0,0,0,0.22)';
                const iframes = Array.from(doc.querySelectorAll('iframe'));
                iframes.forEach((iframe) => {
                  try {
                    const idoc = iframe.contentDocument;
                    if (!idoc) return;
                    const btn = idoc.querySelector('button.myButton');
                    if (!btn) return;
                    let styleTag = idoc.getElementById('mic-style-injected');
                    if (!styleTag) {
                      styleTag = idoc.createElement('style');
                      styleTag.id = 'mic-style-injected';
                      idoc.head.appendChild(styleTag);
                    }
                    styleTag.textContent = `
                      html, body { width: 100%; height: 100%; margin: 0; padding: 0; background: transparent; }
                      #root, .App { width: 100%; height: 100%; margin: 0; padding: 0; }
                      .App { display: flex; }
                      button.myButton {
                        width: 100%;
                        flex: 1;
                        min-height: 44px;
                        border-radius: 10px;
                        border: none;
                        background: ${accentBg.trim()};
                        color: ${accentText.trim()};
                        font-weight: 600;
                        font-size: 0.95rem;
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        box-shadow: 0 12px 28px ${shadowStrong.trim()};
                        cursor: pointer;
                      }
                      button.myButton:hover { filter: brightness(0.95); }
                    `;
                  } catch (e) {
                    // iframe not ready or cross-origin
                  }
                });
              };

              const tick = () => {
                bind();
                styleMicButton();
              };

              tick();
              const observer = new MutationObserver(() => tick());
              observer.observe(doc.body, { childList: true, subtree: true });
            })();
            </script>
            """,
            height=0,
        )

        if not mic_ready:
            st.caption("Microphone disabled (enable Azure Speech for browser mic or install PyAudio).")
        if not TTS_AVAILABLE:
            st.caption("Voice output disabled on this host.")

        if send_clicked:
            prompt = st.session_state.prompt.strip()
            if not prompt:
                st.warning("Please type or speak your question.")
            else:
                history_context = build_context(st.session_state.messages)
                source_context = build_source_context(prompt)
                combined_context = "\n\n".join([c for c in [history_context, source_context] if c])
                add_message("user", prompt)
                st.session_state.clear_prompt = True

                recent_title = prompt[:48] + ("..." if len(prompt) > 48 else "")
                if recent_title in st.session_state.recent_chats:
                    st.session_state.recent_chats.remove(recent_title)
                st.session_state.recent_chats.insert(0, recent_title)
                st.session_state.recent_chats = st.session_state.recent_chats[:8]

                with st.spinner("Thinking..."):
                    source_sig = st.session_state.source_fingerprint if st.session_state.sources_indexed else "none"
                    cache_key = f"{st.session_state.mode}::{prompt}::{st.session_state.source_mode}::{source_sig}"
                    results = collection.query(query_texts=[cache_key], n_results=1)
                    documents = results.get("documents", [[]])[0]
                    distances = results.get("distances", [[]])[0]

                    if documents and distances and distances[0] < 0.2:
                        answer = results.get("metadatas", [[{}]])[0][0].get("answer", "")
                        status = "Cache hit"
                        source = "cache"
                    else:
                        formatted = build_prompt(prompt, st.session_state.mode, combined_context)
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
                    st.session_state.last_tts_audio = b""
                    add_message("assistant", answer, status=status, source=source)

                st.rerun()

        voice_col1, voice_col2 = st.columns(2)
        with voice_col1:
            if st.button("Stop Voice", use_container_width=True, disabled=not TTS_AVAILABLE):
                stop_speech()
                st.session_state.stop_audio = True
        with voice_col2:
            if st.button("Start Voice", use_container_width=True, disabled=not TTS_AVAILABLE):
                if st.session_state.last_answer.strip():
                    start_speech(st.session_state.last_answer)
                else:
                    st.warning("No answer to speak yet.")

        if st.session_state.last_tts_audio:
            st.audio(st.session_state.last_tts_audio, format="audio/wav")
        if st.session_state.auto_play_audio:
            components.html(
                """
                <script>
                (function() {
                  const doc = window.parent.document;
                  const audios = doc.querySelectorAll('audio');
                  if (!audios.length) return;
                  const audio = audios[audios.length - 1];
                  audio.play().catch(() => {});
                })();
                </script>
                """,
                height=0,
            )
            st.session_state.auto_play_audio = False
        if st.session_state.stop_audio:
            components.html(
                """
                <script>
                (function() {
                  const doc = window.parent.document;
                  const audios = doc.querySelectorAll('audio');
                  audios.forEach((audio) => {
                    audio.pause();
                    audio.currentTime = 0;
                  });
                })();
                </script>
                """,
                height=0,
            )
            st.session_state.stop_audio = False
