"""Configuration & Environment Loader for Project J.A.R.V.I.S."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from dotenv import load_dotenv

# Prefer IPv4 on Windows to prevent 10-20s IPv6 AAAA timeout stalls on Gemini & Edge-TTS
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    except Exception:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_first_getaddrinfo

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Centralized runtime settings loaded from .env."""

    ROOT_DIR: Path = ROOT_DIR
    BRAIN_VAULT_PATH: Path = ROOT_DIR / os.getenv("BRAIN_VAULT_PATH", "brain")
    SQLITE_DB_PATH: Path = ROOT_DIR / os.getenv("SQLITE_DB_PATH", "server/jarvis_brain.sqlite3")

    # Primary Cloud Tier (Google AI Studio Free Tier)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_PRIMARY_MODEL: str = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.1-flash-lite")
    GEMINI_PRO_MODEL: str = os.getenv("GEMINI_PRO_MODEL", "gemini-3.5-flash-lite")
    GEMINI_LIVE_MODEL: str = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-exp")
    GEMINI_RPM_LIMIT: int = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
    GEMINI_TPM_LIMIT: int = int(os.getenv("GEMINI_TPM_LIMIT", "250000"))

    # Local Fallback & Privacy Tier (Ollama / Llama.cpp)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    OLLAMA_FALLBACK_MODEL: str = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")

    # Full-Duplex Voice & Audio Subsystem
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "auto")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "auto")
    KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "bm_george")
    BARGE_IN_THRESHOLD_MS: int = int(os.getenv("BARGE_IN_THRESHOLD_MS", "180"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")


settings = Settings()
settings.BRAIN_VAULT_PATH.mkdir(parents=True, exist_ok=True)
settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

