"""
Merkezi konfigürasyon yönetimi.

YAML dosyasından ayarları okur ve Pydantic ile doğrular.
Singleton pattern ile uygulama genelinde tek bir Settings nesnesi kullanılır.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# Proje kök dizini
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"


class LLMSettings(BaseModel):
    """LLM bağlantı ve üretim ayarları."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    system_prompt: str = "Sen yardimci bir AI asistanisin. Turkce cevap ver."


class ServerSettings(BaseModel):
    """FastAPI sunucu ayarları."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class UISettings(BaseModel):
    """Gradio arayüz ayarları."""

    title: str = "Local AI Assistant"
    theme: str = "soft"
    share: bool = False


class ChatSettings(BaseModel):
    """Chat geçmişi ayarları."""

    db_path: str = "data/chat_history.db"
    max_conversations: int = Field(default=100, ge=1)


class RAGSettings(BaseModel):
    """RAG (belge tabanlı cevaplama) ayarları."""

    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = Field(default=768, ge=1)
    store_dir: str = "data/vector_store"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=4, ge=1)


class MemorySettings(BaseModel):
    """Hafıza sistemi ayarları."""

    db_path: str = "data/memory.db"
    # Otomatik gerçek çıkarımı açık mı
    auto_extract: bool = True
    # Geri çağrılacak hafıza sayısı
    recall_top_k: int = Field(default=5, ge=1)
    # Minimum benzerlik eşiği (0-1)
    recall_min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ToolSettings(BaseModel):
    """Araç (tool calling) ayarları."""

    # Araçların çalışabileceği güvenli çalışma dizini
    workspace_dir: str = "data/workspace"
    # Tehlikeli araçlar (shell, python) etkin mi
    allow_dangerous: bool = False
    # Bir tool-calling döngüsünde maksimum tur sayısı (sonsuz döngü koruması)
    max_iterations: int = Field(default=5, ge=1, le=20)
    # Shell/python komut zaman aşımı (saniye)
    exec_timeout: int = Field(default=30, ge=1)


class VisionSettings(BaseModel):
    """Vision (görüntü analizi) ayarları."""

    # Multimodal model (ollama pull ile indirin)
    model: str = "qwen2.5vl:7b"
    # Varsayılan analiz talimatı
    default_prompt: str = "Bu goruntude ne goruyorsun? Detayli acikla."


class VoiceSettings(BaseModel):
    """Ses (STT/TTS) ayarları."""

    # Whisper model boyutu (tiny, base, small, medium, large-v3)
    stt_model: str = "small"
    # STT dili (tr = Türkçe, None = otomatik algıla)
    stt_language: str = "tr"
    # Whisper compute type (int8=hızlı+az RAM, float16=GPU)
    stt_compute_type: str = "int8"
    # TTS konuşma hızı (kelime/dakika)
    tts_rate: int = Field(default=180, ge=50, le=400)


class Settings(BaseModel):
    """Tüm uygulama ayarlarını birleştiren ana model."""

    llm: LLMSettings = LLMSettings()
    server: ServerSettings = ServerSettings()
    ui: UISettings = UISettings()
    chat: ChatSettings = ChatSettings()
    rag: RAGSettings = RAGSettings()
    memory: MemorySettings = MemorySettings()
    tools: ToolSettings = ToolSettings()
    vision: VisionSettings = VisionSettings()
    voice: VoiceSettings = VoiceSettings()


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """YAML dosyasını okur, yoksa boş dict döner."""
    config_path = path or CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Uygulama ayarlarını döner (singleton).

    İlk çağrıda YAML'dan okur ve doğrular.
    Sonraki çağrılarda cache'den döner.
    """
    raw = load_yaml_config()
    return Settings(**raw)
