"""
Chat veri modelleri.

Pydantic modelleri ile API request/response şemaları tanımlanır.
OpenAI uyumlu format kullanılır — ileride farklı provider'larla uyum kolaylaşır.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Mesaj rolleri."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Tek bir chat mesajı."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """Chat API isteği."""

    messages: list[Message]
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = True


class ChatResponse(BaseModel):
    """Chat API yanıtı (streaming olmayan)."""

    content: str
    model: str


class ModelInfo(BaseModel):
    """Model bilgisi."""

    name: str
    size: int | None = None
    modified_at: str | None = None
