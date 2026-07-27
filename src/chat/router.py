"""
Chat API endpoint'leri.

FastAPI router ile chat ve geçmiş işlemleri:
- POST /chat          : Streaming chat (SSE)
- POST /chat/complete : Tam yanıt
- GET  /models        : Yüklü modelleri listele
- GET  /health        : Ollama sağlık kontrolü
- GET  /conversations : Konuşma listesi
- GET  /conversations/{id}/messages : Konuşma mesajları
- DELETE /conversations/{id} : Konuşma silme
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.chat.history import get_chat_history
from src.chat.models import ChatRequest, ChatResponse, ModelInfo
from src.config import get_settings
from src.core import LLMClient
from src.memory.manager import get_memory_manager
from src.rag.engine import get_rag_engine

router = APIRouter()


def _get_client() -> LLMClient:
    return LLMClient(get_settings().llm)


def _build_messages(request: ChatRequest) -> list[dict[str, str]]:
    """Sistem promptunu ekleyerek mesaj listesi oluşturur."""
    settings = get_settings()
    messages = [{"role": "system", "content": settings.llm.system_prompt}]
    messages.extend(
        {"role": msg.role.value, "content": msg.content}
        for msg in request.messages
    )
    return messages


@router.post("/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint'i."""
    client = _get_client()
    messages = _build_messages(request)

    async def generate():
        try:
            async for chunk in client.stream_chat(
                messages,
                model=request.model,
                temperature=request.temperature,
            ):
                yield chunk
        except Exception as e:
            yield f"\n[Hata: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(request: ChatRequest) -> ChatResponse:
    """Tam yanıt dönen chat endpoint'i."""
    client = _get_client()
    messages = _build_messages(request)

    try:
        content = await client.chat(
            messages,
            model=request.model,
            temperature=request.temperature,
        )
        return ChatResponse(
            content=content,
            model=request.model or get_settings().llm.model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM hatasi: {e}")


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """Ollama'da yüklü modelleri listeler."""
    client = _get_client()
    try:
        models = await client.list_models()
        return [
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                modified_at=m.get("modified_at"),
            )
            for m in models
        ]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama baglanti hatasi: {e}")


@router.get("/health")
async def health_check() -> dict[str, bool]:
    """Ollama sunucusunun durumunu kontrol eder."""
    client = _get_client()
    is_healthy = await client.health_check()
    return {"ollama_running": is_healthy}


# ===== Chat Geçmişi Endpoint'leri =====


@router.get("/conversations")
async def list_conversations() -> list[dict]:
    """Tüm konuşmaları listeler."""
    return get_chat_history().list_conversations()


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: int) -> list[dict]:
    """Bir konuşmanın mesajlarını döner."""
    history = get_chat_history()
    conv = history.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Konusma bulunamadi")
    return history.get_messages(conversation_id)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int) -> dict[str, str]:
    """Konuşmayı siler."""
    history = get_chat_history()
    conv = history.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Konusma bulunamadi")
    history.delete_conversation(conversation_id)
    return {"status": "silindi"}


# ===== RAG Endpoint'leri =====


@router.get("/rag/sources")
async def rag_sources() -> dict:
    """Yüklü belgeleri ve parça sayısını döner."""
    engine = get_rag_engine()
    return {"sources": engine.list_sources(), "chunk_count": engine.count()}


@router.post("/rag/query")
async def rag_query(query: str) -> dict:
    """Bir sorgu için ilgili belge parçalarını döner."""
    engine = get_rag_engine()
    results = await engine.retrieve(query)
    return {
        "results": [
            {"source": doc.source, "text": doc.text, "score": score}
            for doc, score in results
        ]
    }


@router.delete("/rag/documents")
async def rag_clear() -> dict[str, str]:
    """Tüm yüklü belgeleri siler."""
    get_rag_engine().clear()
    return {"status": "temizlendi"}


# ===== Hafıza Endpoint'leri =====


@router.get("/memory")
async def memory_list() -> list[dict]:
    """Kaydedilmiş hafızaları listeler."""
    memories = get_memory_manager().list_memories()
    return [
        {"id": m.id, "text": m.text, "category": m.category, "created_at": m.created_at}
        for m in memories
    ]


@router.post("/memory")
async def memory_add(text: str, category: str = "fact") -> dict:
    """Yeni bir hafıza ekler."""
    added = await get_memory_manager().remember(text, category)
    return {"added": added, "text": text}


@router.delete("/memory/{memory_id}")
async def memory_delete(memory_id: int) -> dict[str, str]:
    """Bir hafızayı siler."""
    get_memory_manager().delete_memory(memory_id)
    return {"status": "silindi"}


@router.delete("/memory")
async def memory_clear() -> dict[str, str]:
    """Tüm hafızayı siler."""
    get_memory_manager().clear_memories()
    return {"status": "temizlendi"}


@router.get("/preferences")
async def preferences_list() -> dict[str, str]:
    """Kullanıcı tercihlerini döner."""
    return get_memory_manager().get_preferences()


@router.put("/preferences")
async def preferences_set(key: str, value: str) -> dict[str, str]:
    """Bir tercihi ayarlar."""
    get_memory_manager().set_preference(key, value)
    return {"status": "ayarlandi", "key": key, "value": value}


# ===== Araç (Tool) Endpoint'leri =====


@router.get("/tools")
async def tools_list() -> list[dict]:
    """Kullanılabilir araçları listeler."""
    from src.tools.registry import get_registry
    registry = get_registry()
    return [
        {"name": t.name, "description": t.description, "dangerous": t.requires_permission}
        for t in registry.all()
    ]


# ===== Vision Endpoint'leri =====


@router.post("/vision/ocr")
async def vision_ocr(image_path: str) -> dict[str, str]:
    """Bir görüntüdeki metni çıkarır (workspace içindeki yol)."""
    from src.vision.engine import get_vision_engine
    text = await get_vision_engine().ocr(image_path)
    return {"text": text}
