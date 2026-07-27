"""
Ollama LLM Client.

Ollama REST API ile async iletişim kurar.
httpx kullanarak FastAPI'nin async yapısını bozmaz.

Neden httpx?
- requests kütüphanesi senkron çalışır ve async endpoint'leri bloklar.
- httpx hem sync hem async destekler, streaming için AsyncByteStream sağlar.
- Ollama'nın /api/chat endpoint'i streaming JSON döner (her satır bir JSON objesi).

Kullanım:
    client = LLMClient(settings.llm)
    async for chunk in client.stream_chat(messages):
        print(chunk)
"""

import json
from collections.abc import AsyncGenerator
from typing import Optional

import httpx

from src.config.settings import LLMSettings


class LLMClient:
    """Ollama API ile async iletişim sağlayan istemci."""

    # Embedding cache: aynı metni tekrar tekrar vektörlemeyi önler.
    # Sınıf düzeyinde paylaşılır — tüm LLMClient örnekleri aynı cache'i kullanır.
    # (RAG ve hafıza aynı metinleri embed edebilir; bu tekrarı ortadan kaldırır.)
    _embed_cache: dict[str, list[float]] = {}
    _EMBED_CACHE_LIMIT = 2000

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Ollama'ya mesaj gönderir ve cevabı parça parça (streaming) döner.

        Ollama /api/chat endpoint'i her satırda bir JSON objesi döner:
        {"message": {"content": "Merhaba"}, "done": false}
        {"message": {"content": " nasılsın"}, "done": false}
        {"message": {"content": ""}, "done": true}

        Args:
            messages: OpenAI formatında mesaj listesi [{"role": "user", "content": "..."}]
            model: Kullanılacak model (None ise ayarlardan alınır)
            temperature: Sıcaklık değeri (None ise ayarlardan alınır)

        Yields:
            Her bir metin parçası (token)
        """
        payload = {
            "model": model or self._settings.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self._settings.temperature,
                "num_predict": self._settings.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        return

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Streaming olmadan tam cevap döner.

        Küçük cevaplar veya tool calling gibi durumlar için.
        """
        chunks: list[str] = []
        async for chunk in self.stream_chat(messages, model, temperature):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream_vision(
        self,
        prompt: str,
        images_base64: list[str],
        model: str,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Görüntü(ler) ile birlikte bir soru gönderir, cevabı streaming döner.

        Ollama multimodal modeller (qwen2.5vl, llava) mesajın "images"
        alanında base64 kodlu görüntü listesi alır.

        Args:
            prompt: Görüntü hakkındaki soru/talimat
            images_base64: base64 kodlu görüntüler (data URI öneki OLMADAN)
            model: Vision modeli (ör. qwen2.5vl:7b)
            temperature: Sıcaklık

        Yields:
            Cevap metni parçaları
        """
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": images_base64,
            }],
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self._settings.temperature,
            },
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        return

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: Optional[str] = None,
    ) -> dict:
        """
        Araç (tool) desteğiyle tek bir chat turu yapar.

        Ollama /api/chat "tools" parametresiyle çağrılır. Model ya
        normal bir cevap ("content") ya da araç çağrıları ("tool_calls")
        döner. Streaming yok — tool calling adımı atomik olmalı.

        Args:
            messages: Konuşma geçmişi (tool sonuçları dahil)
            tools: Araç şemaları (registry.schemas())
            model: Kullanılacak model

        Returns:
            Ollama "message" objesi: {"role", "content", "tool_calls"?}
        """
        payload = {
            "model": model or self._settings.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self._settings.temperature},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("message", {})

    async def embed(
        self,
        texts: list[str],
        model: str = "nomic-embed-text",
    ) -> list[list[float]]:
        """
        Metinleri embedding vektörlerine dönüştürür.

        Ollama /api/embed endpoint'i kullanılır.
        RAG için metinleri sayısal vektörlere çevirmemiz gerekir.

        Args:
            texts: Vektöre dönüştürülecek metinler
            model: Embedding modeli (varsayılan: nomic-embed-text)

        Returns:
            Her metin için bir embedding vektörü

        Optimizasyon: Cache'te olan metinler API'ye gönderilmez.
        Sadece cache'te olmayan (yeni) metinler Ollama'ya sorulur.
        """
        # Cache'te olmayan metinleri belirle
        missing = [t for t in texts if t not in self._embed_cache]

        if missing:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": model, "input": missing},
                )
                response.raise_for_status()
                new_embeddings = response.json().get("embeddings", [])

            # Yeni embedding'leri cache'e ekle
            for text, emb in zip(missing, new_embeddings):
                if len(self._embed_cache) >= self._EMBED_CACHE_LIMIT:
                    # Basit temizlik: en eski girdiyi at (FIFO)
                    self._embed_cache.pop(next(iter(self._embed_cache)))
                self._embed_cache[text] = emb

        # Tüm metinleri cache'ten sırayla döndür
        return [self._embed_cache[t] for t in texts]

    async def list_models(self) -> list[dict]:
        """Ollama'da yüklü modelleri listeler."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])

    async def health_check(self) -> bool:
        """Ollama sunucusunun çalışıp çalışmadığını kontrol eder."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self._base_url)
                return response.status_code == 200
        except httpx.ConnectError:
            return False
