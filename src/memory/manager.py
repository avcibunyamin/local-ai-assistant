"""
Hafıza Yöneticisi.

Tüm hafıza bileşenlerini birleştiren ana sınıf:
- MemoryStore: uzun süreli semantik hafıza
- Preferences: kullanıcı tercihleri
- MemoryExtractor: LLM ile otomatik gerçek çıkarımı

Ana işlemler:
- remember(): Bir gerçeği hafızaya ekler (embedding üretip saklar)
- recall(): Bir soruya ilgili hafızaları getirir (semantik arama)
- learn_from_conversation(): Konuşmadan otomatik gerçek çıkarıp kaydeder
- build_memory_context(): Hafızaları LLM promptu için formatlar
"""

from src.config.settings import MemorySettings, RAGSettings
from src.core import LLMClient
from src.memory.extractor import MemoryExtractor
from src.memory.memory_store import Memory, MemoryStore
from src.memory.preferences import Preferences


class MemoryManager:
    """Hafıza sisteminin ana yöneticisi."""

    def __init__(
        self,
        settings: MemorySettings,
        rag_settings: RAGSettings,
        llm_client: LLMClient,
    ) -> None:
        self._settings = settings
        self._rag_settings = rag_settings  # embedding modeli için
        self._llm = llm_client
        self._store = MemoryStore(settings)
        self._prefs = Preferences(settings)
        self._extractor = MemoryExtractor(llm_client)

    async def _embed(self, text: str) -> list[float]:
        """Metni embedding vektörüne çevirir (RAG ile aynı model)."""
        embeddings = await self._llm.embed(
            [text],
            model=self._rag_settings.embedding_model,
        )
        return embeddings[0]

    async def remember(self, text: str, category: str = "fact") -> bool:
        """
        Bir gerçeği hafızaya ekler.

        Args:
            text: Hatırlanacak gerçek
            category: Hafıza türü

        Returns:
            True: yeni eklendi, False: zaten vardı
        """
        if self._store.get_by_text(text):
            return False
        embedding = await self._embed(text)
        self._store.add(text, embedding, category)
        return True

    async def recall(self, query: str) -> list[Memory]:
        """
        Bir soruya en ilgili hafızaları getirir.

        Args:
            query: Kullanıcı sorusu

        Returns:
            İlgili hafızalar (benzerlik sıralı)
        """
        if self._store.count() == 0:
            return []
        query_embedding = await self._embed(query)
        return self._store.search(
            query_embedding,
            top_k=self._settings.recall_top_k,
            min_score=self._settings.recall_min_score,
        )

    async def learn_from_conversation(self, messages: list[dict[str, str]]) -> list[str]:
        """
        Konuşmadan otomatik gerçek çıkarıp hafızaya ekler.

        Args:
            messages: Son konuşma mesajları

        Returns:
            Yeni öğrenilen gerçekler
        """
        if not self._settings.auto_extract or not messages:
            return []

        facts = await self._extractor.extract_facts(messages)
        learned: list[str] = []
        for fact in facts:
            if await self.remember(fact, category="fact"):
                learned.append(fact)
        return learned

    async def build_memory_context(self, query: str) -> str:
        """
        Soru için hafıza bağlamını oluşturur (LLM promptuna eklenir).

        Hem semantik hafızaları hem de kullanıcı tercihlerini içerir.

        Returns:
            Formatlanmış hafıza metni (boş = ilgili hafıza yok)
        """
        parts: list[str] = []

        # Kullanıcı tercihleri (her zaman eklenir)
        prefs = self._prefs.all()
        if prefs:
            pref_lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items())
            parts.append(f"Kullanici tercihleri:\n{pref_lines}")

        # İlgili semantik hafızalar
        memories = await self.recall(query)
        if memories:
            mem_lines = "\n".join(f"- {m.text}" for m in memories)
            parts.append(f"Kullanici hakkinda bilinenler:\n{mem_lines}")

        return "\n\n".join(parts)

    # ===== Tercih işlemleri (dışa aktarım) =====

    def set_preference(self, key: str, value: str) -> None:
        self._prefs.set(key, value)

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        return self._prefs.get(key, default)

    def get_preferences(self) -> dict[str, str]:
        return self._prefs.all()

    def delete_preference(self, key: str) -> None:
        self._prefs.delete(key)

    # ===== Hafıza işlemleri (dışa aktarım) =====

    def list_memories(self) -> list[Memory]:
        return self._store.list_all()

    def delete_memory(self, memory_id: int) -> None:
        self._store.delete(memory_id)

    def clear_memories(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return self._store.count()


_instance: "MemoryManager | None" = None


def get_memory_manager() -> MemoryManager:
    """Singleton MemoryManager döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        settings = get_settings()
        _instance = MemoryManager(
            settings.memory,
            settings.rag,
            LLMClient(settings.llm),
        )
    return _instance
