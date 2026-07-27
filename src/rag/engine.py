"""
RAG Motoru.

Tüm RAG bileşenlerini birleştiren ana sınıf:
1. Belge yükleme (document_loader)
2. Metin bölme (text_splitter)
3. Embedding üretme (LLMClient.embed)
4. Vektör saklama/arama (vector_store)

İki ana işlem:
- ingest(): Belge yükle → böl → embed et → depola
- retrieve(): Soru → embed et → en yakın parçaları bul
- build_context(): Bulunan parçaları LLM promptu için formatla
"""

from pathlib import Path

from src.config.settings import RAGSettings, ROOT_DIR
from src.core import LLMClient
from src.rag.document_loader import is_supported, load_document
from src.rag.text_splitter import split_text
from src.rag.vector_store import Document, VectorStore


class RAGEngine:
    """Belge yükleme ve bağlam getirme motoru."""

    def __init__(self, settings: RAGSettings, llm_client: LLMClient) -> None:
        self._settings = settings
        self._llm = llm_client
        self._store = VectorStore(
            dimension=settings.embedding_dimension,
            store_dir=ROOT_DIR / settings.store_dir,
        )

    async def ingest_file(self, file_path: str | Path) -> int:
        """
        Bir dosyayı yükler, işler ve vektör deposuna ekler.

        Args:
            file_path: Yüklenecek dosyanın yolu

        Returns:
            Eklenen parça (chunk) sayısı

        Raises:
            ValueError: Desteklenmeyen dosya formatı
        """
        path = Path(file_path)
        if not is_supported(path):
            raise ValueError(f"Desteklenmeyen dosya formati: {path.suffix}")

        text = load_document(path)
        if not text.strip():
            return 0

        chunks = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            return 0

        embeddings = await self._llm.embed(
            chunks,
            model=self._settings.embedding_model,
        )
        documents = [
            Document(text=chunk, source=path.name, metadata={"chunk_index": i})
            for i, chunk in enumerate(chunks)
        ]
        self._store.add(embeddings, documents)
        return len(chunks)

    async def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """
        Sorguya en alakalı belge parçalarını getirir.

        Args:
            query: Kullanıcı sorusu

        Returns:
            (Document, benzerlik skoru) çiftleri
        """
        if self._store.count() == 0:
            return []

        query_embedding = await self._llm.embed(
            [query],
            model=self._settings.embedding_model,
        )
        return self._store.search(query_embedding[0], top_k=self._settings.top_k)

    async def build_context(self, query: str) -> str:
        """
        Sorgu için LLM'e verilecek bağlam metnini oluşturur.

        Returns:
            Formatlanmış bağlam metni (boş string = alakalı belge yok)
        """
        results = await self.retrieve(query)
        if not results:
            return ""

        parts: list[str] = []
        for doc, score in results:
            parts.append(f"[Kaynak: {doc.source}]\n{doc.text}")
        return "\n\n---\n\n".join(parts)

    def count(self) -> int:
        """Depodaki toplam parça sayısı."""
        return self._store.count()

    def list_sources(self) -> list[str]:
        """Yüklü dosya isimlerini döner."""
        return self._store.list_sources()

    def clear(self) -> None:
        """Tüm yüklü belgeleri siler."""
        self._store.clear()


_instance: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Singleton RAGEngine döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        settings = get_settings()
        _instance = RAGEngine(settings.rag, LLMClient(settings.llm))
    return _instance
