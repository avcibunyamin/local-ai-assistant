"""
Vektör deposu (FAISS).

Neden FAISS?
- Facebook AI Research tarafından geliştirilmiş, milyarlarca vektörde hızlı arama.
- Önceden derlenmiş wheel ile gelir (C++ derleme gerektirmez).
- Cosine similarity için IndexFlatIP (inner product) + normalize kullanılır.

FAISS sadece vektör araması yapar; metin ve metadata'yı biz ayrıca saklarız.
Bu yüzden iki yapı tutuyoruz:
- FAISS index → vektörler
- _documents listesi → chunk metinleri ve kaynak bilgisi

Kalıcılık:
- index dosyası (.faiss) + metadata dosyası (.pkl) diske yazılır.
- Uygulama yeniden başlasa bile yüklenen belgeler korunur.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Document:
    """Depolanan bir metin parçası."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """FAISS tabanlı vektör deposu."""

    def __init__(self, dimension: int, store_dir: Path) -> None:
        """
        Args:
            dimension: Embedding vektör boyutu (nomic-embed-text için 768)
            store_dir: Index ve metadata dosyalarının saklanacağı dizin
        """
        self._dimension = dimension
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = store_dir / "index.faiss"
        self._meta_path = store_dir / "documents.pkl"

        # Cosine similarity için inner product index (vektörler normalize edilir)
        self._index = faiss.IndexFlatIP(dimension)
        self._documents: list[Document] = []

        self._load()

    def add(self, embeddings: list[list[float]], documents: list[Document]) -> None:
        """
        Vektörleri ve belgeleri depoya ekler.

        Args:
            embeddings: Embedding vektörleri
            documents: Karşılık gelen Document nesneleri
        """
        if not embeddings:
            return
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)  # cosine similarity için normalize
        self._index.add(vectors)
        self._documents.extend(documents)
        self._save()

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[tuple[Document, float]]:
        """
        Sorgu vektörüne en yakın belgeleri bulur.

        Args:
            query_embedding: Sorgunun embedding vektörü
            top_k: Döndürülecek en yakın belge sayısı

        Returns:
            (Document, benzerlik skoru) çiftleri, en yakından uzağa
        """
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._documents[idx], float(score)))
        return results

    def count(self) -> int:
        """Depodaki toplam vektör sayısı."""
        return self._index.ntotal

    def list_sources(self) -> list[str]:
        """Yüklü benzersiz kaynak (dosya) isimlerini döner."""
        return sorted({doc.source for doc in self._documents})

    def clear(self) -> None:
        """Tüm vektörleri ve belgeleri siler."""
        self._index = faiss.IndexFlatIP(self._dimension)
        self._documents = []
        self._save()

    def _save(self) -> None:
        """
        Index ve metadata'yı diske yazar.

        Not: faiss.write_index() C++ dosya IO'su Windows'ta non-ASCII
        yolları (ör. "Masaüstü") açamıyor. Bu yüzden index'i byte'a
        serialize edip Python ile yazıyoruz — Python Unicode yolları sorunsuz açar.
        """
        index_bytes = faiss.serialize_index(self._index)
        with open(self._index_path, "wb") as f:
            f.write(index_bytes.tobytes())
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._documents, f)

    def _load(self) -> None:
        """Diskteki index ve metadata'yı yükler (varsa)."""
        if self._index_path.exists() and self._meta_path.exists():
            with open(self._index_path, "rb") as f:
                index_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            self._index = faiss.deserialize_index(index_bytes)
            with open(self._meta_path, "rb") as f:
                self._documents = pickle.load(f)
