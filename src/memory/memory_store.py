"""
Uzun süreli hafıza deposu.

Kullanıcı ve konuşmalar hakkındaki kalıcı gerçekleri saklar.
Örnek: "Kullanıcının adı Ahmet", "Python'u tercih ediyor".

Neden FAISS yerine SQLite + numpy?
- Hafızalar az sayıda olur (yüzlerce), belge parçaları gibi binlerce değil.
- Bu ölçekte numpy ile brute-force cosine similarity yeterince hızlı (<1ms).
- SQLite ana kaynak (source of truth) olunca silme/güncelleme SQL ile kolay.
- FAISS silmeyi zor destekler; bu senaryoda gereksiz karmaşıklık olurdu.

Her hafıza kaydı:
- text: gerçeğin kendisi
- embedding: semantik arama için vektör (SQLite'da BLOB)
- category: tür (fact, preference, event...)
- created_at: zaman
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config.settings import ROOT_DIR, MemorySettings


@dataclass
class Memory:
    """Tek bir hafıza kaydı."""

    id: int
    text: str
    category: str
    created_at: str
    score: float = 0.0  # arama sırasında doldurulur


class MemoryStore:
    """SQLite + numpy tabanlı uzun süreli hafıza deposu."""

    def __init__(self, settings: MemorySettings) -> None:
        db_path = ROOT_DIR / settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def add(self, text: str, embedding: list[float], category: str = "fact") -> int:
        """
        Yeni bir hafıza ekler.

        Args:
            text: Hatırlanacak gerçek
            embedding: Metnin embedding vektörü
            category: Hafıza türü (fact, preference, event)

        Returns:
            Eklenen hafızanın id'si
        """
        vec = np.array(embedding, dtype=np.float32).tobytes()
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO memories (text, category, embedding, created_at) VALUES (?, ?, ?, ?)",
            (text, category, vec, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def search(self, query_embedding: list[float], top_k: int = 5, min_score: float = 0.5) -> list[Memory]:
        """
        Sorguya en yakın hafızaları semantik olarak bulur.

        Args:
            query_embedding: Sorgu vektörü
            top_k: En fazla kaç hafıza dönsün
            min_score: Minimum benzerlik eşiği (altındakiler elenir)

        Returns:
            Benzerlik skoruna göre sıralı hafızalar
        """
        rows = self._conn.execute(
            "SELECT id, text, category, embedding, created_at FROM memories"
        ).fetchall()
        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        scored: list[Memory] = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            similarity = float(np.dot(query_vec, vec / norm))
            if similarity >= min_score:
                scored.append(Memory(
                    id=row["id"],
                    text=row["text"],
                    category=row["category"],
                    created_at=row["created_at"],
                    score=similarity,
                ))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def list_all(self) -> list[Memory]:
        """Tüm hafızaları yeniden eskiye listeler."""
        rows = self._conn.execute(
            "SELECT id, text, category, created_at FROM memories ORDER BY id DESC"
        ).fetchall()
        return [
            Memory(id=r["id"], text=r["text"], category=r["category"], created_at=r["created_at"])
            for r in rows
        ]

    def get_by_text(self, text: str) -> Memory | None:
        """Aynı metne sahip hafızayı döner (tekrar önleme için)."""
        row = self._conn.execute(
            "SELECT id, text, category, created_at FROM memories WHERE text = ?",
            (text,),
        ).fetchone()
        if row:
            return Memory(id=row["id"], text=row["text"], category=row["category"], created_at=row["created_at"])
        return None

    def delete(self, memory_id: int) -> None:
        """Bir hafızayı siler."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def clear(self) -> None:
        """Tüm hafızaları siler."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def count(self) -> int:
        """Toplam hafıza sayısı."""
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
