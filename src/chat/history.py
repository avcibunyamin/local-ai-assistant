"""
Chat geçmişi yöneticisi.

Neden SQLite?
- Python'da dahili (sqlite3) — ekstra kurulum yok.
- Dosya tabanlı — tek bir .db dosyasında tüm geçmiş.
- SQL ile arama, sıralama, filtreleme kolay.
- JSON dosyalarına göre çok daha performanslı (binlerce konuşmada bile hızlı).
- Thread-safe (check_same_thread=False ile).

Tablo yapısı:
- conversations: Konuşma listesi (id, başlık, oluşturma/güncelleme zamanı)
- messages: Mesajlar (id, conversation_id, role, content, zaman)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config.settings import ROOT_DIR, ChatSettings


class ChatHistory:
    """SQLite tabanlı chat geçmişi yöneticisi."""

    def __init__(self, settings: ChatSettings) -> None:
        db_path = ROOT_DIR / settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_conversations = settings.max_conversations
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'Yeni Sohbet',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
        """)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str = "Yeni Sohbet") -> int:
        """Yeni konuşma oluşturur, id döner."""
        now = self._now()
        cursor = self._conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        self._conn.commit()
        self._cleanup_old_conversations()
        return cursor.lastrowid

    def list_conversations(self) -> list[dict]:
        """Tüm konuşmaları yeniden eskiye listeler."""
        rows = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Tek bir konuşmayı döner."""
        row = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None

    def search_conversations(self, query: str) -> list[dict]:
        """
        Başlıkta VEYA mesaj içeriğinde geçen konuşmaları arar.

        Neden SQL LIKE + DISTINCT?
        - LIKE ile hem başlık hem mesaj metninde alt-dize araması yapılır.
        - Bir konuşmanın birden çok mesajı eşleşebileceği için DISTINCT ile
          her konuşma tek kez döner.
        - Boş sorgu tüm listeyi döndürür (filtre yok gibi davranır).
        """
        if not query or not query.strip():
            return self.list_conversations()
        like = f"%{query.strip()}%"
        rows = self._conn.execute(
            """
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.title LIKE ? OR m.content LIKE ?
            ORDER BY c.updated_at DESC
            """,
            (like, like),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_title(self, conversation_id: int, title: str) -> None:
        """Konuşma başlığını günceller."""
        self._conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, self._now(), conversation_id),
        )
        self._conn.commit()

    def delete_conversation(self, conversation_id: int) -> None:
        """Konuşmayı ve mesajlarını siler."""
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        """Mesaj ekler, mesaj id döner."""
        now = self._now()
        cursor = self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_messages(self, conversation_id: int) -> list[dict]:
        """Konuşmadaki tüm mesajları sırayla döner."""
        rows = self._conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_message(self, message_id: int) -> None:
        """Tek bir mesajı siler."""
        self._conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self._conn.commit()

    def edit_message(self, message_id: int, content: str) -> None:
        """Mesaj içeriğini düzenler."""
        self._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (content, message_id),
        )
        self._conn.commit()

    def auto_title(self, conversation_id: int) -> str:
        """İlk kullanıcı mesajından otomatik başlık oluşturur (ilk 50 karakter)."""
        row = self._conn.execute(
            "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY id LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row:
            title = row["content"][:50].strip()
            if len(row["content"]) > 50:
                title += "..."
            self.update_title(conversation_id, title)
            return title
        return "Yeni Sohbet"

    def _cleanup_old_conversations(self) -> None:
        """Maksimum konuşma sayısını aşarsa en eskileri siler."""
        count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        if count > self._max_conversations:
            self._conn.execute(
                """DELETE FROM conversations WHERE id IN (
                    SELECT id FROM conversations ORDER BY updated_at ASC LIMIT ?
                )""",
                (count - self._max_conversations,),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_instance: Optional[ChatHistory] = None


def get_chat_history() -> ChatHistory:
    """Singleton ChatHistory döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        _instance = ChatHistory(get_settings().chat)
    return _instance
