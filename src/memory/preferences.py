"""
Kullanıcı tercihleri deposu.

Basit key-value ayarlar için (örn. "yanit_dili" = "turkce",
"kod_stili" = "pep8", "hitap" = "sen").

Neden ayrı bir key-value tablo (hafıza deposundan bağımsız)?
- Tercihler yapısal ve benzersiz anahtarlıdır (semantik arama gerektirmez).
- Doğrudan anahtarla erişim gerekir ("yanit_dili nedir?").
- Üzerine yazma (upsert) mantığı gerekir.
Bu yüzden embedding'li hafıza deposundan ayrı, basit bir tablo kullanıyoruz.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import ROOT_DIR, MemorySettings


class Preferences:
    """SQLite key-value kullanıcı tercih deposu."""

    def __init__(self, settings: MemorySettings) -> None:
        db_path = ROOT_DIR / settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def set(self, key: str, value: str) -> None:
        """Bir tercihi ayarlar (varsa üzerine yazar)."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (key, value, now, value, now),
        )
        self._conn.commit()

    def get(self, key: str, default: str | None = None) -> str | None:
        """Bir tercihi döner."""
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def all(self) -> dict[str, str]:
        """Tüm tercihleri döner."""
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete(self, key: str) -> None:
        """Bir tercihi siler."""
        self._conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        """Tüm tercihleri siler."""
        self._conn.execute("DELETE FROM preferences")
        self._conn.commit()
