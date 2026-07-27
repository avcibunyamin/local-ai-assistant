"""
Dosya işlemi araçları.

Güvenlik: Tüm işlemler workspace dizinine kısıtlıdır.
_resolve_safe_path(), yol geçişi (path traversal) saldırılarını engeller —
kullanıcı "../.." ile workspace dışına çıkamaz.

Araçlar:
- read_file:   Dosya oku
- write_file:  Dosya yaz/oluştur
- list_files:  Klasör içeriğini listele
- make_dir:    Klasör oluştur
"""

from pathlib import Path
from typing import Any

from src.config.settings import ROOT_DIR, ToolSettings
from src.tools.base import Tool, ToolResult


def _workspace(settings: ToolSettings) -> Path:
    """Workspace dizinini döner (yoksa oluşturur)."""
    ws = ROOT_DIR / settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _resolve_safe_path(settings: ToolSettings, path: str) -> Path:
    """
    Verilen yolu workspace içinde güvenli bir mutlak yola çevirir.

    Raises:
        ValueError: Yol workspace dışına çıkıyorsa
    """
    ws = _workspace(settings).resolve()
    target = (ws / path).resolve()
    # Yol geçişi kontrolü: hedef workspace altında olmalı
    if ws != target and ws not in target.parents:
        raise ValueError(f"Guvenlik: '{path}' calisma dizini disinda")
    return target


class ReadFileTool(Tool):
    name = "read_file"
    description = "Calisma dizinindeki bir dosyanin icerigini okur."

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Okunacak dosyanin yolu"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        try:
            target = _resolve_safe_path(self._settings, path)
            if not target.exists():
                return ToolResult(False, f"Dosya bulunamadi: {path}")
            content = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, str(e))


class WriteFileTool(Tool):
    name = "write_file"
    description = "Calisma dizininde bir dosyaya yazar (varsa uzerine yazar, yoksa olusturur)."

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Yazilacak dosyanin yolu"},
                "content": {"type": "string", "description": "Dosyaya yazilacak icerik"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            target = _resolve_safe_path(self._settings, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"Dosya yazildi: {path} ({len(content)} karakter)")
        except Exception as e:
            return ToolResult(False, str(e))


class ListFilesTool(Tool):
    name = "list_files"
    description = "Calisma dizinindeki bir klasorun icerigini listeler."

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Listelenecek klasor (varsayilan: kok)"},
            },
            "required": [],
        }

    async def execute(self, path: str = ".") -> ToolResult:
        try:
            target = _resolve_safe_path(self._settings, path)
            if not target.exists():
                return ToolResult(False, f"Klasor bulunamadi: {path}")
            if not target.is_dir():
                return ToolResult(False, f"Bu bir klasor degil: {path}")
            items = []
            for item in sorted(target.iterdir()):
                prefix = "[D]" if item.is_dir() else "[F]"
                items.append(f"{prefix} {item.name}")
            listing = "\n".join(items) if items else "(bos klasor)"
            return ToolResult(True, listing)
        except Exception as e:
            return ToolResult(False, str(e))


class MakeDirTool(Tool):
    name = "make_dir"
    description = "Calisma dizininde yeni bir klasor olusturur."

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Olusturulacak klasor yolu"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        try:
            target = _resolve_safe_path(self._settings, path)
            target.mkdir(parents=True, exist_ok=True)
            return ToolResult(True, f"Klasor olusturuldu: {path}")
        except Exception as e:
            return ToolResult(False, str(e))
