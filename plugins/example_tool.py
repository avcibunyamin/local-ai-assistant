"""
Örnek Plugin.

Kendi araçlarınızı bu klasöre ekleyerek AI'ya yeni yetenekler
kazandırabilirsiniz. Tek yapmanız gereken:
1. Tool sınıfından türeyen bir sınıf yazın.
2. name, description belirleyin.
3. parameters() ve execute() metodlarını doldurun.

Bu dosya uygulama başladığında otomatik yüklenir.
"""

from typing import Any

from src.tools.base import Tool, ToolResult


class WordCountTool(Tool):
    """Örnek: Bir metindeki kelime sayısını sayar."""

    name = "count_words"
    description = "Verilen metindeki kelime sayisini sayar."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Kelimesi sayilacak metin"},
            },
            "required": ["text"],
        }

    async def execute(self, text: str) -> ToolResult:
        count = len(text.split())
        return ToolResult(True, f"Kelime sayisi: {count}")
