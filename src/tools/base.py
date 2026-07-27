"""
Araç (Tool) temel altyapısı.

Her araç, LLM'in çağırabileceği bir yeteneği temsil eder
(dosya oku, terminal çalıştır, vs.).

Tasarım (SOLID):
- Tool: soyut temel sınıf. Her araç bundan türer (Open/Closed prensibi).
- ToolResult: aracın çalışma sonucu.
- Araç, kendini OpenAI/Ollama tool-calling formatına çevirebilir (to_schema).

Bu yapı plugin mimarisine hazırdır (Aşama 12): yeni araç eklemek =
yeni bir Tool alt sınıfı yazıp registry'ye kaydetmek.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Bir aracın çalışma sonucu."""

    success: bool
    output: str

    def to_message(self) -> str:
        """LLM'e geri gönderilecek metin."""
        if self.success:
            return self.output
        return f"HATA: {self.output}"


class Tool(ABC):
    """Tüm araçların türediği soyut temel sınıf."""

    # Aracın benzersiz adı (LLM bu adla çağırır)
    name: str
    # Aracın ne yaptığının açıklaması (LLM buna göre karar verir)
    description: str
    # Aracın tehlikeli olup olmadığı (True ise onay/izin gerekebilir)
    requires_permission: bool = False

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Aracın parametrelerini JSON Schema formatında döner.

        Örnek:
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Dosya yolu"}
            },
            "required": ["path"]
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Aracı verilen argümanlarla çalıştırır."""
        ...

    def to_schema(self) -> dict[str, Any]:
        """
        Aracı Ollama/OpenAI tool-calling şemasına çevirir.

        Ollama /api/chat "tools" parametresi bu formatı bekler.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }
