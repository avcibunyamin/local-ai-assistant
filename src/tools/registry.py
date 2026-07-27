"""
Araç kayıt sistemi (Tool Registry).

Tüm mevcut araçları tutar ve isimle erişim sağlar.
LLM'e sunulacak araç şemalarını üretir.

Plugin mimarisi: yeni araçlar register() ile eklenir.
"""

from src.tools.base import Tool, ToolResult


class ToolRegistry:
    """Araçları kaydeden ve yöneten merkezi kayıt."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Bir aracı kaydeder."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """İsimle bir aracı döner."""
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        """Tüm araçları döner."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Tüm araç isimlerini döner."""
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """Tüm araçların Ollama tool-calling şemalarını döner."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """
        Bir aracı isimle çalıştırır.

        Args:
            name: Araç adı
            arguments: Araç argümanları

        Returns:
            ToolResult (araç yoksa başarısız sonuç)
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, output=f"Bilinmeyen arac: {name}")
        try:
            return await tool.execute(**arguments)
        except TypeError as e:
            return ToolResult(success=False, output=f"Gecersiz argumanlar: {e}")
        except Exception as e:
            return ToolResult(success=False, output=f"Arac hatasi: {e}")


_instance: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """
    Singleton ToolRegistry döner.

    İlk çağrıda tüm builtin araçları kaydeder.
    """
    global _instance
    if _instance is None:
        _instance = ToolRegistry()
        from src.tools.builtin import register_builtin_tools
        register_builtin_tools(_instance)
        # Kullanıcı plugin'lerini yükle (plugins/ klasörü)
        from src.tools.plugin_loader import load_plugins
        load_plugins(_instance)
    return _instance
