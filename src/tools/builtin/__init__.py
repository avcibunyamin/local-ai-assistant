"""
Builtin (hazır) araçların kaydı.

register_builtin_tools() tüm hazır araçları registry'ye ekler.
Tehlikeli araçlar (shell, python) sadece allow_dangerous=True ise eklenir.
"""

from src.config import get_settings
from src.tools.builtin.computer import all_computer_tools
from src.tools.builtin.file_ops import (
    ListFilesTool,
    MakeDirTool,
    ReadFileTool,
    WriteFileTool,
)
from src.tools.builtin.shell import PythonExecTool, ShellTool


def register_builtin_tools(registry) -> None:
    """Hazır araçları registry'ye kaydeder."""
    settings = get_settings().tools

    # Güvenli dosya araçları (her zaman)
    registry.register(ReadFileTool(settings))
    registry.register(WriteFileTool(settings))
    registry.register(ListFilesTool(settings))
    registry.register(MakeDirTool(settings))

    # Tehlikeli araçlar (sadece izin verilmişse)
    if settings.allow_dangerous:
        registry.register(ShellTool(settings))
        registry.register(PythonExecTool(settings))
        # Bilgisayar kontrol araçları (fare, klavye, ekran)
        for tool in all_computer_tools(settings):
            registry.register(tool)
