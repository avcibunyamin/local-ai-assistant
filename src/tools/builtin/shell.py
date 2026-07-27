"""
Terminal ve Python çalıştırma araçları.

DİKKAT: Bu araçlar tehlikelidir (requires_permission=True).
Sadece settings.tools.allow_dangerous=True ise kaydedilir.

Güvenlik önlemleri:
- Komutlar workspace dizininde çalışır (cwd).
- Zaman aşımı (timeout) ile sonsuz çalışma engellenir.
- asyncio.subprocess ile async, event loop'u bloklamaz.

Tam sandbox izolasyonu Aşama 11'de eklenecek. Şu an için
allow_dangerous bayrağı ve workspace kısıtlaması temel korumadır.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from src.config.settings import ROOT_DIR, ToolSettings
from src.tools.base import Tool, ToolResult


def _workspace(settings: ToolSettings) -> Path:
    ws = ROOT_DIR / settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    return ws


async def _run_subprocess(
    args: list[str],
    cwd: Path,
    timeout: int,
) -> ToolResult:
    """Bir alt süreci async çalıştırır, çıktısını döner."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(False, f"Zaman asimi ({timeout}s) - komut durduruldu")

        output = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0:
            return ToolResult(True, output or "(cikti yok)")
        return ToolResult(False, f"Cikis kodu {proc.returncode}:\n{output}")
    except FileNotFoundError:
        return ToolResult(False, f"Komut bulunamadi: {args[0]}")
    except Exception as e:
        return ToolResult(False, str(e))


class ShellTool(Tool):
    name = "run_shell"
    description = (
        "Bir terminal (shell) komutu calistirir ve ciktisini doner. "
        "Komut calisma dizininde calisir."
    )
    requires_permission = True

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Calistirilacak shell komutu"},
            },
            "required": ["command"],
        }

    async def execute(self, command: str) -> ToolResult:
        # Windows'ta cmd, diğerlerinde sh üzerinden çalıştır
        if sys.platform == "win32":
            args = ["cmd", "/c", command]
        else:
            args = ["sh", "-c", command]
        return await _run_subprocess(
            args,
            cwd=_workspace(self._settings),
            timeout=self._settings.exec_timeout,
        )


class PythonExecTool(Tool):
    name = "run_python"
    description = (
        "Verilen Python kodunu calistirir ve ciktisini (print) doner. "
        "Kod calisma dizininde izole bir surecte calisir."
    )
    requires_permission = True

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Calistirilacak Python kodu"},
            },
            "required": ["code"],
        }

    async def execute(self, code: str) -> ToolResult:
        # Kodu geçici bir dosyaya yazıp mevcut Python yorumlayıcısıyla çalıştır
        ws = _workspace(self._settings)
        script = ws / "_temp_exec.py"
        try:
            script.write_text(code, encoding="utf-8")
            result = await _run_subprocess(
                [sys.executable, str(script)],
                cwd=ws,
                timeout=self._settings.exec_timeout,
            )
            return result
        finally:
            script.unlink(missing_ok=True)
