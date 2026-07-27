"""
Bilgisayar kontrol araçları.

DİKKAT: Bu araçlar kullanıcının gerçek fare, klavye ve ekranını kontrol eder.
Hepsi requires_permission=True; sadece allow_dangerous=True ise kaydedilir.

pyautogui kullanır:
- screenshot: Ekran görüntüsü al (Vision ile analiz edilebilir)
- move_mouse / click: Fare kontrolü
- type_text: Klavye ile yazma
- press_key: Tuş basma
- open_program: Program başlatma

Güvenlik önlemleri:
- pyautogui.FAILSAFE=True: Fareyi sol üst köşeye götürünce acil durdurma.
- Ekran görüntüleri workspace'e kaydedilir.
- open_program subprocess ile, kabuk enjeksiyonuna karşı liste argümanı kullanır.

Not: pyautogui sadece görsel koordinatlarla çalışır (akıllı değil).
Agent + Vision kombinasyonu ile "ekranı gör, sonra tıkla" mümkün.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.config.settings import ROOT_DIR, ToolSettings
from src.tools.base import Tool, ToolResult


def _lazy_pyautogui():
    """pyautogui'yi lazy import eder (import anında ekran erişimi ister)."""
    import pyautogui
    pyautogui.FAILSAFE = True
    return pyautogui


class ScreenshotTool(Tool):
    name = "take_screenshot"
    description = "Ekranin goruntusunu alir ve dosyaya kaydeder. Yolu doner."
    requires_permission = True

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        try:
            pyautogui = _lazy_pyautogui()
            out_dir = ROOT_DIR / self._settings.workspace_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "screenshot.png"
            pyautogui.screenshot(str(path))
            return ToolResult(True, f"Ekran goruntusu kaydedildi: {path}")
        except Exception as e:
            return ToolResult(False, str(e))


class ClickTool(Tool):
    name = "click_mouse"
    description = "Belirtilen ekran koordinatina (x, y) fare ile tiklar."
    requires_permission = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Yatay koordinat (piksel)"},
                "y": {"type": "integer", "description": "Dikey koordinat (piksel)"},
            },
            "required": ["x", "y"],
        }

    async def execute(self, x: int, y: int) -> ToolResult:
        try:
            pyautogui = _lazy_pyautogui()
            pyautogui.click(x, y)
            return ToolResult(True, f"Tiklandi: ({x}, {y})")
        except Exception as e:
            return ToolResult(False, str(e))


class TypeTextTool(Tool):
    name = "type_text"
    description = "Klavye ile verilen metni yazar (aktif pencereye)."
    requires_permission = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Yazilacak metin"},
            },
            "required": ["text"],
        }

    async def execute(self, text: str) -> ToolResult:
        try:
            pyautogui = _lazy_pyautogui()
            pyautogui.write(text, interval=0.02)
            return ToolResult(True, f"Yazildi: {len(text)} karakter")
        except Exception as e:
            return ToolResult(False, str(e))


class PressKeyTool(Tool):
    name = "press_key"
    description = "Bir klavye tusuna basar (enter, tab, esc, ctrl, alt gibi)."
    requires_permission = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Tus adi (enter, tab, esc, space...)"},
            },
            "required": ["key"],
        }

    async def execute(self, key: str) -> ToolResult:
        try:
            pyautogui = _lazy_pyautogui()
            pyautogui.press(key)
            return ToolResult(True, f"Tusa basildi: {key}")
        except Exception as e:
            return ToolResult(False, str(e))


class OpenProgramTool(Tool):
    name = "open_program"
    description = "Bir programi baslatir (ornek: notepad, calc, chrome)."
    requires_permission = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "Program adi veya yolu"},
            },
            "required": ["program"],
        }

    async def execute(self, program: str) -> ToolResult:
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", "", program], shell=False)
            else:
                subprocess.Popen([program])
            return ToolResult(True, f"Program baslatildi: {program}")
        except Exception as e:
            return ToolResult(False, str(e))


class ScreenInfoTool(Tool):
    name = "screen_info"
    description = "Ekran boyutunu ve mevcut fare konumunu doner."
    requires_permission = True

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self) -> ToolResult:
        try:
            pyautogui = _lazy_pyautogui()
            width, height = pyautogui.size()
            x, y = pyautogui.position()
            return ToolResult(True, f"Ekran: {width}x{height}, Fare: ({x}, {y})")
        except Exception as e:
            return ToolResult(False, str(e))


def all_computer_tools(settings: ToolSettings) -> list[Tool]:
    """Tüm bilgisayar kontrol araçlarını döner."""
    return [
        ScreenshotTool(settings),
        ClickTool(),
        TypeTextTool(),
        PressKeyTool(),
        OpenProgramTool(),
        ScreenInfoTool(),
    ]
