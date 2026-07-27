"""
Plugin yükleyici.

Kullanıcının `plugins/` klasörüne koyduğu kendi araçlarını
otomatik keşfeder ve registry'ye ekler.

Nasıl çalışır?
- plugins/ klasöründeki her .py dosyası import edilir.
- İçindeki Tool alt sınıfları bulunur.
- Argümansız oluşturulabilenler registry'ye kaydedilir.

Neden bu tasarım?
- Tool sınıfı (Aşama 5) zaten soyut bir arayüz sunuyor (Open/Closed).
- Yeni yetenek eklemek için ana kodu değiştirmeye gerek yok —
  sadece plugins/ klasörüne bir dosya bırakmak yeterli.

Örnek plugin: plugins/example_tool.py
"""

import importlib.util
import inspect
from pathlib import Path

from src.config.settings import ROOT_DIR
from src.tools.base import Tool
from src.tools.registry import ToolRegistry


PLUGINS_DIR = ROOT_DIR / "plugins"


def load_plugins(registry: ToolRegistry) -> list[str]:
    """
    plugins/ klasöründeki tüm araçları yükler ve kaydeder.

    Args:
        registry: Araçların ekleneceği registry

    Returns:
        Yüklenen araç isimleri
    """
    if not PLUGINS_DIR.exists():
        PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
        return []

    loaded: list[str] = []
    for py_file in PLUGINS_DIR.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        tools = _load_tools_from_file(py_file)
        for tool in tools:
            registry.register(tool)
            loaded.append(tool.name)
    return loaded


def _load_tools_from_file(py_file: Path) -> list[Tool]:
    """Bir Python dosyasından Tool alt sınıflarını yükler ve örnekler."""
    module_name = f"plugin_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return []  # Bozuk plugin diğerlerini engellemesin

    tools: list[Tool] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        # Tool alt sınıfı ama Tool'un kendisi değil, ve bu modülde tanımlı
        if issubclass(obj, Tool) and obj is not Tool and obj.__module__ == module_name:
            try:
                tools.append(obj())  # argümansız oluşturulabilenler
            except TypeError:
                continue  # ayar gerektiren araçlar plugin olamaz
    return tools
