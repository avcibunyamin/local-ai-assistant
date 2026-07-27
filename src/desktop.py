"""
Local AI Assistant - Masaüstü Uygulaması.

Uygulamayı tarayıcı yerine yerel bir masaüstü penceresinde açar.

Nasıl çalışır?
1. FastAPI + Gradio sunucusu arka planda bir thread'de başlatılır.
2. Sunucu hazır olana kadar /api/health sorgulanır (bekleme).
3. pywebview ile yerel bir pencere açılıp UI gösterilir.
   (Windows'ta Edge WebView2 kullanılır — zaten sistemde yüklü.)

Neden pywebview (Electron değil)?
- Electron ~150 MB Node/Chromium taşır; pywebview sistemin mevcut
  WebView2'sini kullanır → küçük, hızlı, ekstra bağımlılık az.
- Sunucu mantığı aynen korunur (DRY): sadece görüntüleme katmanı değişir.

Çalıştırma:
    python -m src.desktop
"""

import threading
import time

import httpx
import uvicorn
import webview

from src.config import get_settings
from src.main import create_app


class _ServerThread(uvicorn.Server):
    """
    Ana thread dışında çalışabilen uvicorn sunucusu.

    uvicorn normalde sinyal işleyicilerini (Ctrl+C) yalnızca ana thread'de
    kurabilir; arka plan thread'inde bu hata verir. Bu yüzden sinyal kurulumunu
    devre dışı bırakıyoruz (pencere kapanınca zaten süreç sonlanır).
    """

    def install_signal_handlers(self) -> None:  # noqa: D102
        pass


def _wait_until_ready(url: str, timeout: float = 30.0) -> bool:
    """Sunucu /api/health 200 dönene kadar bekler."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main() -> None:
    """Sunucuyu arka planda başlatır ve masaüstü penceresini açar."""
    settings = get_settings()
    port = settings.server.port

    # 1. Sunucuyu arka plan thread'inde başlat (127.0.0.1 = yalnızca yerel)
    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = _ServerThread(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # 2. Hazır olmasını bekle
    health_url = f"http://127.0.0.1:{port}/api/health"
    if not _wait_until_ready(health_url):
        print("Sunucu zamanında başlamadı. Ollama çalışıyor mu?")
        return

    # 3. Masaüstü penceresini aç
    webview.create_window(
        settings.ui.title,
        f"http://127.0.0.1:{port}/chat",
        width=1280,
        height=860,
        min_size=(900, 600),
    )
    # webview.start() ana thread'i bloklar; pencere kapanınca döner.
    webview.start()

    # 4. Pencere kapandı → sunucuyu durdur (daemon thread zaten sonlanır)
    server.should_exit = True


if __name__ == "__main__":
    main()
