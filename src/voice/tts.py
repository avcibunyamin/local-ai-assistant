"""
Sesli yanıt (Text-to-Speech) modülü.

pyttsx3 kullanır: Windows'un yerleşik SAPI5 motorunu kullanır.

Neden pyttsx3?
- Tamamen offline, ekstra model indirme yok.
- Windows'ta hazır Türkçe sesler (varsa) kullanılabilir.
- Hafif ve bağımlılığı az.

Not: Daha doğal ses için ileride piper-tts eklenebilir (plugin mimarisi).

Tasarım:
- pyttsx3 engine'i her çağrıda yeniden oluşturulur — SAPI5'te aynı
  engine'i tekrar kullanmak Windows'ta bazen takılmaya yol açar.
- synthesize() senkron; UI'da asyncio.to_thread ile sarılır.
"""

from pathlib import Path

from src.config.settings import ROOT_DIR, VoiceSettings


class TextToSpeech:
    """pyttsx3 tabanlı sesli yanıt üretimi."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._output_dir = ROOT_DIR / "data" / "tts_output"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, filename: str = "response.wav") -> str:
        """
        Metni sese çevirir ve WAV dosyası olarak kaydeder.

        Args:
            text: Seslendirilecek metin
            filename: Çıktı dosya adı

        Returns:
            Oluşturulan ses dosyasının yolu
        """
        import pyttsx3

        output_path = self._output_dir / filename

        engine = pyttsx3.init()
        engine.setProperty("rate", self._settings.tts_rate)
        # Türkçe ses varsa seç
        self._select_turkish_voice(engine)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        engine.stop()

        return str(output_path)

    def _select_turkish_voice(self, engine) -> None:
        """Mevcut sesler arasında Türkçe olanı seçmeye çalışır."""
        try:
            for voice in engine.getProperty("voices"):
                langs = " ".join(str(v) for v in getattr(voice, "languages", []))
                haystack = f"{voice.id} {voice.name} {langs}".lower()
                if "tr" in haystack or "turk" in haystack:
                    engine.setProperty("voice", voice.id)
                    return
        except Exception:
            pass  # Türkçe ses yoksa varsayılan kullanılır

    def list_voices(self) -> list[str]:
        """Sistemde kurulu sesleri listeler."""
        import pyttsx3
        engine = pyttsx3.init()
        voices = [v.name for v in engine.getProperty("voices")]
        engine.stop()
        return voices


_instance: TextToSpeech | None = None


def get_tts() -> TextToSpeech:
    """Singleton TextToSpeech döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        _instance = TextToSpeech(get_settings().voice)
    return _instance
