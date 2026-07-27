"""
Konuşma tanıma (Speech-to-Text) modülü.

faster-whisper kullanır: OpenAI Whisper'ın CTranslate2 ile
optimize edilmiş, 4x daha hızlı lokal versiyonu.

Neden faster-whisper?
- Tamamen lokal, internet gerektirmez.
- Türkçe dahil 99 dil destekler.
- int8 quantization ile CPU'da bile hızlı ve az RAM kullanır.
- Model ilk kullanımda otomatik iner (HuggingFace'den), sonra cache'lenir.

Tasarım:
- Model yükleme pahalıdır (~1-2 sn), bu yüzden lazy-load + singleton.
- transcribe() senkron çalışır; UI'da asyncio.to_thread ile sarılır
  ki event loop bloklanmasin.
"""

from pathlib import Path

from src.config.settings import VoiceSettings


class SpeechToText:
    """faster-whisper tabanlı konuşma tanıma."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._model = None  # lazy-load

    def _ensure_model(self):
        """Modeli ilk kullanımda yükler (lazy-load)."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._settings.stt_model,
                device="cpu",
                compute_type=self._settings.stt_compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Bir ses dosyasını metne çevirir.

        Args:
            audio_path: Ses dosyası yolu (wav, mp3, m4a...)

        Returns:
            Tanınan metin
        """
        model = self._ensure_model()
        language = self._settings.stt_language or None
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


_instance: SpeechToText | None = None


def get_stt() -> SpeechToText:
    """Singleton SpeechToText döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        _instance = SpeechToText(get_settings().voice)
    return _instance
