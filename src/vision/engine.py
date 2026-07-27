"""
Vision (görüntü analizi) motoru.

Multimodal LLM (qwen2.5vl) ile görüntü analizi yapar:
- describe: Görüntüyü açıkla
- ocr: Görüntüdeki metni çıkar
- ask: Görüntü hakkında özel soru sor

Neden ayrı bir OCR kütüphanesi (Tesseract) kullanmıyoruz?
- qwen2.5vl zaten OCR yapabiliyor ve Türkçe metinde iyi.
- Ekstra bağımlılık ve kurulum (Tesseract binary) gerektirmez.
- Aynı model hem OCR hem genel analiz yapıyor — tek noktadan yönetim.

Görüntüler base64'e çevrilip Ollama'ya gönderilir.
"""

import base64
from collections.abc import AsyncGenerator
from pathlib import Path

from src.config.settings import VisionSettings
from src.core import LLMClient


# OCR için özel talimat
_OCR_PROMPT = (
    "Bu goruntudeki TUM metni oldugu gibi cikar ve yaz. "
    "Sadece metni ver, yorum ekleme. Metin yoksa 'Metin bulunamadi' de."
)


class VisionEngine:
    """Görüntü analizi motoru."""

    def __init__(self, settings: VisionSettings, llm_client: LLMClient) -> None:
        self._settings = settings
        self._llm = llm_client

    @staticmethod
    def image_to_base64(image_path: str | Path) -> str:
        """
        Bir görüntü dosyasını base64 string'e çevirir.

        Ollama data URI öneki ('data:image/png;base64,') istemez,
        sadece ham base64 ister.
        """
        path = Path(image_path)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def analyze(
        self,
        image_path: str | Path,
        prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Bir görüntüyü verilen talimatla analiz eder (streaming).

        Args:
            image_path: Analiz edilecek görüntü
            prompt: Talimat (None ise varsayılan kullanılır)

        Yields:
            Analiz metni parçaları
        """
        image_b64 = self.image_to_base64(image_path)
        question = prompt or self._settings.default_prompt
        async for chunk in self._llm.stream_vision(
            prompt=question,
            images_base64=[image_b64],
            model=self._settings.model,
        ):
            yield chunk

    async def ocr(self, image_path: str | Path) -> str:
        """Görüntüdeki metni çıkarır (tam sonuç)."""
        image_b64 = self.image_to_base64(image_path)
        chunks: list[str] = []
        async for chunk in self._llm.stream_vision(
            prompt=_OCR_PROMPT,
            images_base64=[image_b64],
            model=self._settings.model,
        ):
            chunks.append(chunk)
        return "".join(chunks)


_instance: VisionEngine | None = None


def get_vision_engine() -> VisionEngine:
    """Singleton VisionEngine döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        settings = get_settings()
        _instance = VisionEngine(settings.vision, LLMClient(settings.llm))
    return _instance
