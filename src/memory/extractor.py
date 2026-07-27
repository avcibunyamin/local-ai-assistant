"""
Hafıza çıkarıcı (fact extraction) ve özetleyici.

LLM'i kullanarak:
1. extract_facts(): Bir konuşmadan hatırlanmaya değer kalıcı gerçekleri çıkarır.
2. summarize(): Uzun bir konuşmayı kısa bir özete indirger.

Neden LLM ile?
- Kural tabanlı çıkarım (regex vs.) doğal dilde işe yaramaz.
- LLM "önemli olanı" bağlamdan anlayabilir.
- Aynı LLM zaten çalışıyor, ekstra bir modele gerek yok.

Tasarım notu:
- Çıkarım JSON formatında istenir ki güvenilir şekilde ayrıştırılabilsin.
- LLM bazen JSON'u markdown kod bloğunda döner; bunu temizliyoruz.
"""

import json

from src.core import LLMClient


_EXTRACT_PROMPT = """Asagidaki konusmadan, kullanici hakkinda GELECEKTE HATIRLANMASI
gereken kalici gerceleri cikar. Sadece kalici ve onemli bilgileri al:
isim, tercihler, meslek, sahip oldugu seyler, hedefler gibi.

Gecici veya onemsiz seyleri ALMA (ornegin "merhaba dedi", "tesekkur etti").

Cevabini SADECE su JSON formatinda ver, baska hicbir sey yazma:
{{"facts": ["gercek 1", "gercek 2"]}}

Eger hatirlanmaya deger hicbir sey yoksa:
{{"facts": []}}

KONUSMA:
{conversation}"""


_SUMMARIZE_PROMPT = """Asagidaki konusmayi Turkce olarak kisaca ozetle.
Ana konulari ve onemli noktalari 2-3 cumlede topla.

KONUSMA:
{conversation}

OZET:"""


class MemoryExtractor:
    """LLM tabanlı gerçek çıkarıcı ve özetleyici."""

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        self._llm = llm_client
        self._model = model

    async def extract_facts(self, messages: list[dict[str, str]]) -> list[str]:
        """
        Konuşmadan kalıcı gerçekleri çıkarır.

        Args:
            messages: [{"role": "user/assistant", "content": "..."}]

        Returns:
            Hatırlanacak gerçekler listesi (boş olabilir)
        """
        conversation = self._format_conversation(messages)
        prompt = _EXTRACT_PROMPT.format(conversation=conversation)

        response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.1,  # düşük sıcaklık = tutarlı JSON
        )
        return self._parse_facts(response)

    async def summarize(self, messages: list[dict[str, str]]) -> str:
        """Konuşmayı kısa bir özete indirger."""
        conversation = self._format_conversation(messages)
        prompt = _SUMMARIZE_PROMPT.format(conversation=conversation)
        return await self._llm.chat(
            [{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.3,
        )

    def _format_conversation(self, messages: list[dict[str, str]]) -> str:
        """Mesajları düz metin konuşmaya çevirir."""
        lines: list[str] = []
        for msg in messages:
            role = "Kullanici" if msg.get("role") == "user" else "Asistan"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _parse_facts(self, response: str) -> list[str]:
        """LLM cevabından JSON gerçekleri ayrıştırır."""
        text = response.strip()
        # Markdown kod bloğunu temizle
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.removeprefix("json").strip()
            text = text.strip("`").strip()

        try:
            data = json.loads(text)
            facts = data.get("facts", [])
            return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
        except (json.JSONDecodeError, AttributeError):
            return []
