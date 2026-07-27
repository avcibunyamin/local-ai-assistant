"""
Görev planlayıcı (Planner).

Bir kullanıcı görevini, sırayla yürütülebilecek küçük adımlara böler.

Neden planlama?
- Karmaşık görevler tek adımda çözülemez ("bir hesap makinesi yaz ve test et").
- Adımlara bölünce her adım basit, yürütülebilir ve doğrulanabilir olur.
- Bu "Plan-and-Execute" desenidir: önce planla, sonra yürüt.

Plan JSON formatında istenir ki güvenilir ayrıştırılabilsin.
"""

import json

from src.core import LLMClient


_PLAN_PROMPT = """Sen bir gorev planlayicisisin. Kullanicinin verdigi gorevi,
sirayla yapilacak KUCUK ve NET adimlara bol.

Kurallar:
- Her adim tek bir eylem olsun (dosya olustur, kod yaz, test et gibi).
- ONEMLI: Ayni isi iki farkli adimda YAPMA. Bir dosyaya yazmak icin
  TEK bir adim ve TEK bir arac kullan (ya write_file YA DA run_python, ikisi degil).
- ONEMLI: Basit gorevleri az adima bol. Gereksiz adim ekleme.
  Ornek: "bir dosyaya X yaz" -> TEK adim yeterli, klasor olusturma gerekmez.
- Ayni dosya yolunu her adimda ayni yaz (alt klasor uydurma).
- Adimlar mantikli sirada olsun.
- En fazla 4 adim olustur.
- Kullanilabilir araclar: {tools}

Cevabini SADECE su JSON formatinda ver:
{{"steps": ["adim 1", "adim 2"]}}

GOREV:
{task}"""


class Planner:
    """LLM tabanlı görev planlayıcı."""

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        self._llm = llm_client
        self._model = model

    async def plan(self, task: str, tool_names: list[str]) -> list[str]:
        """
        Görevi adımlara böler.

        Args:
            task: Kullanıcının görevi
            tool_names: Kullanılabilir araç isimleri

        Returns:
            Adım açıklamaları listesi
        """
        prompt = _PLAN_PROMPT.format(
            task=task,
            tools=", ".join(tool_names) if tool_names else "yok",
        )
        response = await self._llm.chat(
            [{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.2,
        )
        return self._parse_steps(response)

    def _parse_steps(self, response: str) -> list[str]:
        """LLM cevabından adımları ayrıştırır."""
        text = response.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            text = text.removeprefix("json").strip()

        try:
            data = json.loads(text)
            steps = data.get("steps", [])
            return [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, AttributeError):
            # JSON ayrıştırılamazsa tüm görevi tek adım yap
            return []
