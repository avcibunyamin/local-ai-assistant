"""
Araç yürütme motoru (Tool Executor).

Ollama tool-calling döngüsünü yönetir:
1. Kullanıcı mesajı + araç şemaları LLM'e gönderilir.
2. LLM araç çağrısı (tool_calls) döndürürse, araçlar çalıştırılır.
3. Araç sonuçları LLM'e geri gönderilir.
4. LLM nihai cevabı ("content", tool_calls yok) verene kadar döngü sürer.

max_iterations ile sonsuz döngü engellenir.

Döngü boyunca her adım bir "iz" (trace) olarak kaydedilir; UI'da
"AI hangi aracı neden çağırdı" görünür.
"""

import json
from dataclasses import dataclass, field

from src.core import LLMClient
from src.tools.registry import ToolRegistry


@dataclass
class ToolStep:
    """Tool-calling döngüsünde bir adım (izleme için)."""

    tool_name: str
    arguments: dict
    result: str


@dataclass
class ToolResponse:
    """Tool executor'ın nihai sonucu."""

    content: str
    steps: list[ToolStep] = field(default_factory=list)


class ToolExecutor:
    """Araç destekli LLM döngüsünü yürütür."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 5,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._max_iterations = max_iterations

    async def run(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> ToolResponse:
        """
        Araç destekli bir konuşma turu yürütür.

        Args:
            messages: Konuşma geçmişi (sistem + kullanıcı mesajları)
            model: Kullanılacak model

        Returns:
            Nihai cevap ve atılan araç adımları
        """
        tools = self._registry.schemas()
        if not tools:
            # Araç yoksa normal cevap dön
            content = await self._llm.chat(messages, model=model)
            return ToolResponse(content=content)

        work_messages = list(messages)
        steps: list[ToolStep] = []
        # Yürütülen (araç, argüman) çağrılarının imzaları — tekrarı önlemek için.
        # LLM aynı yazma işlemini tekrar isterse kod düzeyinde engelleriz.
        executed: set[str] = set()

        for _ in range(self._max_iterations):
            message = await self._llm.chat_with_tools(work_messages, tools, model=model)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # Araç çağrısı yok = nihai cevap
                return ToolResponse(content=message.get("content", ""), steps=steps)

            # Asistanın tool çağrısı mesajını geçmişe ekle
            work_messages.append(message)

            # Her araç çağrısını çalıştır
            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                # Tekrar koruması: aynı araç + argüman ikinci kez çağrılırsa atla
                signature = f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
                if signature in executed:
                    dup_msg = "Bu islem zaten yapildi, tekrar edilmedi."
                    work_messages.append({
                        "role": "tool", "content": dup_msg, "tool_name": name,
                    })
                    continue
                executed.add(signature)

                result = await self._registry.execute(name, arguments)
                steps.append(ToolStep(
                    tool_name=name,
                    arguments=arguments,
                    result=result.to_message(),
                ))

                # Araç sonucunu LLM'e geri ver
                work_messages.append({
                    "role": "tool",
                    "content": result.to_message(),
                    "tool_name": name,
                })

        # Maksimum tur aşıldı — son bir özet cevap iste
        work_messages.append({
            "role": "user",
            "content": "Arac kullanimini birak ve simdiye kadarki sonuclara gore cevap ver.",
        })
        content = await self._llm.chat(work_messages, model=model)
        return ToolResponse(content=content, steps=steps)


_instance: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """Singleton ToolExecutor döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        from src.tools.registry import get_registry
        settings = get_settings()
        _instance = ToolExecutor(
            LLMClient(settings.llm),
            get_registry(),
            max_iterations=settings.tools.max_iterations,
        )
    return _instance
