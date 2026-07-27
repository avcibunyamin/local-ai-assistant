"""
Multi-Agent sistemi (Plan-and-Execute).

Bir görevi baştan sona otonom yürüten agent:
1. Planner ile görevi adımlara böler.
2. Her adımı ToolExecutor ile yürütür (araçları kullanarak).
3. Adım sonuçlarını biriktirir; sonraki adım önceki sonuçları görür.
4. Sonunda tüm işi özetleyen bir final cevap üretir.

Neden mevcut ToolExecutor'ı yeniden kullanıyoruz?
- Araç çağırma mantığı (Aşama 5) zaten yazılı ve test edilmiş.
- DRY: agent sadece "planlama + orkestrasyon" katmanı ekler.

İlerleme, async generator ile adım adım UI'a akıtılır.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from src.agents.planner import Planner
from src.core import LLMClient
from src.tools.executor import ToolExecutor


@dataclass
class AgentUpdate:
    """Agent yürütmesi sırasında bir ilerleme güncellemesi."""

    kind: str  # "plan" | "step" | "final"
    content: str


_STEP_PROMPT = """Sen araclari kullanarak gorevleri tamamlayan bir agentsin.

ANA GOREV: {task}

PLAN:
{plan}

Simdiye kadar tamamlanan adimlar ve sonuclari:
{history}

SIMDI SU ADIMI YAP: {current_step}

Kurallar:
- Sadece bu adima odaklan, sonraki adimlari yapma.
- ONCEKI adimlarda zaten yapilan bir isi TEKRARLAMA (ornegin dosya zaten
  yazildiysa tekrar yazma).
- Bir araci en fazla 1 kez cagir; ayni araci ust uste tekrar cagirma.
- Adim tamamlandiysa kisa bir onay yaz, gereksiz aciklamalar yapma.
- SADECE Turkce yaz. Cince, Ingilizce veya baska dilden kelime/karakter kullanma."""


_FINAL_PROMPT = """Sen bir gorevi tamamladin.

ANA GOREV: {task}

Yapilan adimlar ve sonuclari:
{history}

Simdi kullaniciya, gorevin sonucunu kisaca ozetle.
Ne yaptigini ve sonucu acikla.
ONEMLI: SADECE Turkce yaz. Cince, Ingilizce veya baska dilden kelime ya da
karakter KULLANMA. Tum cevap tamamen Turkce olsun."""


class Agent:
    """Plan-and-Execute otonom agent."""

    def __init__(
        self,
        llm_client: LLMClient,
        planner: Planner,
        executor: ToolExecutor,
    ) -> None:
        self._llm = llm_client
        self._planner = planner
        self._executor = executor

    async def run(
        self,
        task: str,
        model: str | None = None,
    ) -> AsyncGenerator[AgentUpdate, None]:
        """
        Bir görevi otonom yürütür, ilerlemeyi adım adım akıtır.

        Args:
            task: Yürütülecek görev
            model: Kullanılacak model

        Yields:
            AgentUpdate: plan, her adım ve final özet
        """
        # 1. PLANLA
        tool_names = [t.name for t in self._executor._registry.all()]
        steps = await self._planner.plan(task, tool_names)

        if not steps:
            # Plan çıkarılamadı — görevi doğrudan yürüt
            steps = [task]

        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        yield AgentUpdate("plan", plan_text)

        # 2. HER ADIMI YÜRÜT
        history_lines: list[str] = []
        for i, step in enumerate(steps, 1):
            history_text = "\n".join(history_lines) if history_lines else "(henuz yok)"
            prompt = _STEP_PROMPT.format(
                task=task,
                plan=plan_text,
                history=history_text,
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                result = await self._executor.run(messages, model=model)
                step_result = result.content
                # Araç kullanıldıysa belirt
                if result.steps:
                    tools_used = ", ".join(s.tool_name for s in result.steps)
                    step_result = f"[Araclar: {tools_used}] {step_result}"
            except Exception as e:
                step_result = f"HATA: {e}"

            history_lines.append(f"Adim {i} ({step}): {step_result}")
            yield AgentUpdate("step", f"**Adim {i}/{len(steps)}:** {step}\n{step_result}")

        # 3. FINAL ÖZET
        final_prompt = _FINAL_PROMPT.format(
            task=task,
            history="\n".join(history_lines),
        )
        summary = await self._llm.chat(
            [{"role": "user", "content": final_prompt}],
            model=model,
        )
        yield AgentUpdate("final", summary)


_instance: Agent | None = None


def get_agent() -> Agent:
    """Singleton Agent döner."""
    global _instance
    if _instance is None:
        from src.config import get_settings
        from src.tools.executor import get_tool_executor
        settings = get_settings()
        client = LLMClient(settings.llm)
        _instance = Agent(
            client,
            Planner(client),
            get_tool_executor(),
        )
    return _instance
