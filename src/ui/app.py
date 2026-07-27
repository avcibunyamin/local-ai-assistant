"""
Gradio Chat Arayüzü - Aşama 2.

Yeni özellikler:
- Sol panelde konuşma geçmişi listesi
- Yeni sohbet oluşturma
- Konuşmalar arası geçiş
- Konuşma silme
- Model seçici (Ollama'daki yüklü modeller)
- Chat geçmişi SQLite'da kalıcı

Neden Blocks kullanıyoruz (ChatInterface yerine)?
- ChatInterface hazır ama özelleştirme sınırlı.
- Sol panel, model seçici gibi ekstra bileşenler eklemek için
  Blocks ile kendi layout'umuzu kurmamız gerekiyor.
- Chatbot bileşeni streaming'i doğrudan destekliyor.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import gradio as gr

from src.chat.history import get_chat_history
from src.chat.exporter import export_conversation
from src.config import get_settings
from src.core import LLMClient
from src.memory.manager import get_memory_manager
from src.rag.engine import get_rag_engine
from src.tools.executor import get_tool_executor
from src.tools.registry import get_registry
from src.vision.engine import get_vision_engine
from src.voice.stt import get_stt
from src.voice.tts import get_tts
from src.agents.agent import get_agent

# Sistem promptu override'ının saklandığı tercih anahtarı.
# Kullanıcı UI'dan özel bir prompt girerse burada kalıcı saklanır.
_SYSTEM_PROMPT_PREF_KEY = "system_prompt_override"


def _effective_system_prompt() -> str:
    """
    Aktif sistem promptunu döner.

    Öncelik: kullanıcının UI'dan kaydettiği override > settings.yaml varsayılanı.
    Böylece kullanıcı, dosya düzenlemeden asistanın kişiliğini değiştirebilir.
    """
    override = get_memory_manager().get_preference(_SYSTEM_PROMPT_PREF_KEY)
    if override and override.strip():
        return override
    return get_settings().llm.system_prompt


async def _list_ollama_models() -> list[str]:
    """Ollama'daki yüklü modelleri listeler."""
    settings = get_settings()
    client = LLMClient(settings.llm)
    try:
        models = await client.list_models()
        return [m["name"] for m in models]
    except Exception:
        return [settings.llm.model]


def _load_conversation_list() -> list[list]:
    """Konuşma listesini Dataframe formatında döner."""
    history = get_chat_history()
    convs = history.list_conversations()
    if not convs:
        return []
    return [[c["id"], c["title"]] for c in convs]


def _load_messages(conversation_id: int | None) -> list[dict]:
    """Konuşma mesajlarını Gradio chatbot formatında döner."""
    if not conversation_id:
        return []
    history = get_chat_history()
    messages = history.get_messages(conversation_id)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]


def _format_sources() -> str:
    """Yüklü belge listesini Markdown olarak döner."""
    engine = get_rag_engine()
    sources = engine.list_sources()
    if not sources:
        return "_Henüz belge yüklenmedi._"
    lines = "\n".join(f"- {s}" for s in sources)
    return f"**Yüklü belgeler ({engine.count()} parça):**\n{lines}"


def _format_memories() -> str:
    """Kaydedilmiş hafızaları Markdown olarak döner."""
    mgr = get_memory_manager()
    memories = mgr.list_memories()
    if not memories:
        return "_Henüz hafıza yok._"
    lines = "\n".join(f"- {m.text}" for m in memories[:15])
    suffix = f"\n_...ve {len(memories) - 15} tane daha_" if len(memories) > 15 else ""
    return f"**Hatırlananlar ({len(memories)}):**\n{lines}{suffix}"


def _format_tools() -> str:
    """Kullanılabilir araçları Markdown olarak döner."""
    registry = get_registry()
    names = registry.names()
    if not names:
        return "_Araç yok._"
    lines = "\n".join(f"- `{n}`" for n in names)
    note = ""
    if not get_settings().tools.allow_dangerous:
        note = "\n\n_Not: terminal/python için settings.yaml'da `allow_dangerous: true` yapın._"
    return f"**Kullanılabilir araçlar:**\n{lines}{note}"


# RAG bağlamıyla zenginleştirilmiş sistem promptu şablonu
_RAG_PROMPT_TEMPLATE = """{system_prompt}

Asagida kullanicinin belgelerinden alinan ilgili bilgiler var.
Sorulari MUTLAKA bu bilgilere dayanarak cevapla. Eger cevap bu
bilgilerde yoksa, bunu belirt.

=== BELGELERDEN ILGILI BILGILER ===
{context}
=== BILGILER SONU ==="""


def create_ui() -> gr.Blocks:
    """Gelişmiş Gradio arayüzünü oluşturur."""
    settings = get_settings()

    # Claude tarzı sıcak tema: turuncu accent, taş/krem nötrler
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
    )

    with gr.Blocks(
        theme=theme,
        title=settings.ui.title,
        css=_custom_css(),
    ) as app:

        # Aktif konuşma ID'sini tutan state
        active_conv_id = gr.State(value=None)

        with gr.Row(elem_classes="title-row"):
            gr.Markdown(f"# ✦ {settings.ui.title}", elem_classes="app-title")
            theme_btn = gr.Button("🌙", elem_classes="theme-toggle", size="sm")

        with gr.Row(elem_classes="main-row"):
            # ===== SOL PANEL: Kenar Çubuğu =====
            with gr.Column(scale=1, min_width=280, elem_classes="sidebar"):
                new_chat_btn = gr.Button("＋  Yeni Sohbet", variant="primary", size="lg")

                model_dropdown = gr.Dropdown(
                    choices=[settings.llm.model],
                    value=settings.llm.model,
                    label="Model",
                    interactive=True,
                )

                gr.Markdown("###  Geçmiş", elem_classes="section-label")
                search_box = gr.Textbox(
                    placeholder="🔍  Sohbetlerde ara (başlık + mesaj)…",
                    show_label=False,
                    container=False,
                    elem_classes="search-box",
                )
                conversation_list = gr.Dataframe(
                    headers=["ID", "Başlık"],
                    datatype=["number", "str"],
                    value=_load_conversation_list,
                    col_count=(2, "fixed"),
                    interactive=False,
                    show_label=False,
                    max_height=280,
                )
                with gr.Row(elem_classes="history-actions"):
                    delete_btn = gr.Button("🗑  Sil", variant="stop", size="sm")
                    export_format = gr.Dropdown(
                        choices=["json", "txt", "md"],
                        value="md",
                        show_label=False,
                        container=False,
                        scale=1,
                    )
                    export_btn = gr.Button("⬇  Dışa Aktar", size="sm")
                export_file = gr.File(label="İndirilecek dosya", visible=False)

                # ===== Katlanabilir yetenek panelleri (Claude tarzı) =====
                with gr.Accordion("📄  Belgeler (RAG)", open=False):
                    rag_toggle = gr.Checkbox(
                        label="Belgelere göre cevapla",
                        value=False,
                    )
                    file_upload = gr.File(
                        label="Belge yükle",
                        file_types=[
                            ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md",
                            ".py", ".js", ".ts", ".json", ".html", ".css",
                        ],
                        file_count="multiple",
                    )
                    upload_status = gr.Markdown("")
                    sources_display = gr.Markdown(_format_sources())
                    clear_docs_btn = gr.Button("Belgeleri Temizle", variant="stop", size="sm")

                with gr.Accordion("🧠  Hafıza", open=False):
                    memory_toggle = gr.Checkbox(
                        label="Hafızayı kullan (hatırla + geri çağır)",
                        value=True,
                    )
                    memory_display = gr.Markdown(_format_memories())
                    clear_memory_btn = gr.Button("Hafızayı Temizle", variant="stop", size="sm")

                with gr.Accordion("🛠  Araçlar & Agent", open=False):
                    tools_toggle = gr.Checkbox(
                        label="Araçları kullan (dosya, terminal...)",
                        value=False,
                    )
                    agent_toggle = gr.Checkbox(
                        label="Agent Modu (planla + yürüt + özetle)",
                        value=False,
                    )
                    tools_display = gr.Markdown(_format_tools())

                with gr.Accordion("👁  Görüntü Analizi", open=False):
                    image_input = gr.Image(
                        label="Görüntü yükle",
                        type="filepath",
                        height=180,
                    )
                    image_question = gr.Textbox(
                        placeholder="Görüntü hakkında soru (boş = genel analiz)",
                        show_label=False,
                        lines=2,
                    )
                    analyze_btn = gr.Button("Görüntüyü Analiz Et", variant="primary", size="sm")

                with gr.Accordion("🎙  Ses", open=False):
                    speak_toggle = gr.Checkbox(
                        label="Sesli yanıt (AI cevabını seslendir)",
                        value=False,
                    )
                    mic_input = gr.Audio(
                        label="Mikrofon (konuş → metne çevir)",
                        sources=["microphone"],
                        type="filepath",
                    )
                    transcribe_btn = gr.Button("Sese Çevir → Mesaj Kutusuna", size="sm")

                with gr.Accordion("⚙️  Sistem Promptu", open=False):
                    gr.Markdown(
                        "_Asistanın kişiliğini ve kurallarını belirler. "
                        "Boş bırakıp kaydedersen varsayılana döner._",
                        elem_classes="hint-text",
                    )
                    system_prompt_box = gr.Textbox(
                        value=_effective_system_prompt(),
                        lines=6,
                        show_label=False,
                        placeholder="Örn: Sen kısa ve öz cevap veren bir asistansın…",
                    )
                    with gr.Row():
                        save_prompt_btn = gr.Button(
                            "💾  Kaydet", variant="primary", size="sm"
                        )
                        reset_prompt_btn = gr.Button("↺  Varsayılana Dön", size="sm")
                    prompt_status = gr.Markdown("")

            # ===== SAĞ PANEL: Sohbet Alanı =====
            with gr.Column(scale=3, elem_classes="chat-panel"):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=560,
                    show_label=False,
                    show_copy_button=True,
                    render_markdown=True,
                    avatar_images=(None, None),
                    placeholder="Bir mesaj yazın ya da soldaki bir yeteneği açın…",
                )

                with gr.Row(elem_classes="input-row"):
                    msg_input = gr.MultimodalTextbox(
                        placeholder="Mesajınızı yazın ya da bir belgeyi buraya sürükleyin…",
                        show_label=False,
                        scale=9,
                        container=False,
                        autofocus=True,
                        file_count="multiple",
                        file_types=[
                            ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md",
                            ".py", ".js", ".ts", ".json", ".html", ".css",
                        ],
                    )
                    send_btn = gr.Button("Gönder", variant="primary", scale=1)

                # Sesli yanıt çıktısı (otomatik oynatma)
                audio_output = gr.Audio(
                    label="Sesli Yanıt",
                    autoplay=True,
                    visible=True,
                )

        # ===== EVENT HANDLERS =====

        # Mesaj kutusunu temizlemek için değer (MultimodalTextbox dict bekler)
        clear_input = {"text": "", "files": []}

        # Yeni sohbet
        def on_new_chat():
            history = get_chat_history()
            conv_id = history.create_conversation()
            return conv_id, [], _load_conversation_list(), clear_input

        new_chat_btn.click(
            fn=on_new_chat,
            outputs=[active_conv_id, chatbot, conversation_list, msg_input],
        )

        # Konuşma seçme (dataframe satırına tıklama)
        def on_select_conversation(evt: gr.SelectData, conv_list):
            if evt.index is None or not conv_list:
                return gr.skip(), gr.skip()
            row_idx = evt.index[0] if isinstance(evt.index, list) else evt.index
            if row_idx >= len(conv_list):
                return gr.skip(), gr.skip()
            conv_id = int(conv_list[row_idx][0])
            messages = _load_messages(conv_id)
            return conv_id, messages

        conversation_list.select(
            fn=on_select_conversation,
            inputs=[conversation_list],
            outputs=[active_conv_id, chatbot],
        )

        # Konuşma silme
        def on_delete(conv_id):
            if conv_id:
                history = get_chat_history()
                history.delete_conversation(conv_id)
            return None, [], _load_conversation_list()

        delete_btn.click(
            fn=on_delete,
            inputs=[active_conv_id],
            outputs=[active_conv_id, chatbot, conversation_list],
        )

        # Mesaj gönderme
        async def on_send(
            message,  # MultimodalTextbox: {"text": str, "files": [path]} | str
            chat_history: list[dict],
            conv_id: int | None,
            model: str,
            use_rag: bool,
            use_memory: bool,
            use_tools: bool,
            use_speak: bool,
            use_agent: bool,
        ) -> AsyncGenerator:
            # MultimodalTextbox {text, files} ya da düz metin — normalize et
            if isinstance(message, dict):
                dropped_files = message.get("files", []) or []
                message = message.get("text", "") or ""
            else:
                dropped_files = []

            # Sohbete belge sürüklendiyse: RAG'a ekle ve bu tur RAG'ı aç
            ingested_note = ""
            if dropped_files:
                engine = get_rag_engine()
                names = []
                for fpath in dropped_files:
                    try:
                        await engine.ingest_file(fpath)
                        names.append(Path(fpath).name)
                    except Exception as e:
                        names.append(f"{Path(fpath).name} (hata: {e})")
                use_rag = True
                ingested_note = "📎 _Eklenen belge(ler): " + ", ".join(names) + "_\n\n"
                if not message.strip():
                    message = "Eklediğim belgeleri özetler misin?"

            if not message.strip():
                yield chat_history, conv_id, _load_conversation_list(), clear_input, _format_memories(), None
                return

            history_db = get_chat_history()

            # Konuşma yoksa oluştur
            if not conv_id:
                conv_id = history_db.create_conversation()

            # Kullanıcı mesajını kaydet (sürüklenen belge notuyla birlikte göster)
            display_message = ingested_note + message
            history_db.add_message(conv_id, "user", display_message)
            chat_history = chat_history + [{"role": "user", "content": display_message}]

            # İlk mesajsa otomatik başlık
            msgs = history_db.get_messages(conv_id)
            user_msgs = [m for m in msgs if m["role"] == "user"]
            if len(user_msgs) == 1:
                history_db.auto_title(conv_id)

            # LLM'e gönder
            llm_settings = get_settings().llm
            client = LLMClient(llm_settings)

            # Sistem promptunu bağlamlarla zenginleştir
            # (kullanıcı UI'dan özel prompt kaydettiyse o kullanılır)
            system_parts = [_effective_system_prompt()]

            # Hafıza açıksa: ilgili hafızaları geri çağır
            if use_memory:
                mem_context = await get_memory_manager().build_memory_context(message)
                if mem_context:
                    system_parts.append(
                        "=== HAFIZA (kullanici hakkinda bilinenler) ===\n"
                        f"{mem_context}\n=== HAFIZA SONU ==="
                    )

            # RAG açıksa: belgelerden bağlam getir
            if use_rag:
                engine = get_rag_engine()
                if engine.count() > 0:
                    context = await engine.build_context(message)
                    if context:
                        system_parts.append(
                            "=== BELGELERDEN ILGILI BILGILER ===\n"
                            f"{context}\n=== BILGILER SONU ==="
                        )

            system_content = "\n\n".join(system_parts)

            ollama_messages = [{"role": "system", "content": system_content}]
            for msg in chat_history:
                ollama_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

            chat_history = chat_history + [{"role": "assistant", "content": ""}]

            if use_agent:
                # AGENT MODU: planla + yürüt + özetle
                accumulated = ""
                try:
                    async for update in get_agent().run(message, model=model):
                        if update.kind == "plan":
                            accumulated += f"### 📋 Plan\n{update.content}\n\n"
                        elif update.kind == "step":
                            accumulated += f"{update.content}\n\n"
                        elif update.kind == "final":
                            accumulated += f"### ✅ Sonuc\n{update.content}"
                        chat_history[-1]["content"] = accumulated
                        yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()
                    response = accumulated
                except Exception as e:
                    response = f"Agent hatasi: {e}"
                    chat_history[-1]["content"] = response
                    yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()
            elif use_tools:
                # ARAÇ MODU: tool-calling döngüsü (streaming yok)
                chat_history[-1]["content"] = "_Araclar calisiyor..._"
                yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()
                try:
                    result = await get_tool_executor().run(ollama_messages, model=model)
                    # Atılan araç adımlarını göster
                    steps_md = ""
                    if result.steps:
                        step_lines = "\n".join(
                            f"- `{s.tool_name}` calistirildi" for s in result.steps
                        )
                        steps_md = f"**Kullanilan araclar:**\n{step_lines}\n\n---\n\n"
                    response = steps_md + result.content
                except Exception as e:
                    response = f"Arac hatasi: {e}"
                chat_history[-1]["content"] = response
                yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()
            else:
                # NORMAL MOD: streaming cevap
                response = ""
                try:
                    async for chunk in client.stream_chat(ollama_messages, model=model):
                        response += chunk
                        chat_history[-1]["content"] = response
                        yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()
                except Exception as e:
                    response = f"Hata: {e}"
                    chat_history[-1]["content"] = response
                    yield chat_history, conv_id, _load_conversation_list(), clear_input, gr.skip(), gr.skip()

            # Asistan cevabını kaydet
            history_db.add_message(conv_id, "assistant", response)

            # Hafıza açıksa: bu turdan otomatik gerçek çıkar
            if use_memory:
                await get_memory_manager().learn_from_conversation([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ])

            # Sesli yanıt açıksa: cevabı seslendir
            audio_path = None
            if use_speak and response.strip():
                import asyncio
                try:
                    audio_path = await asyncio.to_thread(get_tts().synthesize, response)
                except Exception:
                    audio_path = None

            yield chat_history, conv_id, _load_conversation_list(), clear_input, _format_memories(), audio_path

        _send_inputs = [msg_input, chatbot, active_conv_id, model_dropdown, rag_toggle, memory_toggle, tools_toggle, speak_toggle, agent_toggle]
        _send_outputs = [chatbot, active_conv_id, conversation_list, msg_input, memory_display, audio_output]

        # Enter ile gönder
        msg_input.submit(fn=on_send, inputs=_send_inputs, outputs=_send_outputs)

        # Buton ile gönder
        send_btn.click(fn=on_send, inputs=_send_inputs, outputs=_send_outputs)

        # ===== RAG EVENT HANDLERS =====

        # Belge yükleme
        async def on_upload(files):
            if not files:
                return "", _format_sources()
            engine = get_rag_engine()
            total_chunks = 0
            uploaded = []
            errors = []
            for file_obj in files:
                try:
                    count = await engine.ingest_file(file_obj.name)
                    total_chunks += count
                    uploaded.append(Path(file_obj.name).name)
                except Exception as e:
                    errors.append(f"{Path(file_obj.name).name}: {e}")

            msg_parts = []
            if uploaded:
                msg_parts.append(f"✓ {len(uploaded)} belge yüklendi ({total_chunks} parça)")
            if errors:
                msg_parts.append("⚠ Hatalar:\n" + "\n".join(errors))
            return "\n\n".join(msg_parts), _format_sources()

        file_upload.upload(
            fn=on_upload,
            inputs=[file_upload],
            outputs=[upload_status, sources_display],
        )

        # Belgeleri temizleme
        def on_clear_docs():
            engine = get_rag_engine()
            engine.clear()
            return "Tüm belgeler silindi.", _format_sources()

        clear_docs_btn.click(
            fn=on_clear_docs,
            outputs=[upload_status, sources_display],
        )

        # ===== HAFIZA EVENT HANDLERS =====

        # Hafızayı temizleme
        def on_clear_memory():
            get_memory_manager().clear_memories()
            return _format_memories()

        clear_memory_btn.click(
            fn=on_clear_memory,
            outputs=[memory_display],
        )

        # ===== VISION EVENT HANDLER =====

        # Görüntü analizi (streaming, chatbot'a yazar)
        async def on_analyze_image(
            image_path: str | None,
            question: str,
            chat_history: list[dict],
            conv_id: int | None,
        ) -> AsyncGenerator:
            if not image_path:
                yield chat_history, conv_id, ""
                return

            history_db = get_chat_history()
            if not conv_id:
                conv_id = history_db.create_conversation()

            # Kullanıcı turunu göster (görüntü + soru)
            q_text = question.strip() or "(görüntüyü analiz et)"
            user_label = f"🖼️ Görüntü analizi: {q_text}"
            history_db.add_message(conv_id, "user", user_label)
            chat_history = chat_history + [
                {"role": "user", "content": user_label},
                {"role": "assistant", "content": ""},
            ]

            engine = get_vision_engine()
            response = ""
            try:
                async for chunk in engine.analyze(image_path, question.strip() or None):
                    response += chunk
                    chat_history[-1]["content"] = response
                    yield chat_history, conv_id, ""
            except Exception as e:
                response = f"Görüntü hatası: {e}\n\n(Model indirildi mi? 'ollama pull qwen2.5vl:7b')"
                chat_history[-1]["content"] = response
                yield chat_history, conv_id, ""

            history_db.add_message(conv_id, "assistant", response)
            yield chat_history, conv_id, ""

        analyze_btn.click(
            fn=on_analyze_image,
            inputs=[image_input, image_question, chatbot, active_conv_id],
            outputs=[chatbot, active_conv_id, image_question],
        )

        # ===== SES EVENT HANDLERS =====

        # Mikrofon kaydını metne çevir (STT) -> mesaj kutusuna yaz
        # MultimodalTextbox olduğu için {"text", "files"} dict'i döneriz.
        async def on_transcribe(audio_path: str | None) -> dict:
            if not audio_path:
                return {"text": "", "files": []}
            import asyncio
            try:
                # STT senkron; event loop'u bloklamamak için thread'e al
                text = await asyncio.to_thread(get_stt().transcribe, audio_path)
                return {"text": text, "files": []}
            except Exception as e:
                return {"text": f"[Ses tanima hatasi: {e}]", "files": []}

        transcribe_btn.click(
            fn=on_transcribe,
            inputs=[mic_input],
            outputs=[msg_input],
        )

        # ===== ARAMA / DIŞA AKTARMA EVENT HANDLERS =====

        # Sohbetlerde arama (yazdıkça filtreler)
        def on_search(query: str) -> list[list]:
            history = get_chat_history()
            convs = history.search_conversations(query)
            return [[c["id"], c["title"]] for c in convs]

        search_box.change(
            fn=on_search,
            inputs=[search_box],
            outputs=[conversation_list],
        )

        # Aktif konuşmayı dışa aktar (JSON/TXT/MD)
        def on_export(conv_id: int | None, fmt: str):
            if not conv_id:
                return gr.update(visible=False)
            history = get_chat_history()
            conv = history.get_conversation(conv_id)
            if not conv:
                return gr.update(visible=False)
            messages = history.get_messages(conv_id)
            path = export_conversation(conv, messages, fmt=fmt)
            return gr.update(value=path, visible=True)

        export_btn.click(
            fn=on_export,
            inputs=[active_conv_id, export_format],
            outputs=[export_file],
        )

        # ===== SİSTEM PROMPTU EVENT HANDLERS =====

        # Özel sistem promptunu kaydet (tercihlerde kalıcı saklanır)
        def on_save_prompt(text: str) -> str:
            mgr = get_memory_manager()
            if text and text.strip():
                mgr.set_preference(_SYSTEM_PROMPT_PREF_KEY, text.strip())
                return "✓ Sistem promptu kaydedildi. Sonraki mesajlarda geçerli."
            # Boş kaydedilirse override'ı sil → varsayılana dön
            mgr.delete_preference(_SYSTEM_PROMPT_PREF_KEY)
            return "↺ Boş bırakıldı, varsayılan prompta dönüldü."

        save_prompt_btn.click(
            fn=on_save_prompt,
            inputs=[system_prompt_box],
            outputs=[prompt_status],
        )

        # Varsayılan sisteme dön (settings.yaml'daki prompt)
        def on_reset_prompt():
            get_memory_manager().delete_preference(_SYSTEM_PROMPT_PREF_KEY)
            default_prompt = get_settings().llm.system_prompt
            return default_prompt, "↺ Varsayılan prompt geri yüklendi."

        reset_prompt_btn.click(
            fn=on_reset_prompt,
            outputs=[system_prompt_box, prompt_status],
        )

        # Tema değiştirme butonu — tamamen istemci tarafında (JS) çalışır.
        # Gradio'nun tüm renk değişkenlerini override ederek sistem temasından
        # bağımsız, tek tıkla açık/koyu geçiş sağlar.
        theme_btn.click(fn=None, inputs=None, outputs=None, js=_theme_toggle_click_js())

        # Sayfa yüklendiğinde model listesini güncelle
        async def on_load():
            models = await _list_ollama_models()
            return gr.update(choices=models, value=models[0] if models else settings.llm.model)

        app.load(fn=on_load, outputs=[model_dropdown])

    return app


# Açık tema için Gradio renk değişkenleri (stone/turuncu paleti)
_LIGHT_VARS = {
    "--body-background-fill": "#fafaf9",
    "--block-background-fill": "#ffffff",
    "--body-text-color": "#1c1917",
    "--block-border-color": "#e7e5e4",
    "--border-color-primary": "#e7e5e4",
    "--border-color-accent": "#fed7aa",
    "--background-fill-primary": "#ffffff",
    "--background-fill-secondary": "#f5f5f4",
    "--neutral-50": "#fafaf9",
    "--neutral-100": "#f5f5f4",
    "--neutral-200": "#e7e5e4",
    "--neutral-300": "#d6d3d1",
    "--neutral-400": "#a8a29e",
    "--neutral-500": "#78716c",
    "--neutral-600": "#57534e",
    "--neutral-700": "#44403c",
    "--neutral-800": "#292524",
    "--neutral-900": "#1c1917",
    "--neutral-950": "#0f0e0d",
    "--input-background-fill": "#ffffff",
    "--block-title-text-color": "#1c1917",
    "--block-label-text-color": "#57534e",
    "--block-label-background-fill": "#ffffff",
    "--panel-background-fill": "#ffffff",
    "--table-even-background-fill": "#fafaf9",
    "--table-odd-background-fill": "#ffffff",
    "--table-border-color": "#e7e5e4",
    "--button-primary-background-fill": "#ea580c",
    "--button-primary-text-color": "#ffffff",
    "--button-secondary-background-fill": "#f5f5f4",
    "--button-secondary-text-color": "#1c1917",
    "--button-secondary-border-color": "#d6d3d1",
    "--checkbox-background-color": "#ffffff",
    "--checkbox-border-color": "#d6d3d1",
    "--input-border-color": "#d6d3d1",
    "--chatbot-code-background-color": "#f5f5f4",
}


def _theme_toggle_click_js() -> str:
    """
    Tema butonuna tıklanınca çalışan JS.

    Neden istemci tarafı (JS)?
    - Tema tamamen görsel bir tercih; sunucuya gitmesine gerek yok.
    - Anında geçiş olur, sayfa yenilenmez, sohbet kaybolmaz.

    Nasıl çalışır?
    - Açık moda geçerken Gradio'nun tüm renk değişkenlerini override eder.
    - Koyu moda dönerken bu override'ları kaldırıp `.dark` class'ını geri koyar,
      böylece Gradio kendi koyu temasını + bizim `.dark` CSS'imizi uygular.
    """
    import json
    light_vars_json = json.dumps(_LIGHT_VARS)
    return (
        "() => {"
        f"  const L = {light_vars_json};"
        "  const b = document.body;"
        "  const root = document.documentElement;"
        "  const btn = document.querySelector('.theme-toggle');"
        "  const isDark = b.classList.contains('dark');"
        "  if (isDark) {"
        "    b.classList.remove('dark');"
        "    b.classList.add('light-forced');"
        "    Object.entries(L).forEach(([k,v]) => root.style.setProperty(k, v, 'important'));"
        "    b.style.setProperty('background-color', L['--body-background-fill'], 'important');"
        "    b.style.setProperty('color', L['--body-text-color'], 'important');"
        "    document.querySelectorAll('.gradio-container').forEach(e => {"
        "      e.style.setProperty('background', L['--body-background-fill'], 'important');"
        "      e.style.setProperty('color', L['--body-text-color'], 'important');"
        "    });"
        "    if (btn) btn.textContent = '\\u2600\\ufe0f';"  # ☀️
        "  } else {"
        "    Object.keys(L).forEach(k => root.style.removeProperty(k));"
        "    b.style.removeProperty('background-color');"
        "    b.style.removeProperty('color');"
        "    document.querySelectorAll('.gradio-container').forEach(e => {"
        "      e.style.removeProperty('background');"
        "      e.style.removeProperty('color');"
        "    });"
        "    b.classList.remove('light-forced');"
        "    b.classList.add('dark');"
        "    if (btn) btn.textContent = '\\ud83c\\udf19';"  # 🌙
        "  }"
        "}"
    )


def _custom_css() -> str:
    """Claude tarzı sıcak, minimal arayüz stili (light + dark uyumlu)."""
    return """
    /* ===== LIGHT TEMA ===== */

    /* Genel yerleşim */
    .gradio-container {
        max-width: 1360px !important;
        margin: 0 auto !important;
    }

    /* Uygulama başlığı */
    .app-title h1 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: var(--primary-500) !important;
        padding: 0.2em 0 0.1em 0 !important;
    }

    .main-row { gap: 20px !important; }

    /* Sol kenar çubuğu — hafif kart görünümü */
    .sidebar {
        background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        border-radius: 16px !important;
        padding: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Bölüm etiketleri */
    .section-label h3 {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.65;
        margin: 8px 0 4px 0 !important;
    }

    /* Accordion başlıkları — sakin, tıklanabilir */
    .sidebar .label-wrap {
        font-weight: 500 !important;
        border-radius: 10px !important;
    }

    /* "Yeni Sohbet" butonu — belirgin */
    .sidebar button.primary {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* Sağ sohbet paneli */
    .chat-panel {
        background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Sohbet balonları — yumuşak köşeler */
    .chat-panel .message {
        border-radius: 14px !important;
        line-height: 1.55 !important;
    }

    /* Mesaj giriş satırı — sabit, hizalı */
    .input-row {
        gap: 8px !important;
        align-items: center !important;
        margin-top: 10px !important;
    }
    .input-row textarea {
        border-radius: 14px !important;
        font-size: 1rem !important;
    }
    .input-row button.primary {
        border-radius: 14px !important;
        min-width: 90px !important;
        font-weight: 600 !important;
    }

    /* Arama kutusu */
    .search-box textarea, .search-box input {
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        margin-bottom: 6px !important;
    }

    /* Geçmiş aksiyon satırı (sil + dışa aktar) */
    .history-actions {
        gap: 6px !important;
        align-items: center !important;
        margin-top: 6px !important;
    }

    /* İpucu / yardım metni */
    .hint-text p {
        font-size: 0.8rem !important;
        opacity: 0.7;
        margin: 0 0 6px 0 !important;
    }

    /* ===== DARK TEMA ===== */
    .dark {
        --dark-bg: #1a1a1a;
        --dark-surface: #242424;
        --dark-surface-hover: #2e2e2e;
        --dark-border: #3a3a3a;
        --dark-text: #e8e0d8;
        --dark-text-muted: #a39888;
        --dark-accent: #e8804c;
        --dark-accent-soft: rgba(232, 128, 76, 0.12);
    }

    .dark .gradio-container {
        background: var(--dark-bg) !important;
        color: var(--dark-text) !important;
    }

    .dark .app-title h1 {
        color: var(--dark-accent) !important;
    }

    /* Sidebar — koyu kart */
    .dark .sidebar {
        background: var(--dark-surface) !important;
        border-color: var(--dark-border) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Chat panel — koyu kart */
    .dark .chat-panel {
        background: var(--dark-surface) !important;
        border-color: var(--dark-border) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }

    /* Chatbot arka plan */
    .dark .chatbot {
        background: var(--dark-bg) !important;
    }

    /* Kullanıcı mesaj balonu */
    .dark .chatbot .message.user {
        background: var(--dark-accent-soft) !important;
        border-color: transparent !important;
    }

    /* Asistan mesaj balonu */
    .dark .chatbot .message.bot,
    .dark .chatbot .message.assistant {
        background: var(--dark-surface-hover) !important;
        border-color: transparent !important;
    }

    /* Mesaj kutusu */
    .dark .input-row textarea {
        background: var(--dark-surface-hover) !important;
        border-color: var(--dark-border) !important;
        color: var(--dark-text) !important;
    }
    .dark .input-row textarea:focus {
        border-color: var(--dark-accent) !important;
        box-shadow: 0 0 0 2px var(--dark-accent-soft) !important;
    }

    /* Butonlar */
    .dark button.primary {
        background: var(--dark-accent) !important;
        color: #fff !important;
        border: none !important;
    }
    .dark button.primary:hover {
        background: #d0703e !important;
    }
    .dark button.secondary,
    .dark button:not(.primary):not(.stop) {
        background: var(--dark-surface-hover) !important;
        border-color: var(--dark-border) !important;
        color: var(--dark-text) !important;
    }
    .dark button.stop,
    .dark .sidebar button.stop {
        background: #5c2020 !important;
        border-color: #7a2e2e !important;
        color: #f5c0c0 !important;
    }

    /* Dropdown / Select */
    .dark select,
    .dark .wrap.svelte-1sk0pmu,
    .dark input[type="text"],
    .dark .dropdown-container {
        background: var(--dark-surface-hover) !important;
        border-color: var(--dark-border) !important;
        color: var(--dark-text) !important;
    }

    /* Accordion başlıkları */
    .dark .sidebar .label-wrap {
        color: var(--dark-text) !important;
    }
    .dark .sidebar .label-wrap:hover {
        background: var(--dark-surface-hover) !important;
    }

    /* Checkbox label */
    .dark label span {
        color: var(--dark-text) !important;
    }

    /* Dataframe (sohbet geçmişi) */
    .dark table {
        background: var(--dark-surface) !important;
        color: var(--dark-text) !important;
    }
    .dark table th {
        background: var(--dark-surface-hover) !important;
        color: var(--dark-text-muted) !important;
        border-color: var(--dark-border) !important;
    }
    .dark table td {
        border-color: var(--dark-border) !important;
        color: var(--dark-text) !important;
    }
    .dark table tr:hover td {
        background: var(--dark-surface-hover) !important;
    }

    /* Markdown içerik */
    .dark .prose, .dark .prose * {
        color: var(--dark-text) !important;
    }
    .dark .prose code {
        background: var(--dark-surface-hover) !important;
        color: var(--dark-accent) !important;
    }
    .dark .prose pre {
        background: #1e1e1e !important;
        border-color: var(--dark-border) !important;
    }

    /* Bölüm etiketleri */
    .dark .section-label h3 {
        color: var(--dark-text-muted) !important;
    }

    /* Dosya yükleme alanları */
    .dark .upload-container,
    .dark .file-upload {
        background: var(--dark-surface-hover) !important;
        border-color: var(--dark-border) !important;
        color: var(--dark-text) !important;
    }

    /* Ses bileşeni */
    .dark .audio-container {
        background: var(--dark-surface) !important;
    }

    /* Placeholder metin */
    .dark textarea::placeholder,
    .dark input::placeholder {
        color: var(--dark-text-muted) !important;
    }

    /* Scrollbar — koyu */
    .dark ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    .dark ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }
    .dark ::-webkit-scrollbar-thumb {
        background: var(--dark-border);
        border-radius: 4px;
    }
    .dark ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Footer */
    .dark footer {
        background: var(--dark-bg) !important;
        border-color: var(--dark-border) !important;
    }

    /* ===== LIGHT TEMA ZORLAMA (sistem dark modda bile) ===== */
    .light-forced,
    .light-forced .gradio-container {
        background: #fafaf9 !important;
        color: #1c1917 !important;
    }
    .light-forced .sidebar {
        background: #ffffff !important;
        border-color: #e7e5e4 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .light-forced .chat-panel {
        background: #ffffff !important;
        border-color: #e7e5e4 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .light-forced .app-title h1 {
        color: #ea580c !important;
    }
    .light-forced .chatbot {
        background: #fafaf9 !important;
    }
    .light-forced .chatbot .message.user {
        background: #fff7ed !important;
        color: #1c1917 !important;
    }
    .light-forced .chatbot .message.bot,
    .light-forced .chatbot .message.assistant {
        background: #f5f5f4 !important;
        color: #1c1917 !important;
    }
    .light-forced .input-row textarea {
        background: #ffffff !important;
        border-color: #d6d3d1 !important;
        color: #1c1917 !important;
    }
    .light-forced .input-row textarea:focus {
        border-color: #ea580c !important;
        box-shadow: 0 0 0 2px rgba(234,88,12,0.15) !important;
    }
    .light-forced button.primary {
        background: #ea580c !important;
        color: #fff !important;
    }
    .light-forced button.primary:hover {
        background: #c2410c !important;
    }
    .light-forced button:not(.primary):not(.stop) {
        background: #f5f5f4 !important;
        border-color: #d6d3d1 !important;
        color: #1c1917 !important;
    }
    .light-forced button.stop,
    .light-forced .sidebar button.stop {
        background: #fef2f2 !important;
        border-color: #fca5a5 !important;
        color: #991b1b !important;
    }
    .light-forced select,
    .light-forced input[type="text"] {
        background: #ffffff !important;
        border-color: #d6d3d1 !important;
        color: #1c1917 !important;
    }
    .light-forced table {
        background: #ffffff !important;
        color: #1c1917 !important;
    }
    .light-forced table th {
        background: #f5f5f4 !important;
        color: #78716c !important;
        border-color: #e7e5e4 !important;
    }
    .light-forced table td {
        border-color: #e7e5e4 !important;
        color: #1c1917 !important;
    }
    .light-forced table tr:hover td {
        background: #fafaf9 !important;
    }
    .light-forced .prose, .light-forced .prose * {
        color: #1c1917 !important;
    }
    .light-forced .prose code {
        background: #f5f5f4 !important;
        color: #ea580c !important;
    }
    .light-forced .prose pre {
        background: #f5f5f4 !important;
        border-color: #e7e5e4 !important;
    }
    .light-forced .section-label h3 {
        color: #78716c !important;
    }
    .light-forced .sidebar .label-wrap {
        color: #1c1917 !important;
    }
    .light-forced .sidebar .label-wrap:hover {
        background: #f5f5f4 !important;
    }
    .light-forced label span {
        color: #1c1917 !important;
    }
    .light-forced textarea::placeholder,
    .light-forced input::placeholder {
        color: #a8a29e !important;
    }
    .light-forced footer {
        background: #fafaf9 !important;
        border-color: #e7e5e4 !important;
    }
    .light-forced .upload-container,
    .light-forced .file-upload {
        background: #f5f5f4 !important;
        border-color: #d6d3d1 !important;
        color: #1c1917 !important;
    }

    /* ===== TEMA TOGGLE BUTONU ===== */
    .title-row {
        display: flex !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
        padding: 0 !important;
    }
    /* Başlık kalan alanı kaplasın, buton sağa itilsin */
    .title-row .app-title {
        flex: 1 1 auto !important;
    }
    /* Butonun Gradio sarmalayıcı bloğu genişlemesin */
    .title-row > div:last-child {
        flex: 0 0 auto !important;
        width: 48px !important;
        min-width: 48px !important;
    }
    .theme-toggle {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        border-radius: 50% !important;
        font-size: 1.3rem !important;
        line-height: 1 !important;
        padding: 0 !important;
        border: 1px solid var(--border-color-primary) !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: transform 0.2s ease !important;
        flex: 0 0 auto !important;
    }
    .theme-toggle:hover {
        transform: scale(1.1) !important;
    }
    """
