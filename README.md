# Local AI Assistant

Tamamen lokal calisan, moduler, acik kaynakli AI asistani.

## Ozellikler

### Asama 1 - Temel Chat
- Ollama uzerinden lokal LLM destegi (Qwen, Llama, Gemma, DeepSeek, Mistral)
- Streaming chat (token token cevap)
- Modern Gradio arayuzu (Claude tarzi tema, tek tikla karanlik/aydinlik, markdown, kod bloklari)
- FastAPI REST API
- YAML tabanli konfigurasyon
- Tek satirla model degistirme

### Asama 2 - Chat Gecmisi + UI Gelistirmeleri
- SQLite tabanli kalici chat gecmisi
- Sol panelde konusma listesi
- Konusmalar arasi gecis
- Yeni sohbet olusturma
- Konusma silme
- Otomatik baslik olusturma (ilk mesajdan)
- UI'dan model secme (Ollama'daki tum modeller)
- Mesaj kopyalama butonu

### Asama 3 - RAG (Belge Tabanli Cevaplama)
- Belge yukleme: PDF, DOCX, XLSX, CSV, TXT, MD, kod dosyalari
- FAISS vektor veritabani ile hizli arama
- Ollama embedding (nomic-embed-text) ile lokal vektorlestirme
- Otomatik metin bolme (chunking) + ortusme
- "Belgelere gore cevapla" toggle'i
- Yuklu belgeler kalici (diskte saklanir)

**RAG icin ek model gerekli:**
```bash
ollama pull nomic-embed-text
```

### Asama 4 - Hafiza Sistemi
- Uzun sureli hafiza: kullanici hakkinda kalici gercekler (semantik arama)
- Otomatik ogrenme: her konusmadan LLM ile gercek cikarimi
- Otomatik geri cagirma: soruya gore ilgili hafizalar prompta eklenir
- Kullanici tercihleri (key-value)
- Konusma ozeti (LLM ile)
- Hafiza yonetim paneli (goruntule/temizle)

### Asama 5 - Tool Calling (Arac Kullanimi)
- Ollama native tool calling (qwen2.5 destekli)
- Plugin mimarisi: her arac bir Tool alt sinifi
- Hazir araclar: read_file, write_file, list_files, make_dir
- Tehlikeli araclar: run_shell, run_python (allow_dangerous ile)
- Guvenlik: araclar workspace dizinine kisitli, path traversal korumasi
- Tool-calling dongusu: AI araci secip calistirir, sonuca gore devam eder

**Guvenlik notu:** Terminal ve Python araclari varsayilan KAPALI.
Acmak icin `config/settings.yaml`'da `tools.allow_dangerous: true` yapin.

### Asama 6 - Vision (Goruntu Analizi)
- Multimodal model (qwen2.5vl:7b) ile goruntu analizi
- Goruntu aciklama, OCR (metin cikarma), ozel soru
- UI'dan goruntu yukleme + analiz
- Ekran goruntusu ve UI analizi
- Ayri OCR kutuphanesi gerekmez (model dahili yapar)

**Vision icin ek model gerekli:**
```bash
ollama pull qwen2.5vl:7b
```

### Asama 7 - Ses (STT / TTS)
- STT (konusma->metin): faster-whisper, lokal, Turkce destekli
- TTS (metin->konusma): pyttsx3, Windows SAPI5, offline
- Mikrofon ile sesli komut -> otomatik metne cevirme
- Sesli yanit: AI cevabini otomatik seslendirme
- STT modeli ilk kullanimda otomatik iner (~500 MB)

**Turkce sesli yanit notu:** pyttsx3 Windows'un kurulu seslerini kullanir.
Turkce ses icin: Ayarlar > Zaman ve Dil > Konusma > Ses ekle > Turkce.
Kurulu degilse Ingilizce ses Turkce metni bozuk okur (STT etkilenmez).

### Asama 8 - Multi-Agent
- Plan-and-Execute deseni: gorevi otomatik adimlara boler
- Planner: LLM ile gorevi alt adimlara ayirir
- Agent: her adimi araclarla yurutur, sonuclari biriktirir
- Kendi kendini yonlendirme: sonraki adim onceki sonuclari gorur
- Final ozet: yapilan islerin toparlanmasi
- UI'da "Agent Modu" ile plan + adimlar + sonuc canli gorunur

Agent, Asama 5'teki arac sistemini yeniden kullanir (DRY).

### Asama 9 - Bilgisayar Kontrolu
- pyautogui ile fare, klavye, ekran kontrolu
- Araclar: take_screenshot, click_mouse, type_text, press_key, open_program, screen_info
- Tool sisteminin parcasi -> agent bile bilgisayari kontrol edebilir
- Vision + kontrol: "ekrani gor, sonra tikla" kombinasyonu
- GUVENLIK: hepsi allow_dangerous ile kapali, FAILSAFE aktif
  (fareyi sol ust koseye goturunce acil durdurma)

### Asama 10 - Optimizasyon + Plugin Sistemi
- Plugin sistemi: plugins/ klasorune kendi araclarinizi ekleyin
  (uygulama baslarken otomatik yuklenir)
- Ornek plugin: plugins/example_tool.py (count_words)
- Embedding cache: ayni metni tekrar vektorlemez (6 sn -> 0 ms)
- Async + streaming + singleton model yukleme (event loop bloklanmaz)

**Kendi araclarinizi yazma:**
```python
# plugins/benim_aracim.py
from src.tools.base import Tool, ToolResult

class BenimAracim(Tool):
    name = "benim_aracim"
    description = "Ne yaptigini acikla"

    def parameters(self):
        return {"type": "object", "properties": {
            "girdi": {"type": "string", "description": "..."}
        }, "required": ["girdi"]}

    async def execute(self, girdi: str) -> ToolResult:
        return ToolResult(True, f"Sonuc: {girdi}")
```

### Ek Gelistirmeler - Arayuz + Kullanim Kolayligi

10 asama tamamlandiktan sonra eklenen, Claude benzeri kullanim ozellikleri:

- **Tema butonu (karanlik / aydinlik):** Sag ust kosedeki dugme ile tek tikla
  gecis. Sistem temasindan bagimsiz calisir (JS ile tum renk degiskenleri
  override edilir; sayfa yenilenmez, sohbet kaybolmaz).
- **Konusma arama:** Sol paneldeki arama kutusu; hem baslikta hem mesaj
  iceriginde arar (SQL `LIKE` + `DISTINCT`). Yazdikca canli filtreler.
- **Sohbeti disa aktarma:** Aktif konusmayi **JSON / TXT / Markdown** olarak
  indir. (Ayri modul: `src/chat/exporter.py`)
- **Sistem promptu UI'dan duzenleme:** "Sistem Promptu" panelinden asistanin
  kisiligini degistir; tercih deposunda **kalici** saklanir. Bos birak ->
  `settings.yaml`'daki varsayilana doner.
- **Sohbete belge surukle-birak:** Mesaj kutusuna (MultimodalTextbox) belge
  surukle -> otomatik RAG'a eklenir, o tur belgeye gore cevaplar.
- **Kod bloklari:** Markdown kod bloklari acik/koyu temada tutarli stillendi.

> Not: Arama, disa aktarma ve sistem promptu duzenleme UI uzerinden calisir;
> ayri REST endpoint gerektirmez.

## Kurulum

### 1. Ollama Kur

[ollama.com](https://ollama.com) adresinden Ollama'yi indirip kurun.

Bir model indirin:
```bash
ollama pull qwen2.5:7b
```

### 2. Python Bagimliliklar

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Calistir

```bash
python -m src.main
```

Tarayicida `http://localhost:8000/chat` adresine gidin.

## Ayarlar

`config/settings.yaml` dosyasini duzenleyerek ayarlari degistirebilirsiniz:

```yaml
llm:
  model: "llama3.1:8b"     # Model degistir
  temperature: 0.7          # Yaraticilik
  max_tokens: 4096          # Maks token

ui:
  theme: "soft"             # soft, default, glass, monochrome

chat:
  db_path: "data/chat_history.db"
  max_conversations: 100
```

## API

| Endpoint                              | Method | Aciklama                |
|---------------------------------------|--------|-------------------------|
| `/api/chat`                           | POST   | Streaming chat          |
| `/api/chat/complete`                  | POST   | Tam yanit               |
| `/api/models`                         | GET    | Yuklu modelleri listele |
| `/api/health`                         | GET    | Ollama durumu           |
| `/api/conversations`                  | GET    | Konusma listesi         |
| `/api/conversations/{id}/messages`    | GET    | Konusma mesajlari       |
| `/api/conversations/{id}`             | DELETE | Konusma silme           |
| `/api/rag/sources`                    | GET    | Yuklu belgeler          |
| `/api/rag/query`                      | POST   | Belge parcasi ara       |
| `/api/rag/documents`                  | DELETE | Belgeleri temizle       |
| `/api/memory`                         | GET    | Hafizalari listele      |
| `/api/memory`                         | POST   | Hafiza ekle             |
| `/api/memory/{id}`                    | DELETE | Hafiza sil              |
| `/api/preferences`                    | GET    | Tercihleri listele      |
| `/api/preferences`                    | PUT    | Tercih ayarla           |
| `/api/tools`                          | GET    | Araclari listele        |
| `/api/vision/ocr`                     | POST   | Goruntuden metin cikar  |

## Proje Yapisi

```
local-ai-assistant/
├── config/
│   └── settings.yaml        # Ayarlar
├── data/
│   └── chat_history.db      # SQLite veritabani (otomatik olusur)
├── src/
│   ├── main.py               # Ana uygulama
│   ├── config/
│   │   └── settings.py       # Pydantic ayar modelleri
│   ├── core/
│   │   └── llm_client.py     # Ollama async istemci
│   ├── chat/
│   │   ├── models.py         # Veri modelleri
│   │   ├── router.py         # API endpoint'leri
│   │   ├── history.py        # SQLite chat gecmisi (+ arama)
│   │   └── exporter.py       # Sohbet disa aktarma (JSON/TXT/MD)
│   ├── rag/
│   │   ├── document_loader.py # Belge okuma (PDF, DOCX, ...)
│   │   ├── text_splitter.py   # Metin bolme (chunking)
│   │   ├── vector_store.py     # FAISS vektor deposu
│   │   └── engine.py          # RAG motoru
│   ├── memory/
│   │   ├── memory_store.py    # Uzun sureli hafiza (SQLite + numpy)
│   │   ├── preferences.py     # Kullanici tercihleri
│   │   ├── extractor.py       # LLM ile gercek cikarimi
│   │   └── manager.py         # Hafiza yoneticisi
│   ├── tools/
│   │   ├── base.py            # Tool soyut sinifi
│   │   ├── registry.py        # Arac kayit sistemi
│   │   ├── executor.py        # Tool-calling dongusu
│   │   └── builtin/           # Hazir araclar (dosya, shell, python, bilgisayar)
│   ├── vision/
│   │   └── engine.py          # Goruntu analizi (multimodal)
│   ├── voice/
│   │   ├── stt.py             # Konusma tanima (faster-whisper)
│   │   └── tts.py             # Sesli yanit (pyttsx3)
│   ├── agents/
│   │   ├── planner.py         # Gorev planlayici
│   │   └── agent.py           # Plan-and-Execute agent
│   └── ui/
│       └── app.py            # Gradio arayuzu
├── plugins/                   # Kullanici araclari (otomatik yuklenir)
│   └── example_tool.py
├── tests/
├── requirements.txt
└── README.md
```

## Yol Haritasi

- [x] Asama 1: Proje mimarisi + temel chat
- [x] Asama 2: Chat gecmisi + UI gelistirmeleri
- [x] Asama 3: RAG sistemi
- [x] Asama 4: Hafiza sistemi
- [x] Asama 5: Tool Calling
- [x] Asama 6: Vision
- [x] Asama 7: Ses (STT/TTS)
- [x] Asama 8: Multi-Agent
- [x] Asama 9: Bilgisayar kontrolu
- [x] Asama 10: Optimizasyon + Plugin sistemi
- [x] Ek: Tema butonu, konusma arama, disa aktarma, sistem promptu duzenleme, surukle-birak RAG

**Tum asamalar tamamlandi!** 🎉
