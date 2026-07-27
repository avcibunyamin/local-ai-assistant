"""
Sohbet dışa aktarma (export).

Bir konuşmayı diske dosya olarak yazar; UI bu dosyayı indirme olarak sunar.

Neden ayrı bir modül?
- SRP (tek sorumluluk): history.py sadece veritabanı; export biçimlendirme ayrı.
- Yeni format eklemek (ör. PDF) burada tek noktadan yapılır.

Desteklenen formatlar:
- JSON: yapısal, programatik yeniden kullanım için (tam veri).
- TXT:  düz metin, insan okuması için sade.
- Markdown: başlıklı, kopyalanabilir; not uygulamalarına yapıştırmak için.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import ROOT_DIR

# Dışa aktarılan dosyaların yazılacağı dizin (proje köküne göre)
_EXPORT_DIR = ROOT_DIR / "data" / "exports"


def _safe_filename(title: str) -> str:
    """Başlığı güvenli bir dosya adına çevirir (özel karakterleri temizler)."""
    # Türkçe karakterleri koru ama dosya-sistemi için tehlikeli olanları at
    cleaned = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", title).strip("_ ")
    cleaned = cleaned[:50] if cleaned else "sohbet"
    return cleaned


def export_conversation(
    conversation: dict,
    messages: list[dict],
    fmt: str = "json",
) -> str:
    """
    Bir konuşmayı dosyaya yazar ve dosya yolunu döner.

    Args:
        conversation: {id, title, created_at, updated_at}
        messages: [{role, content, created_at}, ...]
        fmt: "json" | "txt" | "md"

    Returns:
        Yazılan dosyanın mutlak yolu (str)
    """
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    title = conversation.get("title", "sohbet")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"{_safe_filename(title)}_{stamp}"

    if fmt == "json":
        path = _EXPORT_DIR / f"{base}.json"
        payload = {
            "baslik": title,
            "olusturma": conversation.get("created_at"),
            "guncelleme": conversation.get("updated_at"),
            "mesajlar": [
                {
                    "rol": m.get("role"),
                    "icerik": m.get("content"),
                    "zaman": m.get("created_at"),
                }
                for m in messages
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    if fmt == "md":
        path = _EXPORT_DIR / f"{base}.md"
        lines = [f"# {title}", ""]
        for m in messages:
            who = "🧑 Kullanıcı" if m.get("role") == "user" else "🤖 Asistan"
            lines.append(f"### {who}")
            lines.append("")
            lines.append(m.get("content", ""))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    # Varsayılan: TXT
    path = _EXPORT_DIR / f"{base}.txt"
    lines = [f"Başlık: {title}", "=" * 40, ""]
    for m in messages:
        who = "Kullanıcı" if m.get("role") == "user" else "Asistan"
        lines.append(f"[{who}]")
        lines.append(m.get("content", ""))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
