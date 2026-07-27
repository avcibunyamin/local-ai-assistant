"""
Metin bölme (chunking) modülü.

Neden metni bölüyoruz?
- LLM'in context penceresi sınırlı (örn. 4096 token).
- Embedding modelleri de belirli uzunlukta metinlerde en iyi çalışır.
- Küçük parçalar sayesinde soruya en alakalı bölümü bulabiliriz.

Neden overlap (örtüşme)?
- Bir chunk'ın sonundaki cümle diğerinin başında da olsun ki
  bir cümlenin ortasından bölünen bilgi kaybolmasın.

Yaklaşım: Recursive character splitting
- Önce paragraflardan (çift satır), sonra satırlardan, sonra
  cümlelerden, en son kelimelerden böler.
- Böylece anlamsal bütünlük mümkün olduğunca korunur.
"""


# Bölme için deneme sırası: en büyük anlamsal birimden en küçüğe
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Metni belirtilen boyutta örtüşen parçalara böler.

    Args:
        text: Bölünecek metin
        chunk_size: Her parçanın maksimum karakter sayısı
        chunk_overlap: Ardışık parçalar arası örtüşme karakteri

    Returns:
        Metin parçaları listesi
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = _recursive_split(text, chunk_size, _SEPARATORS)
    return _merge_with_overlap(chunks, chunk_size, chunk_overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Metni ayırıcıları sırayla deneyerek küçük parçalara böler."""
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        # Son çare: karakter karakter böl
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    splits = text.split(separator)
    result: list[str] = []
    for part in splits:
        if len(part) <= chunk_size:
            if part:
                result.append(part)
        else:
            # Hala çok büyükse bir sonraki ayırıcıyla böl
            result.extend(_recursive_split(part, chunk_size, remaining))
    return result


def _merge_with_overlap(
    pieces: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Küçük parçaları chunk_size'a yaklaşacak şekilde birleştirir, overlap ekler."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Overlap: önceki chunk'ın sonundan bir miktar al
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                current = f"{tail} {piece}".strip()
            else:
                current = piece

    if current.strip():
        chunks.append(current.strip())

    return chunks
