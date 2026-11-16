from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(slots=True)
class TextSegment:
    text: str
    page: int | None = None


def extract_segments(filename: str, content: bytes) -> list[TextSegment]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return [TextSegment(text=content.decode("utf-8", errors="replace"))]
    if suffix == ".pdf":
        return _extract_pdf(content)
    raise UnsupportedDocumentError("仅支持 PDF、TXT、MD 文件")


def _extract_pdf(content: bytes) -> list[TextSegment]:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:  # pypdf raises several parser-specific errors
        raise UnsupportedDocumentError("PDF 文件无法解析，请确认文件未损坏") from exc

    segments: list[TextSegment] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            segments.append(TextSegment(text=text, page=page_number))
    if not segments:
        raise UnsupportedDocumentError("PDF 中没有可提取的文本，扫描件需要先进行 OCR")
    return segments
