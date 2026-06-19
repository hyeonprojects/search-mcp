from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_BYTES = int(os.getenv("PDF_READER_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
DEFAULT_TIMEOUT = int(os.getenv("PDF_READER_TIMEOUT", "30"))

mcp = FastMCP(name="PDFReaderMCP", mask_error_details=True)


class PDFPageResult(BaseModel):
    page_number: int
    text: str
    width: float
    height: float


class PDFReadResult(BaseModel):
    file_path: str
    total_pages: int
    pages: list[PDFPageResult]
    full_text: str


class PDFMetadata(BaseModel):
    file_path: str
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    total_pages: int
    file_size_bytes: int | None = None
    is_encrypted: bool = False
    page_sizes: list[dict[str, float]] = Field(default_factory=list)


def _open_pdf(file_path: str) -> fitz.Document:
    path = Path(file_path).resolve()
    if not path.exists():
        raise ToolError(f"파일이 존재하지 않습니다: {file_path}")
    if not path.is_file():
        raise ToolError(f"경로가 파일이 아닙니다: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ToolError(f"PDF 파일이 아닙니다: {file_path}")
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ToolError(f"PDF 열기 실패: {e}")
    if doc.is_encrypted:
        raise ToolError("암호화된 PDF는 지원하지 않습니다.")
    return doc


def _extract_pages(doc: fitz.Document, page_numbers: list[int] | None) -> list[PDFPageResult]:
    total = doc.page_count
    if page_numbers:
        indices = [p - 1 for p in page_numbers if 1 <= p <= total]
    else:
        indices = list(range(total))

    results: list[PDFPageResult] = []
    for idx in indices:
        page = doc[idx]
        rect = page.rect
        text = page.get_text("text")
        results.append(PDFPageResult(
            page_number=idx + 1,
            text=text,
            width=rect.width,
            height=rect.height,
        ))
    return results


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False}, timeout=60.0)
def read_pdf_file(
    file_path: str,
    page_numbers: list[int] | None = None,
) -> dict[str, Any]:
    """로컬 PDF 파일에서 텍스트를 추출합니다.

    Args:
        file_path: PDF 파일의 절대 또는 상대 경로
        page_numbers: 추출할 페이지 번호 목록 (1-indexed). None이면 전체 페이지 추출.

    Returns:
        전체 텍스트, 페이지별 텍스트, 총 페이지 수
    """
    doc = _open_pdf(file_path)
    try:
        pages = _extract_pages(doc, page_numbers)
        full_text = "\n\n".join(
            f"=== Page {p.page_number} ===\n{p.text}" for p in pages
        )
        result = PDFReadResult(
            file_path=str(Path(file_path).resolve()),
            total_pages=doc.page_count,
            pages=pages,
            full_text=full_text,
        )
        return result.model_dump()
    finally:
        doc.close()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=90.0)
async def read_pdf_url(
    url: str,
    page_numbers: list[int] | None = None,
    save_local: bool = False,
) -> dict[str, Any]:
    """URL에서 PDF를 다운로드하여 텍스트를 추출합니다.

    Args:
        url: PDF 파일의 URL (http/https)
        page_numbers: 추출할 페이지 번호 목록 (1-indexed). None이면 전체.
        save_local: True이면 data/pdfs/ 디렉터리에 파일 저장

    Returns:
        전체 텍스트, 페이지별 텍스트, 총 페이지 수
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("http/https URL만 허용됩니다.")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(url)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                raise ToolError(f"PDF 파일이 아닌 것 같습니다. Content-Type: {content_type}")
            data = r.content
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"PDF 다운로드 실패: {e}")

    if len(data) > MAX_PDF_BYTES:
        raise ToolError(f"PDF 파일이 너무 큽니다: {len(data)} bytes (최대 {MAX_PDF_BYTES} bytes)")

    saved_path: str | None = None
    if save_local:
        filename = Path(parsed.path).name or "downloaded.pdf"
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        saved_path = str(PDF_DIR / filename)
        Path(saved_path).write_bytes(data)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ToolError(f"PDF 파싱 실패: {e}")

    try:
        if doc.is_encrypted:
            raise ToolError("암호화된 PDF는 지원하지 않습니다.")
        pages = _extract_pages(doc, page_numbers)
        full_text = "\n\n".join(
            f"=== Page {p.page_number} ===\n{p.text}" for p in pages
        )
        result = PDFReadResult(
            file_path=saved_path or url,
            total_pages=doc.page_count,
            pages=pages,
            full_text=full_text,
        )
        return result.model_dump()
    finally:
        doc.close()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False}, timeout=30.0)
def get_pdf_metadata(file_path: str) -> dict[str, Any]:
    """PDF 파일의 메타데이터를 반환합니다.

    Args:
        file_path: PDF 파일 경로

    Returns:
        제목, 저자, 페이지 수, 생성일 등 메타데이터
    """
    doc = _open_pdf(file_path)
    try:
        meta = doc.metadata or {}
        page_sizes = []
        for i in range(doc.page_count):
            rect = doc[i].rect
            page_sizes.append({"page": i + 1, "width": rect.width, "height": rect.height})

        stat = Path(file_path).stat()
        result = PDFMetadata(
            file_path=str(Path(file_path).resolve()),
            title=meta.get("title") or None,
            author=meta.get("author") or None,
            subject=meta.get("subject") or None,
            creator=meta.get("creator") or None,
            producer=meta.get("producer") or None,
            creation_date=meta.get("creationDate") or None,
            modification_date=meta.get("modDate") or None,
            total_pages=doc.page_count,
            file_size_bytes=stat.st_size,
            is_encrypted=doc.is_encrypted,
            page_sizes=page_sizes,
        )
        return result.model_dump()
    finally:
        doc.close()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False}, timeout=30.0)
def list_pdf_files(directory: str | None = None) -> dict[str, Any]:
    """디렉터리 내 PDF 파일 목록을 반환합니다.

    Args:
        directory: 탐색할 디렉터리 경로. None이면 data/pdfs/ 사용.

    Returns:
        PDF 파일 목록과 기본 정보
    """
    target = Path(directory).resolve() if directory else PDF_DIR
    if not target.exists():
        raise ToolError(f"디렉터리가 존재하지 않습니다: {target}")
    if not target.is_dir():
        raise ToolError(f"경로가 디렉터리가 아닙니다: {target}")

    files = []
    for p in sorted(target.glob("**/*.pdf")):
        stat = p.stat()
        files.append({
            "path": str(p),
            "name": p.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
        })

    return {"directory": str(target), "count": len(files), "files": files}


if __name__ == "__main__":
    mcp.run()
