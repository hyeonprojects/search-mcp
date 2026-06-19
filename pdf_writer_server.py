from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_OUT_DIR = DATA_DIR / "pdf_output"
PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(name="PDFWriterMCP", mask_error_details=True)

# 한글 폰트 등록 (Windows 기본 폰트 경로)
_FONT_REGISTERED = False
_KOREAN_FONT_NAME = "NanumGothic"

_FONT_SEARCH_PATHS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    BASE_DIR / "fonts",
]

_FONT_CANDIDATES = [
    ("NanumGothic", ["NanumGothic.ttf", "NanumGothicBold.ttf"]),
    ("MalgunGothic", ["malgun.ttf", "malgunbd.ttf"]),
    ("Gulim", ["gulim.ttc"]),
    ("Batang", ["batang.ttc"]),
    ("Arial Unicode MS", ["ARIALUNI.TTF"]),
]


def _find_font_file(filename: str) -> Path | None:
    for base in _FONT_SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def _register_korean_font() -> str:
    global _FONT_REGISTERED, _KOREAN_FONT_NAME
    if _FONT_REGISTERED:
        return _KOREAN_FONT_NAME

    for font_name, files in _FONT_CANDIDATES:
        font_file = files[0]
        found = _find_font_file(font_file)
        if found:
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(found)))
                _KOREAN_FONT_NAME = font_name
                _FONT_REGISTERED = True
                return font_name
            except Exception:
                continue

    # 폰트를 찾지 못하면 기본 Helvetica 사용 (한글 깨질 수 있음)
    _KOREAN_FONT_NAME = "Helvetica"
    _FONT_REGISTERED = True
    return _KOREAN_FONT_NAME


PAGE_SIZES = {
    "A4": A4,
    "LETTER": LETTER,
}


class PDFCreateResult(BaseModel):
    output_path: str
    page_count_estimate: int
    font_used: str
    file_size_bytes: int


def _resolve_output_path(output_path: str) -> Path:
    p = Path(output_path)
    if not p.is_absolute():
        p = PDF_OUT_DIR / p
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _build_styles(font_name: str, font_size: int) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "KTitle",
            fontName=font_name,
            fontSize=font_size + 8,
            leading=(font_size + 8) * 1.4,
            spaceAfter=6 * mm,
            spaceBefore=4 * mm,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1a1a2e"),
        ),
        "heading1": ParagraphStyle(
            "KHeading1",
            fontName=font_name,
            fontSize=font_size + 4,
            leading=(font_size + 4) * 1.4,
            spaceAfter=3 * mm,
            spaceBefore=5 * mm,
            textColor=colors.HexColor("#16213e"),
        ),
        "heading2": ParagraphStyle(
            "KHeading2",
            fontName=font_name,
            fontSize=font_size + 2,
            leading=(font_size + 2) * 1.4,
            spaceAfter=2 * mm,
            spaceBefore=4 * mm,
            textColor=colors.HexColor("#0f3460"),
        ),
        "body": ParagraphStyle(
            "KBody",
            fontName=font_name,
            fontSize=font_size,
            leading=font_size * 1.6,
            spaceAfter=2 * mm,
        ),
        "bullet": ParagraphStyle(
            "KBullet",
            fontName=font_name,
            fontSize=font_size,
            leading=font_size * 1.5,
            leftIndent=8 * mm,
            spaceAfter=1 * mm,
        ),
        "code": ParagraphStyle(
            "KCode",
            fontName="Courier",
            fontSize=max(font_size - 2, 8),
            leading=max(font_size - 2, 8) * 1.4,
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            spaceAfter=2 * mm,
        ),
    }


def _markdown_to_flowables(
    markdown_text: str,
    styles: dict[str, ParagraphStyle],
) -> list:
    flowables = []
    lines = markdown_text.splitlines()
    in_code_block = False
    code_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 코드 블록 처리
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                code_text = "\n".join(code_lines)
                safe = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                flowables.append(Paragraph(safe.replace("\n", "<br/>"), styles["code"]))
                flowables.append(Spacer(1, 2 * mm))
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # 제목
        if line.startswith("# "):
            flowables.append(Paragraph(_escape(line[2:].strip()), styles["title"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_escape(line[3:].strip()), styles["heading1"]))
        elif line.startswith("### "):
            flowables.append(Paragraph(_escape(line[4:].strip()), styles["heading2"]))
        # 수평선
        elif line.strip() in ("---", "***", "___"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            flowables.append(Spacer(1, 2 * mm))
        # 페이지 구분 (<!-- pagebreak -->)
        elif line.strip().lower() in ("<!-- pagebreak -->", "<pagebreak/>", "\\newpage"):
            flowables.append(PageBreak())
        # 빈 줄
        elif not line.strip():
            flowables.append(Spacer(1, 3 * mm))
        # 글머리 기호
        elif re.match(r"^[\*\-\+] ", line):
            text = line[2:].strip()
            flowables.append(Paragraph(f"• {_escape(text)}", styles["bullet"]))
        elif re.match(r"^\d+\. ", line):
            text = re.sub(r"^\d+\. ", "", line).strip()
            num = re.match(r"^(\d+)\.", line).group(1)
            flowables.append(Paragraph(f"{num}. {_escape(text)}", styles["bullet"]))
        # 일반 단락
        else:
            flowables.append(Paragraph(_escape(line), styles["body"]))

        i += 1

    return flowables


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, timeout=60.0)
def create_pdf_from_text(
    content: str,
    output_path: str,
    title: str | None = None,
    font_size: int = 11,
    page_size: str = "A4",
    margin_mm: float = 20.0,
) -> dict[str, Any]:
    """일반 텍스트로 PDF를 생성합니다.

    Args:
        content: PDF에 넣을 텍스트 내용
        output_path: 저장할 파일 경로 (절대경로 또는 data/pdf_output/ 기준 상대경로)
        title: PDF 제목 (None이면 첫 줄을 제목으로 사용)
        font_size: 본문 폰트 크기 (기본 11pt)
        page_size: 페이지 크기 ("A4" 또는 "LETTER")
        margin_mm: 여백 크기 (mm 단위, 기본 20mm)

    Returns:
        생성된 파일 경로, 예상 페이지 수, 사용된 폰트
    """
    font_name = _register_korean_font()
    out = _resolve_output_path(output_path)
    psize = PAGE_SIZES.get(page_size.upper(), A4)
    margin = margin_mm * mm

    styles = _build_styles(font_name, font_size)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=psize,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=title or "",
    )

    flowables = []
    if title:
        flowables.append(Paragraph(_escape(title), styles["title"]))
        flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        flowables.append(Spacer(1, 4 * mm))

    for line in content.splitlines():
        if line.strip():
            flowables.append(Paragraph(_escape(line), styles["body"]))
        else:
            flowables.append(Spacer(1, 3 * mm))

    try:
        doc.build(flowables)
    except Exception as e:
        raise ToolError(f"PDF 생성 실패: {e}")

    stat = out.stat()
    result = PDFCreateResult(
        output_path=str(out),
        page_count_estimate=max(1, len(content) // 2000),
        font_used=font_name,
        file_size_bytes=stat.st_size,
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, timeout=60.0)
def create_pdf_from_markdown(
    content: str,
    output_path: str,
    title: str | None = None,
    font_size: int = 11,
    page_size: str = "A4",
    margin_mm: float = 20.0,
) -> dict[str, Any]:
    """마크다운 텍스트로 서식이 있는 PDF를 생성합니다.

    Args:
        content: 마크다운 형식의 텍스트 (# 제목, ## 소제목, - 목록, ``` 코드블록 지원)
        output_path: 저장할 파일 경로 (절대경로 또는 data/pdf_output/ 기준 상대경로)
        title: 문서 제목 (메타데이터용). None이면 첫 번째 # 제목 사용.
        font_size: 본문 폰트 크기 (기본 11pt)
        page_size: 페이지 크기 ("A4" 또는 "LETTER")
        margin_mm: 여백 크기 (mm 단위, 기본 20mm)

    Returns:
        생성된 파일 경로, 예상 페이지 수, 사용된 폰트
    """
    font_name = _register_korean_font()
    out = _resolve_output_path(output_path)
    psize = PAGE_SIZES.get(page_size.upper(), A4)
    margin = margin_mm * mm

    styles = _build_styles(font_name, font_size)
    doc_title = title or ""
    if not doc_title:
        for line in content.splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break

    doc = SimpleDocTemplate(
        str(out),
        pagesize=psize,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=doc_title,
    )

    flowables = _markdown_to_flowables(content, styles)
    if not flowables:
        flowables = [Paragraph("(내용 없음)", styles["body"])]

    try:
        doc.build(flowables)
    except Exception as e:
        raise ToolError(f"PDF 생성 실패: {e}")

    stat = out.stat()
    result = PDFCreateResult(
        output_path=str(out),
        page_count_estimate=max(1, len(content) // 2000),
        font_used=font_name,
        file_size_bytes=stat.st_size,
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, timeout=60.0)
def create_translated_pdf(
    translated_pages: list[dict[str, Any]],
    output_path: str,
    title: str | None = None,
    source_language: str = "원문",
    target_language: str = "번역",
    font_size: int = 11,
    page_size: str = "A4",
    include_page_numbers: bool = True,
) -> dict[str, Any]:
    """번역된 텍스트로 PDF를 생성합니다. read_pdf_file의 결과에서 번역된 내용을 받아 새 PDF로 만듭니다.

    Args:
        translated_pages: 번역된 페이지 목록. 각 항목은 {"page_number": int, "text": str} 형식.
                          read_pdf_file 결과의 pages 필드를 번역 후 그대로 전달 가능.
        output_path: 저장할 파일 경로 (절대경로 또는 data/pdf_output/ 기준 상대경로)
        title: 문서 제목
        source_language: 원문 언어 표시명 (헤더용)
        target_language: 번역 언어 표시명 (헤더용)
        font_size: 본문 폰트 크기 (기본 11pt)
        page_size: 페이지 크기 ("A4" 또는 "LETTER")
        include_page_numbers: True이면 각 섹션에 원본 페이지 번호 표시

    Returns:
        생성된 파일 경로, 페이지 수, 사용된 폰트
    """
    if not translated_pages:
        raise ToolError("번역된 페이지 목록이 비어 있습니다.")

    font_name = _register_korean_font()
    out = _resolve_output_path(output_path)
    psize = PAGE_SIZES.get(page_size.upper(), A4)
    margin = 20 * mm

    styles = _build_styles(font_name, font_size)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=psize,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=title or f"{target_language} 번역 문서",
    )

    flowables = []

    # 제목 섹션
    if title:
        flowables.append(Paragraph(_escape(title), styles["title"]))
        flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e")))
        flowables.append(Spacer(1, 6 * mm))

    lang_info = f"{source_language} → {target_language} 번역본"
    flowables.append(Paragraph(_escape(lang_info), ParagraphStyle(
        "LangInfo",
        fontName=font_name,
        fontSize=font_size - 1,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )))

    for page_data in translated_pages:
        page_num = page_data.get("page_number", "?")
        text = page_data.get("text", "").strip()

        if not text:
            continue

        if include_page_numbers:
            flowables.append(Paragraph(
                f"— {source_language} {page_num}페이지 —",
                ParagraphStyle(
                    "PageLabel",
                    fontName=font_name,
                    fontSize=font_size - 1,
                    textColor=colors.HexColor("#888888"),
                    spaceBefore=4 * mm,
                    spaceAfter=2 * mm,
                )
            ))

        for line in text.splitlines():
            if line.strip():
                flowables.append(Paragraph(_escape(line), styles["body"]))
            else:
                flowables.append(Spacer(1, 2 * mm))

        flowables.append(HRFlowable(width="100%", thickness=0.3, color=colors.lightgrey))
        flowables.append(Spacer(1, 3 * mm))

    try:
        doc.build(flowables)
    except Exception as e:
        raise ToolError(f"PDF 생성 실패: {e}")

    stat = out.stat()
    result = PDFCreateResult(
        output_path=str(out),
        page_count_estimate=len(translated_pages),
        font_used=font_name,
        file_size_bytes=stat.st_size,
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, timeout=30.0)
def list_output_pdfs() -> dict[str, Any]:
    """생성된 PDF 파일 목록을 반환합니다 (data/pdf_output/ 디렉터리).

    Returns:
        생성된 PDF 파일 목록과 파일 크기 정보
    """
    files = []
    for p in sorted(PDF_OUT_DIR.glob("*.pdf")):
        stat = p.stat()
        files.append({
            "path": str(p),
            "name": p.name,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
        })
    return {"output_directory": str(PDF_OUT_DIR), "count": len(files), "files": files}


if __name__ == "__main__":
    mcp.run()
