from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_OUT_DIR = DATA_DIR / "csv_output"
PDF_DIR = DATA_DIR / "pdfs"
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_BYTES = int(os.getenv("PDF_CSV_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
DEFAULT_TIMEOUT = int(os.getenv("PDF_CSV_TIMEOUT", "30"))

mcp = FastMCP(name="PDFToCSVMCP", mask_error_details=True)


class TableInfo(BaseModel):
    page: int
    table_index: int
    row_count: int
    col_count: int


class ConvertResult(BaseModel):
    output_path: str
    total_tables: int
    total_rows: int
    tables: list[TableInfo]
    source: str


class PreviewResult(BaseModel):
    source: str
    total_pages_scanned: int
    total_tables: int
    tables: list[dict[str, Any]]


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


def _page_indices(doc: fitz.Document, page_numbers: list[int] | None) -> list[int]:
    total = doc.page_count
    if page_numbers:
        return [p - 1 for p in page_numbers if 1 <= p <= total]
    return list(range(total))


def _extract_tables(doc: fitz.Document, indices: list[int]) -> list[dict[str, Any]]:
    """PyMuPDF find_tables()로 테이블 추출."""
    results: list[dict[str, Any]] = []
    for idx in indices:
        page = doc[idx]
        try:
            finder = page.find_tables()
            tables = finder.tables
        except Exception:
            tables = []

        for t_idx, table in enumerate(tables):
            rows = table.extract()
            if not rows:
                continue
            # None 셀을 빈 문자열로 정규화
            cleaned = [
                [str(cell).strip() if cell is not None else "" for cell in row]
                for row in rows
            ]
            results.append({
                "page": idx + 1,
                "table_index": t_idx + 1,
                "rows": cleaned,
                "row_count": len(cleaned),
                "col_count": max(len(r) for r in cleaned) if cleaned else 0,
            })

    return results


def _tables_to_csv_bytes(
    tables: list[dict[str, Any]],
    include_markers: bool,
) -> tuple[bytes, int]:
    """테이블 목록을 CSV 바이트로 직렬화. 총 행 수 반환."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    total_rows = 0

    for i, t in enumerate(tables):
        if include_markers:
            if i > 0:
                writer.writerow([])  # 테이블 간 빈 줄
            writer.writerow([f"# Page {t['page']} / Table {t['table_index']}"])

        for row in t["rows"]:
            writer.writerow(row)
            total_rows += 1

    return buf.getvalue().encode("utf-8-sig"), total_rows  # BOM for Excel compatibility


def _resolve_output_path(output_path: str | None, stem: str) -> Path:
    if output_path:
        p = Path(output_path)
        if not p.is_absolute():
            p = CSV_OUT_DIR / p
    else:
        p = CSV_OUT_DIR / f"{stem}.csv"
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": False}, timeout=60.0)
def convert_pdf_to_csv(
    file_path: str,
    output_path: str | None = None,
    page_numbers: list[int] | None = None,
    include_markers: bool = True,
) -> dict[str, Any]:
    """로컬 PDF 파일의 테이블을 추출하여 CSV 파일로 저장합니다.

    Args:
        file_path: PDF 파일 경로 (절대 또는 상대)
        output_path: 저장할 CSV 경로. None이면 data/csv_output/<원본파일명>.csv
        page_numbers: 처리할 페이지 번호 목록 (1-indexed). None이면 전체 페이지.
        include_markers: True이면 각 테이블 앞에 "# Page X / Table Y" 주석 행 삽입

    Returns:
        저장된 CSV 경로, 추출된 테이블 수, 총 행 수
    """
    doc = _open_pdf(file_path)
    try:
        indices = _page_indices(doc, page_numbers)
        tables = _extract_tables(doc, indices)
    finally:
        doc.close()

    if not tables:
        raise ToolError("PDF에서 테이블을 찾을 수 없습니다. 표 형태의 데이터가 없거나 이미지 기반 PDF일 수 있습니다.")

    stem = Path(file_path).stem
    out = _resolve_output_path(output_path, stem)
    csv_bytes, total_rows = _tables_to_csv_bytes(tables, include_markers)
    out.write_bytes(csv_bytes)

    result = ConvertResult(
        output_path=str(out),
        total_tables=len(tables),
        total_rows=total_rows,
        tables=[
            TableInfo(
                page=t["page"],
                table_index=t["table_index"],
                row_count=t["row_count"],
                col_count=t["col_count"],
            )
            for t in tables
        ],
        source=str(Path(file_path).resolve()),
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": False, "openWorldHint": True}, timeout=90.0)
async def convert_pdf_url_to_csv(
    url: str,
    output_path: str | None = None,
    page_numbers: list[int] | None = None,
    include_markers: bool = True,
    save_pdf: bool = False,
) -> dict[str, Any]:
    """URL에서 PDF를 다운로드하여 테이블을 추출하고 CSV로 저장합니다.

    Args:
        url: PDF 파일 URL (http/https)
        output_path: 저장할 CSV 경로. None이면 data/csv_output/<파일명>.csv
        page_numbers: 처리할 페이지 번호 목록 (1-indexed). None이면 전체.
        include_markers: True이면 각 테이블 앞에 페이지/테이블 마커 행 삽입
        save_pdf: True이면 다운로드한 PDF를 data/pdfs/ 에도 저장

    Returns:
        저장된 CSV 경로, 추출된 테이블 수, 총 행 수
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
        raise ToolError(f"PDF가 너무 큽니다: {len(data)} bytes (최대 {MAX_PDF_BYTES} bytes)")

    filename = Path(parsed.path).name or "downloaded"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    if save_pdf:
        pdf_path = PDF_DIR / filename
        pdf_path.write_bytes(data)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ToolError(f"PDF 파싱 실패: {e}")

    try:
        if doc.is_encrypted:
            raise ToolError("암호화된 PDF는 지원하지 않습니다.")
        indices = _page_indices(doc, page_numbers)
        tables = _extract_tables(doc, indices)
    finally:
        doc.close()

    if not tables:
        raise ToolError("PDF에서 테이블을 찾을 수 없습니다. 표 형태의 데이터가 없거나 이미지 기반 PDF일 수 있습니다.")

    stem = Path(filename).stem
    out = _resolve_output_path(output_path, stem)
    csv_bytes, total_rows = _tables_to_csv_bytes(tables, include_markers)
    out.write_bytes(csv_bytes)

    result = ConvertResult(
        output_path=str(out),
        total_tables=len(tables),
        total_rows=total_rows,
        tables=[
            TableInfo(
                page=t["page"],
                table_index=t["table_index"],
                row_count=t["row_count"],
                col_count=t["col_count"],
            )
            for t in tables
        ],
        source=url,
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False}, timeout=60.0)
def preview_pdf_tables(
    file_path: str,
    page_numbers: list[int] | None = None,
    max_rows_per_table: int = 10,
) -> dict[str, Any]:
    """PDF에서 테이블을 추출하여 미리보기 데이터를 반환합니다 (파일 저장 없음).

    Args:
        file_path: PDF 파일 경로
        page_numbers: 확인할 페이지 번호 목록. None이면 전체.
        max_rows_per_table: 각 테이블에서 반환할 최대 행 수 (기본 10)

    Returns:
        발견된 테이블 목록과 행 데이터 미리보기
    """
    doc = _open_pdf(file_path)
    try:
        indices = _page_indices(doc, page_numbers)
        tables = _extract_tables(doc, indices)
        pages_scanned = len(indices)
    finally:
        doc.close()

    preview_tables = []
    for t in tables:
        preview_tables.append({
            "page": t["page"],
            "table_index": t["table_index"],
            "row_count": t["row_count"],
            "col_count": t["col_count"],
            "preview_rows": t["rows"][:max_rows_per_table],
            "truncated": t["row_count"] > max_rows_per_table,
        })

    result = PreviewResult(
        source=str(Path(file_path).resolve()),
        total_pages_scanned=pages_scanned,
        total_tables=len(tables),
        tables=preview_tables,
    )
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False}, timeout=30.0)
def list_csv_files(directory: str | None = None) -> dict[str, Any]:
    """변환된 CSV 파일 목록을 반환합니다.

    Args:
        directory: 탐색할 디렉터리. None이면 data/csv_output/ 사용.

    Returns:
        CSV 파일 목록과 기본 정보
    """
    target = Path(directory).resolve() if directory else CSV_OUT_DIR
    if not target.exists():
        raise ToolError(f"디렉터리가 존재하지 않습니다: {target}")
    if not target.is_dir():
        raise ToolError(f"경로가 디렉터리가 아닙니다: {target}")

    files = []
    for p in sorted(target.glob("**/*.csv")):
        stat = p.stat()
        files.append({
            "path": str(p),
            "name": p.name,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
        })

    return {"directory": str(target), "count": len(files), "files": files}


if __name__ == "__main__":
    mcp.run()
