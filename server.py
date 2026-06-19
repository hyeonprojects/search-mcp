from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse, quote_plus

import httpx
import tldextract
from browser_use import Agent as BrowserAgent
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langchain_openai import ChatOpenAI
from markdownify import markdownify as md
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from readability import Document
from urllib.robotparser import RobotFileParser

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.getenv("LOCAL_CRAWLER_USER_AGENT", "LocalCrawlerMCP/1.0 (+local)")
ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in os.getenv("LOCAL_CRAWLER_ALLOWED_DOMAINS", "docs.python.org,playwright.dev").split(",")
    if d.strip()
}
BLOCK_PRIVATE_IPS = os.getenv("LOCAL_CRAWLER_BLOCK_PRIVATE_IPS", "true").lower() == "true"
REQUEST_DELAY_SECONDS = float(os.getenv("LOCAL_CRAWLER_REQUEST_DELAY", "1.0"))
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("LOCAL_CRAWLER_TIMEOUT", "20"))
MAX_HTML_BYTES = int(os.getenv("LOCAL_CRAWLER_MAX_HTML_BYTES", str(2 * 1024 * 1024)))
MAX_MARKDOWN_CHARS = int(os.getenv("LOCAL_CRAWLER_MAX_MARKDOWN_CHARS", "40000"))
MAX_LINKS_PER_PAGE = int(os.getenv("LOCAL_CRAWLER_MAX_LINKS_PER_PAGE", "200"))
DEFAULT_MAX_PAGES = int(os.getenv("LOCAL_CRAWLER_DEFAULT_MAX_PAGES", "10"))
SAVE_RAW_HTML = os.getenv("LOCAL_CRAWLER_SAVE_RAW_HTML", "false").lower() == "true"
RESPECT_ROBOTS = os.getenv("LOCAL_CRAWLER_RESPECT_ROBOTS", "true").lower() == "true"

BROWSER_USE_BASE_URL = os.getenv("BROWSER_USE_BASE_URL", "http://localhost:1234/v1")
BROWSER_USE_MODEL = os.getenv("BROWSER_USE_MODEL", "")
BROWSER_USE_MAX_STEPS = int(os.getenv("BROWSER_USE_MAX_STEPS", "15"))

mcp = FastMCP(name="LocalCrawMCP", mask_error_details=True)
_last_request_time: dict[str, float] = {}
_robots_cache: dict[str, RobotFileParser] = {}


class PageResult(BaseModel):
    url: str
    final_url: str
    title: str | None = None
    text_markdown: str = ""
    links: list[str] = Field(default_factory=list)
    meta_description: str | None = None
    canonical_url: str | None = None
    status_code: int | None = None
    used_browser: bool = False


class CrawlResult(BaseModel):
    start_url: str
    count: int
    pages: list[PageResult]


class ExtractFieldsResult(BaseModel):
    url: str
    title: str | None = None
    extracted: dict[str, str]


class SearchHit(BaseModel):
    url: str
    title: str | None = None
    snippet: str
    matched_terms: list[str]


class SearchResult(BaseModel):
    start_url: str
    query: str
    count: int
    hits: list[SearchHit]


class GoogleSearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class GoogleSearchResult(BaseModel):
    query: str
    count: int
    hits: list[GoogleSearchHit]


class GoogleSearchCrawlResult(BaseModel):
    query: str
    search_count: int
    crawled_count: int
    hits: list[GoogleSearchHit]
    pages: list[PageResult]


class SiteDeepResult(BaseModel):
    hit: GoogleSearchHit
    pages: list[PageResult]
    page_count: int


class GoogleSearchDeepCrawlResult(BaseModel):
    query: str
    search_count: int
    investigated_sites: int
    total_pages_crawled: int
    sites: list[SiteDeepResult]


def hostname_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def registrable_domain(host: str) -> str:
    ext = tldextract.extract(host)
    if not ext.domain or not ext.suffix:
        return host
    return f"{ext.domain}.{ext.suffix}".lower()


def domain_allowed(host: str) -> bool:
    return True
    # host = host.lower()
    # if host in ALLOWED_DOMAINS:
    #     return True
    # reg = registrable_domain(host)
    # return reg in ALLOWED_DOMAINS


def is_private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except Exception:
        return True
    return False


def validate_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise ToolError("http/https URL만 허용됩니다.")
    host = hostname_of(url)
    if not host:
        raise ToolError("유효한 호스트가 없습니다.")
    if BLOCK_PRIVATE_IPS and is_private_host(host):
        raise ToolError("사설망/로컬 주소는 허용되지 않습니다.")
    if not domain_allowed(host):
        raise ToolError(f"허용되지 않은 도메인입니다: {host}")


def normalize_url(base: str, href: str | None) -> str | None:
    if not href:
        return None
    joined = urljoin(base, href)
    joined, _ = urldefrag(joined)
    p = urlparse(joined)
    if p.scheme not in {"http", "https"}:
        return None
    return joined


async def rate_limit(host: str) -> None:
    now = time.monotonic()
    last = _last_request_time.get(host, 0.0)
    wait_for = REQUEST_DELAY_SECONDS - (now - last)
    if wait_for > 0:
        await asyncio.sleep(wait_for)
    _last_request_time[host] = time.monotonic()


async def can_fetch(url: str) -> bool:
    if not RESPECT_ROBOTS:
        return True
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        robots_url = f"{root}/robots.txt"
        rp = RobotFileParser()
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                r = await client.get(robots_url, headers={"User-Agent": USER_AGENT})
                if r.status_code >= 400:
                    rp.parse([])
                else:
                    rp.parse(r.text.splitlines())
        except Exception:
            rp.parse([])
        _robots_cache[root] = rp
    return _robots_cache[root].can_fetch(USER_AGENT, url)


async def fetch_static(url: str, timeout_s: int = DEFAULT_TIMEOUT_SECONDS) -> tuple[str, str, int]:
    host = hostname_of(url)
    await rate_limit(host)
    if not await can_fetch(url):
        raise ToolError("robots.txt 정책상 이 URL은 가져올 수 없습니다.")
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            raise ToolError("HTML 문서가 아닙니다.")
        content = r.content[:MAX_HTML_BYTES]
        html = content.decode(r.encoding or "utf-8", errors="ignore")
        return str(r.url), html, r.status_code


async def fetch_browser(url: str, timeout_s: int = 30) -> tuple[str, str, int]:
    host = hostname_of(url)
    await rate_limit(host)
    if not await can_fetch(url):
        raise ToolError("robots.txt 정책상 이 URL은 가져올 수 없습니다.")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=USER_AGENT)
        response = await page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
        html = await page.content()
        final_url = page.url
        status = response.status if response else 200
        await browser.close()
        return final_url, html[:MAX_HTML_BYTES], status



def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()



def parse_html(url: str, final_url: str, html: str, status_code: int, used_browser: bool) -> PageResult:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    meta_description = None
    canonical_url = None

    meta_tag = soup.select_one('meta[name="description"]')
    if meta_tag:
        meta_description = meta_tag.get("content")

    canonical_tag = soup.select_one('link[rel="canonical"]')
    if canonical_tag:
        canonical_url = canonical_tag.get("href")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    links: list[str] = []
    for a in soup.select("a[href]"):
        u = normalize_url(final_url, a.get("href"))
        if u:
            links.append(u)

    try:
        doc = Document(html)
        article_html = doc.summary()
        article_title = doc.short_title() or title
        text_markdown = md(article_html)
        title = article_title
    except Exception:
        text_markdown = md(str(soup.body or soup))

    text_markdown = clean_text(text_markdown)[:MAX_MARKDOWN_CHARS]

    return PageResult(
        url=url,
        final_url=final_url,
        title=title,
        text_markdown=text_markdown,
        links=list(dict.fromkeys(links))[:MAX_LINKS_PER_PAGE],
        meta_description=meta_description,
        canonical_url=canonical_url,
        status_code=status_code,
        used_browser=used_browser,
    )



async def _google_search_playwright(query: str, max_results: int = 10) -> list[dict[str, str]]:
    """Playwright을 이용해 Google 검색 결과(URL·제목·스니펫)를 추출합니다."""
    encoded = quote_plus(query)
    search_url = f"https://www.google.com/search?q={encoded}&num={min(max_results * 2, 20)}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        try:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            html = await page.content()
        finally:
            await browser.close()

    soup = BeautifulSoup(html, "lxml")
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for div in soup.select("div.g"):
        if len(results) >= max_results:
            break
        a_tag = div.select_one("a[href]")
        if not a_tag:
            continue
        href: str = a_tag.get("href", "")
        if not href or not href.startswith("http"):
            continue
        if "google" in urlparse(href).netloc:
            continue
        if href in seen:
            continue
        seen.add(href)

        h3 = div.select_one("h3")
        title = h3.get_text(strip=True) if h3 else ""
        if not title:
            continue

        snippet_tag = div.select_one("div.VwiC3b, span.aCOpRe, div[data-sncf]")
        snippet = snippet_tag.get_text(" ", strip=True)[:300] if snippet_tag else ""

        results.append({"url": href, "title": title, "snippet": snippet})

    return results


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._") or "result"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=45.0)
async def extract_page(url: str, use_browser: bool = False) -> dict[str, Any]:
    """단일 페이지를 읽어 제목, 본문 Markdown, 링크, 메타데이터를 추출합니다."""
    validate_url(url)
    try:
        if use_browser:
            final_url, html, status_code = await fetch_browser(url)
            result = parse_html(url, final_url, html, status_code, True)
        else:
            try:
                final_url, html, status_code = await fetch_static(url)
                result = parse_html(url, final_url, html, status_code, False)
            except Exception:
                final_url, html, status_code = await fetch_browser(url)
                result = parse_html(url, final_url, html, status_code, True)

        if SAVE_RAW_HTML:
            filename = safe_slug(urlparse(result.final_url).netloc + "_" + (result.title or "page")) + ".html"
            (RAW_DIR / filename).write_text(html, encoding="utf-8", errors="ignore")

        return result.model_dump()
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"페이지 추출 실패: {type(e).__name__}")


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=180.0)
async def crawl_site(start_url: str, max_pages: int = DEFAULT_MAX_PAGES, same_domain_only: bool = True, use_browser: bool = False) -> dict[str, Any]:
    """시작 URL에서 내부 링크를 따라가며 여러 페이지를 수집합니다."""
    validate_url(start_url)
    root_domain = registrable_domain(hostname_of(start_url))

    seen: set[str] = set()
    queued: set[str] = {start_url}
    q = deque([start_url])
    pages: list[dict[str, Any]] = []

    max_pages = max(1, min(int(max_pages), 100))

    while q and len(pages) < max_pages:
        url = q.popleft()
        queued.discard(url)
        if url in seen:
            continue
        seen.add(url)

        try:
            page = await extract_page(url=url, use_browser=use_browser)
            pages.append(page)

            for link in page.get("links", []):
                try:
                    validate_url(link)
                except Exception:
                    continue
                if same_domain_only and registrable_domain(hostname_of(link)) != root_domain:
                    continue
                if link not in seen and link not in queued:
                    q.append(link)
                    queued.add(link)
        except Exception:
            continue

    return CrawlResult(start_url=start_url, count=len(pages), pages=[PageResult(**p) for p in pages]).model_dump()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=180.0)
async def search_crawled_content(start_url: str, query: str, max_pages: int = 10, same_domain_only: bool = True) -> dict[str, Any]:
    """사이트를 크롤링한 뒤 본문에서 키워드를 검색합니다."""
    crawl = await crawl_site(start_url=start_url, max_pages=max_pages, same_domain_only=same_domain_only, use_browser=False)
    terms = [t.lower() for t in re.findall(r"\w+", query) if t.strip()]
    hits: list[SearchHit] = []

    for page in crawl["pages"]:
        text = (page.get("text_markdown") or "")
        low = text.lower()
        matched = [t for t in terms if t in low]
        if matched:
            idx = min((low.find(t) for t in matched if low.find(t) >= 0), default=0)
            snippet = text[max(0, idx - 120): idx + 280].replace("\n", " ").strip()
            hits.append(
                SearchHit(
                    url=page["final_url"],
                    title=page.get("title"),
                    snippet=snippet,
                    matched_terms=matched,
                )
            )

    result = SearchResult(start_url=start_url, query=query, count=len(hits), hits=hits)
    return result.model_dump()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=60.0)
async def extract_fields(url: str, fields: list[str], use_browser: bool = False) -> dict[str, Any]:
    """페이지 본문에서 지정한 필드명 주변 텍스트를 간단 추출합니다. LLM 전처리용 보조 도구입니다."""
    page = await extract_page(url=url, use_browser=use_browser)
    text = page.get("text_markdown", "")
    extracted: dict[str, str] = {}
    for field in fields:
        pattern = re.compile(rf"(?i)({re.escape(field)})\s*[:\-]?\s*(.+)")
        found = ""
        for line in text.splitlines():
            m = pattern.search(line)
            if m:
                found = m.group(2).strip()[:500]
                break
        extracted[field] = found
    return ExtractFieldsResult(url=page["final_url"], title=page.get("title"), extracted=extracted).model_dump()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=300.0)
async def browser_agent(task: str, max_steps: int = BROWSER_USE_MAX_STEPS) -> dict[str, Any]:
    """browser-use Agent로 웹을 자율 탐색합니다. 검색, 정보 수집, 로그인, 양식 작성 등 복잡한 웹 작업을 자연어로 지시할 수 있습니다."""
    if not BROWSER_USE_MODEL:
        raise ToolError("BROWSER_USE_MODEL 환경변수에 LM Studio에서 로드한 모델명을 설정해주세요.")
    max_steps = max(1, min(int(max_steps), 50))
    try:
        llm = ChatOpenAI(
            base_url=BROWSER_USE_BASE_URL,
            api_key="lm-studio",
            model=BROWSER_USE_MODEL,
            temperature=0.0,
        )
        agent = BrowserAgent(task=task, llm=llm)
        history = await agent.run(max_steps=max_steps)
        final = history.final_result()
        return {
            "task": task,
            "result": final or "결과를 추출하지 못했습니다.",
            "steps_taken": len(history.history),
        }
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"browser_agent 실행 실패: {type(e).__name__}: {e}")


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=60.0)
async def google_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Google 검색을 수행하고 결과 URL, 제목, 스니펫을 반환합니다. Playwright 기반으로 동작합니다."""
    max_results = max(1, min(int(max_results), 20))
    try:
        raw = await _google_search_playwright(query, max_results)
        hits = [GoogleSearchHit(**h) for h in raw]
        return GoogleSearchResult(query=query, count=len(hits), hits=hits).model_dump()
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Google 검색 실패: {type(e).__name__}: {e}")


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=300.0)
async def google_search_and_crawl(
    query: str,
    max_results: int = 5,
    use_browser: bool = False,
) -> dict[str, Any]:
    """Google 검색 후 상위 결과 페이지들의 본문을 자동으로 크롤링하여 반환합니다.

    Args:
        query: 검색어
        max_results: 크롤링할 최대 결과 수 (1–10)
        use_browser: True이면 JS 렌더링이 필요한 페이지에 Playwright 사용
    """
    max_results = max(1, min(int(max_results), 10))
    try:
        raw = await _google_search_playwright(query, max_results)
        hits = [GoogleSearchHit(**h) for h in raw]

        pages: list[PageResult] = []
        for hit in hits:
            try:
                validate_url(hit.url)
                page_data = await extract_page(url=hit.url, use_browser=use_browser)
                pages.append(PageResult(**page_data))
            except Exception:
                continue

        return GoogleSearchCrawlResult(
            query=query,
            search_count=len(hits),
            crawled_count=len(pages),
            hits=hits,
            pages=pages,
        ).model_dump()
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Google 검색 크롤링 실패: {type(e).__name__}: {e}")


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True}, timeout=600.0)
async def google_search_and_deep_crawl(
    query: str,
    max_results: int = 3,
    max_pages_per_site: int = 5,
    use_browser: bool = False,
) -> dict[str, Any]:
    """Google 검색 후 각 결과 사이트를 내부 링크까지 따라가며 심층 크롤링합니다.

    Args:
        query: 검색어
        max_results: 조사할 사이트 수 (1–10)
        max_pages_per_site: 사이트당 크롤링할 최대 페이지 수 (1–20)
        use_browser: True이면 JS 렌더링이 필요한 페이지에 Playwright 사용
    """
    max_results = max(1, min(int(max_results), 10))
    max_pages_per_site = max(1, min(int(max_pages_per_site), 20))
    try:
        raw = await _google_search_playwright(query, max_results)
        hits = [GoogleSearchHit(**h) for h in raw]

        sites: list[SiteDeepResult] = []
        total_pages = 0

        for hit in hits:
            try:
                validate_url(hit.url)
                root_domain = registrable_domain(hostname_of(hit.url))

                seen: set[str] = set()
                queued: set[str] = {hit.url}
                q: deque[str] = deque([hit.url])
                pages: list[PageResult] = []

                while q and len(pages) < max_pages_per_site:
                    url = q.popleft()
                    queued.discard(url)
                    if url in seen:
                        continue
                    seen.add(url)

                    try:
                        page_data = await extract_page(url=url, use_browser=use_browser)
                        page = PageResult(**page_data)
                        pages.append(page)

                        for link in page.links:
                            try:
                                validate_url(link)
                            except Exception:
                                continue
                            if registrable_domain(hostname_of(link)) != root_domain:
                                continue
                            if link not in seen and link not in queued:
                                q.append(link)
                                queued.add(link)
                    except Exception:
                        continue

                total_pages += len(pages)
                sites.append(SiteDeepResult(hit=hit, pages=pages, page_count=len(pages)))
            except Exception:
                continue

        return GoogleSearchDeepCrawlResult(
            query=query,
            search_count=len(hits),
            investigated_sites=len(sites),
            total_pages_crawled=total_pages,
            sites=sites,
        ).model_dump()
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Google 심층 크롤링 실패: {type(e).__name__}: {e}")


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
def save_json(filename: str, data: dict[str, Any]) -> str:
    """결과를 data/extracted 아래 JSON 파일로 저장합니다."""
    safe = Path(filename).name
    out = EXTRACTED_DIR / safe
    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def get_server_config() -> dict[str, Any]:
    """현재 서버의 보안/실행 설정을 반환합니다."""
    return {
        "allowed_domains": sorted(ALLOWED_DOMAINS),
        "block_private_ips": BLOCK_PRIVATE_IPS,
        "respect_robots": RESPECT_ROBOTS,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "default_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_html_bytes": MAX_HTML_BYTES,
        "max_markdown_chars": MAX_MARKDOWN_CHARS,
        "max_links_per_page": MAX_LINKS_PER_PAGE,
        "default_max_pages": DEFAULT_MAX_PAGES,
        "save_raw_html": SAVE_RAW_HTML,
        "data_dir": str(DATA_DIR),
    }


from pdf_reader_server import mcp as _pdf_reader_mcp
from pdf_writer_server import mcp as _pdf_writer_mcp
from pdf_to_csv_server import mcp as _pdf_to_csv_mcp

mcp.mount(_pdf_reader_mcp)
mcp.mount(_pdf_writer_mcp)
mcp.mount(_pdf_to_csv_mcp)


if __name__ == "__main__":
    mcp.run()
