"""Shared scraping clients."""
from __future__ import annotations

import asyncio
import logging
import re

import html2text
import httpx
from bs4 import BeautifulSoup
from firecrawl import Firecrawl

from scrapper.core.config import FIRECRAWL_API_KEY

logger = logging.getLogger(__name__)

# Firecrawl — used only for /map (URL discovery)
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

# html2text converter config
_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.ignore_images = True
_h2t.body_width = 0  # no wrapping

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags/selectors to remove (nav, boilerplate, etc.)
_REMOVE_SELECTORS = [
    "nav", "header", "footer", "script", "style", "noscript",
    "iframe", "svg", "[role='navigation']", "[role='banner']",
    "[role='contentinfo']", ".breadcrumb", ".sidebar", ".menu",
    ".cookie-banner", ".social-share", ".back-to-top",
    "#skip-nav", ".skip-link",
]

# Patterns to strip from final markdown
_JUNK_RE = re.compile(
    r"^("
    r"Skip to .*"
    r"|[\[\(]?\s*(?:Apply|Give Now|Donate|Sign Up|Log In|Subscribe).*"
    r"|(?:Cookie|Privacy|We use cookies).*"
    r")$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_main_content(html: str) -> str:
    """Parse HTML, strip boilerplate, return clean HTML of main content."""
    soup = BeautifulSoup(html, "lxml")

    # Remove junk elements
    for selector in _REMOVE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    # Try to find main content container
    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article")
    if main and len(main.get_text(strip=True)) > 200:
        return str(main)

    # Fallback: return the body (already stripped of nav/header/footer)
    body = soup.find("body")
    return str(body) if body else str(soup)


def _clean_markdown(md: str) -> str:
    """Remove remaining boilerplate lines from converted markdown."""
    md = _JUNK_RE.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)  # collapse blank lines
    return md.strip()


async def _fetch_url(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch a single URL, extract main content, convert to clean markdown."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        clean_html = _extract_main_content(resp.text)
        md = _h2t.handle(clean_html)
        md = _clean_markdown(md)

        if len(md) > 50:
            return md
    except Exception as e:
        logger.warning(f"  Failed to scrape {url}: {e}")
    return None


async def _scrape_batch(urls: list[str], concurrency: int = 20) -> dict[str, str]:
    """Scrape URLs concurrently, return {url: markdown}."""
    results: dict[str, str] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _bounded_fetch(client: httpx.AsyncClient, url: str):
        async with sem:
            md = await _fetch_url(client, url)
            if md:
                results[url] = md

    async with httpx.AsyncClient(headers=_HEADERS) as client:
        tasks = [_bounded_fetch(client, url) for url in urls]
        await asyncio.gather(*tasks)

    logger.info(f"  Scraped {len(results)}/{len(urls)} URLs successfully")
    return results


def scrape_urls(urls: list[str]) -> dict[str, str]:
    """Scrape a list of URLs using httpx + html2text (free). Returns {url: markdown}."""
    return asyncio.run(_scrape_batch(urls))
