"""University pipeline tasks."""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langgraph.func import task

from scrapper.core.config import CLASSIFICATION_MODEL, INDEXING_MODEL, STRUCTURING_MODEL
from scrapper.core.llm import llm_call
from scrapper.core.scraper import firecrawl, scrape_urls
from scrapper.core.storage import upload_markdown_to_s3
from scrapper.pipelines.university.schemas import CATEGORY_REGISTRY, CategoryDef

logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# Max concurrent LLM calls within a task (structuring chunks, merges, index batches).
# Keep this conservative: structuring calls are ~40K tokens each, and structuring
# and indexing run concurrently, so high values trip OpenAI's per-minute token
# limit (TPM). The retry/backoff in llm_call absorbs occasional spillover.
# Override with LLM_CONCURRENCY=N for higher-tier accounts.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "3"))


def _map_concurrent(fn, items: list, max_workers: int = LLM_CONCURRENCY) -> list:
    """Run fn over items in parallel threads, preserving input order.

    LLM calls are I/O-bound, so threads give real speedup. Exceptions propagate.
    """
    if not items:
        return []
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


@task
def discover_urls(university_url: str) -> dict:
    """Use Firecrawl /map to discover all URLs on the site."""
    url = university_url
    if not url.startswith("http"):
        url = f"https://{url}"

    logger.info(f"Mapping URLs for {url}...")
    result = firecrawl.map(url=url)

    links = result.links if hasattr(result, "links") else result.get("links", [])

    # Firecrawl /map returns url + (often) title + description per link. Capture
    # the metadata to give the classifier real signal beyond the URL string.
    urls: list[str] = []
    url_meta: dict[str, dict] = {}
    for link in links:
        if isinstance(link, dict):
            u, title, desc = link.get("url"), link.get("title"), link.get("description")
        else:
            u = link.url
            title = getattr(link, "title", None)
            desc = getattr(link, "description", None)
        if not u:
            continue
        urls.append(u)
        url_meta[u] = {"title": title or "", "description": desc or ""}

    n_title = sum(1 for m in url_meta.values() if m["title"])
    n_desc = sum(1 for m in url_meta.values() if m["description"])
    logger.info(f"Discovered {len(urls)} URLs ({n_title} with title, {n_desc} with description)")

    domain = urlparse(url).netloc.replace("www.", "")
    name = domain.split(".")[0].upper()

    return {"urls": urls, "url_meta": url_meta, "university_name": name, "university_url": url}


def _build_classify_prompt(categories: list[CategoryDef]) -> str:
    """Build classification prompt dynamically from selected categories."""
    cat_lines = "\n".join(
        f"- {cat.name}: {cat.classify_hint}" for cat in categories
    )
    cat_json = ",\n  ".join(f'"{cat.name}": ["url1", "url2"]' for cat in categories)
    return f"""Classify these university website URLs into categories.
Each URL should go into AT MOST one category. Skip irrelevant URLs (login pages, news, events, social media, generic pages).
Only include URLs that are clearly relevant to the category.

Each entry is a URL, optionally followed by indented "title:" and "description:" lines describing that page — use them to judge relevance. Return only the URL strings (never the title/description text).

Categories:
{cat_lines}

URLs:
{{url_text}}

Respond with ONLY valid JSON matching this exact structure (no other text):
{{{{
  {cat_json}
}}}}"""


def _format_url_entry(url: str, url_meta: dict[str, dict] | None) -> str:
    """Render a URL plus its title/description (if any) for the classify prompt."""
    meta = url_meta.get(url) if url_meta else None
    if not meta:
        return url
    entry = url
    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or "").strip()
    if title:
        entry += f"\n  title: {title[:150]}"
    if desc:
        entry += f"\n  description: {desc[:200]}"
    return entry


def _classify_batch(
    url_batch: list[str],
    prompt_template: str,
    cat_names: list[str],
    url_meta: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """Classify a single batch of URLs (with optional title/description signal)."""
    url_text = "\n".join(_format_url_entry(u, url_meta) for u in url_batch)
    raw = llm_call(CLASSIFICATION_MODEL, prompt_template.format(url_text=url_text), max_tokens=8192, stage="classification")

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    # Only keep keys we asked for
    return {k: data.get(k, []) for k in cat_names}


def _merge_batches(batches: list[dict[str, list[str]]], cat_names: list[str]) -> dict[str, list[str]]:
    """Merge multiple batch results, deduplicating."""
    merged: dict[str, set[str]] = {k: set() for k in cat_names}
    for batch in batches:
        for k in cat_names:
            merged[k].update(batch.get(k, []))
    return {k: list(v) for k, v in merged.items()}


@task
def classify_urls(
    urls: list[str],
    categories: list[str],
    url_meta: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """Use LLM to classify discovered URLs into selected categories, in batches.

    url_meta maps each URL to {"title", "description"} from Firecrawl /map, fed
    into the prompt to improve relevance decisions.
    """
    if not urls:
        raise ValueError("No URLs discovered to classify")

    cat_defs = [CATEGORY_REGISTRY[c] for c in categories]
    cat_names = [c.name for c in cat_defs]
    prompt_template = _build_classify_prompt(cat_defs)

    batches = [urls[i:i + BATCH_SIZE] for i in range(0, len(urls), BATCH_SIZE)]
    logger.info(f"Classifying {len(urls)} URLs in {len(batches)} batches (parallel) for categories: {cat_names}")

    results = _map_concurrent(
        lambda batch: _classify_batch(batch, prompt_template, cat_names, url_meta),
        batches,
    )

    classified = _merge_batches(results, cat_names)

    # Trim to the per-category safety ceiling (the LLM already decided relevance).
    for cat_def in cat_defs:
        kept = len(classified[cat_def.name])
        if kept > cat_def.max_urls:
            logger.warning(
                f"  {cat_def.name}: {kept} classified exceeds ceiling {cat_def.max_urls} — trimming"
            )
            classified[cat_def.name] = classified[cat_def.name][:cat_def.max_urls]

    total = sum(len(v) for v in classified.values())
    logger.info(f"Classified {total} URLs across {len(categories)} categories")
    for name, urls_list in classified.items():
        logger.info(f"  {name}: {len(urls_list)} URLs")

    return classified


@task
def scrape_pages(classified: dict[str, list[str]], categories: list[str]) -> dict[str, list[str]]:
    """Scrape pages per category using Crawl4AI (free)."""
    scraped: dict[str, list[str]] = {}

    for cat in categories:
        urls = classified.get(cat, [])
        if not urls:
            logger.info(f"  {cat}: no URLs, skipping")
            continue

        logger.info(f"Scraping {len(urls)} pages for '{cat}' with Crawl4AI...")
        results = scrape_urls(urls)
        pages = [f"<!-- Source: {url} -->\n{md}" for url, md in results.items()]

        scraped[cat] = pages
        logger.info(f"  {cat}: scraped {len(pages)} pages successfully")

    return scraped


CHUNK_CHAR_LIMIT = 100_000
MAX_PAGE_CHARS = 80_000  # truncate individual pages exceeding this


def _chunk_pages(pages: list[str], limit: int = CHUNK_CHAR_LIMIT) -> list[list[str]]:
    """Split pages into chunks that fit within the character limit.

    Individual pages exceeding MAX_PAGE_CHARS are truncated first.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for page in pages:
        # Truncate oversized individual pages
        if len(page) > MAX_PAGE_CHARS:
            page = page[:MAX_PAGE_CHARS] + "\n\n[Page truncated due to length]"
        page_len = len(page) + 5  # account for separator
        if current and current_len + page_len > limit:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(page)
        current_len += page_len
    if current:
        chunks.append(current)
    return chunks


def _structure_chunk(pages: list[str], university_name: str, cat_def: CategoryDef, chunk_idx: int, total_chunks: int) -> str:
    """Structure a single chunk of pages."""
    combined = "\n\n---\n\n".join(pages)
    context = f" (batch {chunk_idx + 1}/{total_chunks})" if total_chunks > 1 else ""

    prompt = f"""You are extracting structured information about {university_name}'s {cat_def.name.replace('_', ' ')}{context}.

{cat_def.structure_prompt}

Rules:
- Use clear markdown headings (##, ###, ####)
- Remove duplicate information
- Keep factual content only, no marketing fluff
- Preserve ALL specific details (names, numbers, dates, URLs, scores, dollar amounts)
- If info is missing or unclear, omit it rather than guessing

Raw scraped content from multiple pages:

{combined}"""

    return llm_call(STRUCTURING_MODEL, prompt, max_tokens=16384, stage="structuring")


MERGE_BATCH_SIZE = 5  # merge this many parts at a time


def _merge_pair(parts: list[str], university_name: str, cat_def: CategoryDef) -> str:
    """Merge a small group of structured parts into one."""
    all_parts = "\n\n---\n\n".join(parts)

    prompt = f"""You have {len(parts)} partial structured documents about {university_name}'s {cat_def.name.replace('_', ' ')}.
Merge them into ONE cohesive, well-organized markdown document.

{cat_def.structure_prompt}

Rules:
- Start with: # {university_name} - {cat_def.name.replace('_', ' ').title()}
- Merge duplicate entries (keep the most complete version)
- Maintain consistent heading hierarchy (##, ###, ####)
- Preserve ALL specific details (names, numbers, dates, URLs, scores, dollar amounts)
- Keep factual content only
- Do NOT summarize or shorten — include everything from the parts

Partial documents to merge:

{all_parts}"""

    return llm_call(STRUCTURING_MODEL, prompt, max_tokens=16384, stage="structuring")


def _merge_structured(parts: list[str], university_name: str, cat_def: CategoryDef) -> str:
    """Hierarchically merge structured parts in groups."""
    current = parts
    round_num = 1
    while len(current) > 1:
        batches = [current[i:i + MERGE_BATCH_SIZE] for i in range(0, len(current), MERGE_BATCH_SIZE)]
        logger.info(f"    Merge round {round_num}: {len(current)} parts -> {len(batches)} merged (parallel)")
        # Batches in a round are independent; rounds stay sequential.
        current = _map_concurrent(
            lambda b: b[0] if len(b) == 1 else _merge_pair(b, university_name, cat_def),
            batches,
        )
        round_num += 1
    return current[0]


@task
def structure_content(scraped: dict[str, list[str]], university_name: str, categories: list[str]) -> dict[str, str]:
    """Use LLM to structure raw markdown into cohesive documents, processing in chunks."""
    structured: dict[str, str] = {}

    for cat in categories:
        cat_def = CATEGORY_REGISTRY[cat]
        pages = scraped.get(cat, [])
        if not pages:
            structured[cat] = f"# {university_name} - {cat.replace('_', ' ').title()}\n\nNo information available.\n"
            continue

        chunks = _chunk_pages(pages)
        logger.info(f"Structuring '{cat}' from {len(pages)} pages in {len(chunks)} chunks (parallel)...")

        parts = _map_concurrent(
            lambda ic: _structure_chunk(ic[1], university_name, cat_def, ic[0], len(chunks)),
            list(enumerate(chunks)),
        )

        if len(parts) == 1:
            # Single chunk — just add the heading
            result = parts[0]
            if not result.startswith(f"# {university_name}"):
                result = f"# {university_name} - {cat.replace('_', ' ').title()}\n\n{result}"
            structured[cat] = result
        else:
            logger.info(f"  Merging {len(parts)} parts for '{cat}'...")
            structured[cat] = _merge_structured(parts, university_name, cat_def)

        logger.info(f"  {cat}: structured to {len(structured[cat])} chars")

    return structured


# --- Page index -------------------------------------------------------------

# Each scraped page is stored as "<!-- Source: {url} -->\n{markdown}".
_SOURCE_RE = re.compile(r"^<!-- Source: (.*?) -->\n(.*)$", re.DOTALL)

INDEX_BATCH_SIZE = 15      # pages described per LLM call
INDEX_PAGE_CHARS = 2500    # chars of each page sent for description


def _parse_page(page: str) -> tuple[str, str]:
    """Split a stored page back into (url, content)."""
    m = _SOURCE_RE.match(page)
    if m:
        return m.group(1), m.group(2)
    return "", page


def _describe_batch(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Ask the LLM for a one-line description of each page. Maps url -> description.

    Pages are numbered and results keyed by number (not URL) so a model that
    rewrites a URL can't desync the mapping.
    """
    blocks = [
        f"[{i}] URL: {url}\n{content[:INDEX_PAGE_CHARS].strip()}"
        for i, (url, content) in enumerate(pairs, 1)
    ]
    prompt = f"""You are building a lookup index of web pages. For each numbered page below, write ONE concise sentence (max ~25 words) stating the specific information a reader would find there — concrete topics, data types, names, numbers. No marketing language, do not start with "This page".

Respond with ONLY valid JSON mapping each page number (as a string) to its description:
{{"1": "...", "2": "..."}}

Pages:

{chr(10).join(f"{b}{chr(10)}---" for b in blocks)}"""

    try:
        raw = llm_call(INDEXING_MODEL, prompt, max_tokens=4096, stage="indexing")
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"  Index batch failed ({len(pairs)} pages): {e}")
        data = {}

    return {
        pairs[i - 1][0]: str(data.get(str(i), "")).strip()
        for i in range(1, len(pairs) + 1)
    }


@task
def build_index(scraped: dict[str, list[str]], university_name: str) -> str:
    """Build index.md mapping every scraped page URL to a one-line description.

    Gives a downstream agent a routing table: when a query can't be answered
    from the category documents, it can look up which source page to fetch.
    """
    # Collect (url, content) pairs per category and flatten into LLM batches.
    cat_pairs: dict[str, list[tuple[str, str]]] = {}
    batches: list[list[tuple[str, str]]] = []
    for cat, pages in scraped.items():
        pairs = [p for p in (_parse_page(pg) for pg in pages) if p[0]]
        if not pairs:
            continue
        cat_pairs[cat] = pairs
        batches.extend(pairs[i:i + INDEX_BATCH_SIZE] for i in range(0, len(pairs), INDEX_BATCH_SIZE))

    total_pages = sum(len(p) for p in cat_pairs.values())
    if total_pages == 0:
        return f"# {university_name} — Page Index\n\nNo pages indexed.\n"

    logger.info(f"Indexing {total_pages} pages in {len(batches)} batches (parallel)...")
    descriptions: dict[str, str] = {}
    for result in _map_concurrent(_describe_batch, batches):
        descriptions.update(result)

    lines = [
        f"# {university_name} — Page Index",
        "",
        "Lookup table mapping each scraped page to a short description of its contents. "
        "Use it to find the source page for details not covered in the category documents.",
        "",
    ]
    for cat, pairs in cat_pairs.items():
        lines.append(f"## {cat.replace('_', ' ').title()}")
        lines.append("")
        for url, _ in pairs:
            desc = descriptions.get(url, "").strip() or "(no description available)"
            lines.append(f"- {url} — {desc}")
        lines.append("")

    logger.info(f"  Index built: {total_pages} pages indexed")
    return "\n".join(lines)


@task
def upload_to_s3(structured_markdown: dict[str, str], university_name: str) -> list[str]:
    """Upload structured markdown files to S3."""
    prefix = university_name.lower().replace(" ", "_")
    return upload_markdown_to_s3(structured_markdown, prefix)
