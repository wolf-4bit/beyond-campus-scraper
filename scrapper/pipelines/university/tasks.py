"""University pipeline tasks."""
from __future__ import annotations

import json
import logging
from urllib.parse import urlparse

from langgraph.func import task

from scrapper.core.config import CLASSIFICATION_MODEL, STRUCTURING_MODEL
from scrapper.core.llm import llm_call
from scrapper.core.scraper import firecrawl, scrape_urls
from scrapper.core.storage import upload_markdown_to_s3
from scrapper.pipelines.university.schemas import CATEGORY_REGISTRY, CategoryDef

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


@task
def discover_urls(university_url: str) -> dict:
    """Use Firecrawl /map to discover all URLs on the site."""
    url = university_url
    if not url.startswith("http"):
        url = f"https://{url}"

    logger.info(f"Mapping URLs for {url}...")
    result = firecrawl.map(url=url)

    links = result.links if hasattr(result, "links") else result.get("links", [])
    urls = [link["url"] if isinstance(link, dict) else link.url for link in links]
    logger.info(f"Discovered {len(urls)} URLs")

    domain = urlparse(url).netloc.replace("www.", "")
    name = domain.split(".")[0].upper()

    return {"urls": urls, "university_name": name, "university_url": url}


def _build_classify_prompt(categories: list[CategoryDef]) -> str:
    """Build classification prompt dynamically from selected categories."""
    cat_lines = "\n".join(
        f"- {cat.name}: {cat.classify_hint}" for cat in categories
    )
    cat_json = ",\n  ".join(f'"{cat.name}": ["url1", "url2"]' for cat in categories)
    return f"""Classify these university website URLs into categories.
Each URL should go into AT MOST one category. Skip irrelevant URLs (login pages, news, events, social media, generic pages).
Only include URLs that are clearly relevant to the category.

Categories:
{cat_lines}

URLs:
{{url_text}}

Respond with ONLY valid JSON matching this exact structure (no other text):
{{{{
  {cat_json}
}}}}"""


def _classify_batch(url_batch: list[str], prompt_template: str, cat_names: list[str]) -> dict[str, list[str]]:
    """Classify a single batch of URLs."""
    url_text = "\n".join(url_batch)
    raw = llm_call(CLASSIFICATION_MODEL, prompt_template.format(url_text=url_text), max_tokens=8192)

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
def classify_urls(urls: list[str], categories: list[str]) -> dict[str, list[str]]:
    """Use LLM to classify discovered URLs into selected categories, in batches."""
    if not urls:
        raise ValueError("No URLs discovered to classify")

    cat_defs = [CATEGORY_REGISTRY[c] for c in categories]
    cat_names = [c.name for c in cat_defs]
    prompt_template = _build_classify_prompt(cat_defs)

    batches = [urls[i:i + BATCH_SIZE] for i in range(0, len(urls), BATCH_SIZE)]
    logger.info(f"Classifying {len(urls)} URLs in {len(batches)} batches for categories: {cat_names}")

    results = []
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i + 1}/{len(batches)} ({len(batch)} URLs)...")
        results.append(_classify_batch(batch, prompt_template, cat_names))

    classified = _merge_batches(results, cat_names)

    # Trim to max per category
    for cat_def in cat_defs:
        if len(classified[cat_def.name]) > cat_def.max_urls:
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

    return llm_call(STRUCTURING_MODEL, prompt, max_tokens=16384)


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

    return llm_call(STRUCTURING_MODEL, prompt, max_tokens=16384)


def _merge_structured(parts: list[str], university_name: str, cat_def: CategoryDef) -> str:
    """Hierarchically merge structured parts in groups."""
    current = parts
    round_num = 1
    while len(current) > 1:
        logger.info(f"    Merge round {round_num}: {len(current)} parts -> ~{(len(current) + MERGE_BATCH_SIZE - 1) // MERGE_BATCH_SIZE} merged")
        next_level = []
        for i in range(0, len(current), MERGE_BATCH_SIZE):
            batch = current[i:i + MERGE_BATCH_SIZE]
            if len(batch) == 1:
                next_level.append(batch[0])
            else:
                next_level.append(_merge_pair(batch, university_name, cat_def))
        current = next_level
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
        logger.info(f"Structuring '{cat}' from {len(pages)} pages in {len(chunks)} chunks...")

        parts = []
        for i, chunk in enumerate(chunks):
            chunk_chars = sum(len(p) for p in chunk)
            logger.info(f"  Chunk {i + 1}/{len(chunks)}: {len(chunk)} pages, {chunk_chars} chars")
            part = _structure_chunk(chunk, university_name, cat_def, i, len(chunks))
            parts.append(part)

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


@task
def upload_to_s3(structured_markdown: dict[str, str], university_name: str) -> list[str]:
    """Upload structured markdown files to S3."""
    prefix = university_name.lower().replace(" ", "_")
    return upload_markdown_to_s3(structured_markdown, prefix)
