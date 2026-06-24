"""University scraping pipeline using LangGraph functional API."""
from __future__ import annotations

from langgraph.func import entrypoint

from scrapper.core.cost import tracker
from scrapper.pipelines.university.schemas import DEFAULT_CATEGORIES
from scrapper.pipelines.university.tasks import (
    build_index,
    classify_urls,
    discover_urls,
    scrape_pages,
    structure_content,
    upload_to_s3,
)


@entrypoint()
def scrape_university(inputs: dict) -> dict:
    """Run the full scraping pipeline for a university URL.

    inputs:
        url: str - university website URL
        categories: list[str] | None - which categories to scrape (default: all)
    """
    url = inputs["url"]
    categories = inputs.get("categories") or DEFAULT_CATEGORIES

    tracker.reset()

    discovered = discover_urls(url).result()

    classified = classify_urls(discovered["urls"], categories, discovered["url_meta"]).result()

    scraped = scrape_pages(classified, categories).result()

    # Structuring and indexing both read `scraped` and are independent — run concurrently.
    structured_future = structure_content(scraped, discovered["university_name"], categories)
    index_future = build_index(scraped, discovered["university_name"])
    structured = structured_future.result()
    index_md = index_future.result()

    keys = upload_to_s3({**structured, "index": index_md}, discovered["university_name"]).result()

    return {
        "university_name": discovered["university_name"],
        "s3_keys": keys,
        "structured_markdown": structured,
        "index_markdown": index_md,
        "cost": tracker.as_dict(),
    }
