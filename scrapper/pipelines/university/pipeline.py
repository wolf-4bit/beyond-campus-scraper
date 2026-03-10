"""University scraping pipeline using LangGraph functional API."""
from __future__ import annotations

from langgraph.func import entrypoint

from scrapper.pipelines.university.schemas import DEFAULT_CATEGORIES
from scrapper.pipelines.university.tasks import (
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

    discovered = discover_urls(url).result()

    classified = classify_urls(discovered["urls"], categories).result()

    scraped = scrape_pages(classified, categories).result()

    structured = structure_content(scraped, discovered["university_name"], categories).result()

    keys = upload_to_s3(structured, discovered["university_name"]).result()

    return {
        "university_name": discovered["university_name"],
        "s3_keys": keys,
        "structured_markdown": structured,
    }
