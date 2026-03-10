"""CLI entry point for the university scraper."""
from __future__ import annotations

import argparse
import logging

from scrapper.pipelines.university.schemas import DEFAULT_CATEGORIES, CATEGORY_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Scrape a university website into structured markdown on S3")
    parser.add_argument("url", help="University website URL (e.g., smu.edu)")
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=list(CATEGORY_REGISTRY.keys()),
        default=None,
        help=f"Categories to scrape (default: all). Choices: {', '.join(DEFAULT_CATEGORIES)}",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--local", action="store_true", help="Save files locally instead of uploading to S3")
    args = parser.parse_args()

    categories = args.categories or DEFAULT_CATEGORIES

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.local:
        _run_local(args.url, categories)
    else:
        from scrapper.pipelines.university.pipeline import scrape_university
        result = scrape_university.invoke({"url": args.url, "categories": categories})
        print(f"\nDone! Uploaded {len(result['s3_keys'])} files:")
        for key in result["s3_keys"]:
            print(f"  s3://{key}")


def _run_local(url: str, categories: list[str]):
    """Run pipeline but save to local files instead of S3."""
    import os

    from scrapper.pipelines.university.tasks import (
        classify_urls,
        discover_urls,
        scrape_pages,
        structure_content,
    )

    discovered = discover_urls.__wrapped__(url)
    classified = classify_urls.__wrapped__(discovered["urls"], categories)
    scraped = scrape_pages.__wrapped__(classified, categories)
    structured = structure_content.__wrapped__(scraped, discovered["university_name"], categories)

    name_slug = discovered["university_name"].lower().replace(" ", "_")
    out_dir = os.path.join("output", name_slug)
    os.makedirs(out_dir, exist_ok=True)

    for cat, md in structured.items():
        path = os.path.join(out_dir, f"{cat}.md")
        with open(path, "w") as f:
            f.write(md)
        print(f"  Wrote {path}")

    print(f"\nDone! Saved {len(structured)} files to {out_dir}/")


if __name__ == "__main__":
    main()
