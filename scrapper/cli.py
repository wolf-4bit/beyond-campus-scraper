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
        from scrapper.core.cost import tracker
        from scrapper.pipelines.university.pipeline import scrape_university
        result = scrape_university.invoke({"url": args.url, "categories": categories})
        print(f"\nDone! Uploaded {len(result['s3_keys'])} files:")
        for key in result["s3_keys"]:
            print(f"  s3://{key}")
        print(f"\n{tracker.summary()}")


def _run_local(url: str, categories: list[str]):
    """Run pipeline but save to local files instead of S3."""
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    from scrapper.core.cost import tracker
    from scrapper.pipelines.university.tasks import (
        build_index,
        classify_urls,
        discover_urls,
        scrape_pages,
        structure_content,
    )

    tracker.reset()

    discovered = discover_urls.__wrapped__(url)
    classified = classify_urls.__wrapped__(discovered["urls"], categories, discovered["url_meta"])
    scraped = scrape_pages.__wrapped__(classified, categories)

    # Structure content and build the page index concurrently — both read `scraped`.
    name = discovered["university_name"]
    with ThreadPoolExecutor(max_workers=2) as ex:
        structured_f = ex.submit(structure_content.__wrapped__, scraped, name, categories)
        index_f = ex.submit(build_index.__wrapped__, scraped, name)
        structured = structured_f.result()
        index_md = index_f.result()

    name_slug = discovered["university_name"].lower().replace(" ", "_")
    out_dir = os.path.join("output", name_slug)
    os.makedirs(out_dir, exist_ok=True)

    for cat, md in structured.items():
        path = os.path.join(out_dir, f"{cat}.md")
        with open(path, "w") as f:
            f.write(md)
        print(f"  Wrote {path}")

    index_path = os.path.join(out_dir, "index.md")
    with open(index_path, "w") as f:
        f.write(index_md)
    print(f"  Wrote {index_path}")

    cost_path = os.path.join(out_dir, "cost.json")
    with open(cost_path, "w") as f:
        json.dump(tracker.as_dict(), f, indent=2)
    print(f"  Wrote {cost_path}")

    print(f"\nDone! Saved {len(structured)} files to {out_dir}/")
    print(f"\n{tracker.summary()}")


if __name__ == "__main__":
    main()
