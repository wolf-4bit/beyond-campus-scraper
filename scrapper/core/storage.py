"""Shared S3 upload utilities."""
from __future__ import annotations

import logging

import boto3

from scrapper.core.config import AWS_REGION, S3_BUCKET

logger = logging.getLogger(__name__)


def upload_markdown_to_s3(files: dict[str, str], prefix: str) -> list[str]:
    """Upload a dict of {filename: content} to S3 under the given prefix.

    Returns list of uploaded S3 keys.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    keys = []

    for name, content in files.items():
        if not content:
            continue
        key = f"{prefix}/{name}.md"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )
        keys.append(key)
        logger.info(f"Uploaded s3://{S3_BUCKET}/{key}")

    return keys
