import os
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
S3_BUCKET = os.environ.get("S3_BUCKET", "beyondcampus")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# litellm model strings (provider/model format)
# Swap provider prefix to switch: anthropic/claude-sonnet-4-6, openai/gpt-4.1-mini, etc.
CLASSIFICATION_MODEL = "openai/gpt-4.1-mini"
STRUCTURING_MODEL = "openai/gpt-4.1-mini"
