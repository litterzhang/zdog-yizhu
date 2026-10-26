"""Small, attributed species summaries, separate from private plant descriptions."""

import json
from pathlib import Path
from urllib.parse import urlsplit

PATH = Path(__file__).parent / "data/species-knowledge.json"


def validate_knowledge(entries):
    if not isinstance(entries, dict):
        raise ValueError("Species knowledge must be keyed by exact scientific name.")
    for name, item in entries.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            raise ValueError("Invalid species knowledge entry.")
        summary = item.get("summary")
        if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 1600:
            raise ValueError("Species summary must contain 1–1600 characters.")
        if not isinstance(item.get("isAiGenerated"), bool) or item.get("reviewStatus") not in {
            "unreviewed", "reviewed",
        }:
            raise ValueError("Summary generation/review status must be explicit.")
        sources = item.get("sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 5:
            raise ValueError("Species summaries need attributable sources.")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("url"), str):
                raise ValueError("Invalid summary source.")
            url = urlsplit(source["url"])
            if (url.scheme not in {"https", "http"} or not url.netloc or url.username
                    or url.password or not isinstance(source.get("title"), str)
                    or not source["title"].strip()):
                raise ValueError("Summary sources require a title and public HTTP(S) URL.")
    return entries


def load_knowledge(path=PATH):
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Knowledge file is too large.")
    return validate_knowledge(json.loads(path.read_text(encoding="utf-8")))
