"""Collect unseen media URLs from GBIF records even when the occurrence is already known."""

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from yizhu_server.gbif_target_collection import (
    ALLOWED_MEDIA_LICENSES,
    download,
    request_json,
)
from yizhu_server.training_data import difference_hash, now, write_json


def eligible_media(record, *, allow_by_sa):
    for media in record.get("media", []):
        license_url = (media.get("license") or "").lower().rstrip("/")
        url = media.get("identifier")
        author = (
            media.get("creator") or media.get("rightsHolder")
            or record.get("rightsHolder") or record.get("recordedBy")
            or media.get("publisher") or record.get("datasetName")
        )
        allowed = any(token in license_url for token in ALLOWED_MEDIA_LICENSES)
        allowed = allowed or (
            allow_by_sa and "creativecommons.org/licenses/by-sa/" in license_url
        )
        if (
            media.get("type") == "StillImage" and isinstance(url, str)
            and url.startswith("https://") and allowed and author
        ):
            yield media, str(author).strip()


def collect(targets_path, names_path, exclusions_path, output, *, per_species=5,
            max_pages=50, author_cap=3, allow_by_sa=True):
    from PIL import Image, ImageOps

    if output.exists() or not 1 <= per_species <= 20 or not 1 <= max_pages <= 100:
        raise ValueError("Use a new bounded additional-media collection.")
    targets = json.loads(targets_path.read_text())
    names = json.loads(names_path.read_text())["scientificNames"]
    exclusions = json.loads(exclusions_path.read_text())
    by_name = {row["scientificName"]: row for row in targets["classes"]}
    if len(names) != len(set(names)) or not set(names) <= set(by_name):
        raise ValueError("Require distinct reviewed target names.")
    excluded_urls = set(exclusions.get("sourceImageUrls", []))
    excluded_photos = {str(value) for value in exclusions.get("inatPhotoIds", [])}
    excluded_hashes = set(exclusions.get("imageSha256", []))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    write_json(output / "plan.json", {
        "createdAt": now(), "purpose": "independent-additional-media-image-level-disjoint",
        "scientificNames": names, "perSpeciesLimit": per_species,
        "maxPages": max_pages, "allowBySa": allow_by_sa,
        "knownOccurrencesAllowed": True, "trainingEligible": False,
    })
    candidates, rejected = [], Counter()
    seen_urls, seen_hashes, seen_dhashes = set(), set(), set()
    for position, name in enumerate(names, 1):
        target = by_name[name]
        match = request_json("species/match", {
            "name": name, "kingdom": "Plantae", "strict": "true",
        })
        query_key = match.get("usageKey")
        key = match.get("acceptedUsageKey") or match.get("speciesKey") or query_key
        if (
            match.get("confidence") != 100 or match.get("rank") != "SPECIES"
            or match.get("matchType") != "EXACT"
            or not isinstance(query_key, int) or not isinstance(key, int)
        ):
            rejected["species-resolution"] += 1
            continue
        accepted, offset = 0, 0
        author_counts = Counter()
        for _ in range(max_pages):
            page = request_json("occurrence/search", {
                "taxon_key": key, "media_type": "StillImage",
                "limit": 100, "offset": offset,
            })
            rows = page.get("results", [])
            if not rows:
                break
            for record in rows:
                if (
                    record.get("speciesKey") != key or record.get("acceptedTaxonKey") != key
                    or record.get("taxonRank") != "SPECIES"
                    or record.get("occurrenceStatus") != "PRESENT"
                ):
                    rejected["taxon-contract"] += 1
                    continue
                for media, author in eligible_media(record, allow_by_sa=allow_by_sa):
                    url = media["identifier"]
                    photo = re.search(r"/photos/(\d+)/(?:original|medium|large|small|square)", url)
                    if (
                        url in excluded_urls or url in seen_urls
                        or (photo and photo.group(1) in excluded_photos)
                    ):
                        rejected["known-media-url"] += 1
                        continue
                    author_group = f"{record.get('datasetKey')}:{author.casefold()}"
                    if author_counts[author_group] >= author_cap:
                        rejected["author-cap"] += 1
                        continue
                    try:
                        raw = download(url)
                        with Image.open(io.BytesIO(raw)) as source:
                            if min(source.size) < 100 or source.width * source.height > 40_000_000:
                                raise ValueError("Image dimensions outside collection bounds.")
                            image = ImageOps.exif_transpose(source).convert("RGB")
                            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                            buffer = io.BytesIO()
                            image.save(buffer, "JPEG", quality=90, optimize=True)
                            encoded = buffer.getvalue()
                            perceptual = difference_hash(image)
                    except Exception:
                        rejected["download-or-decode"] += 1
                        continue
                    digest = hashlib.sha256(encoded).hexdigest()
                    if (
                        digest in excluded_hashes or digest in seen_hashes
                        or perceptual in seen_dhashes
                    ):
                        rejected["known-or-duplicate-image"] += 1
                        continue
                    path = output / "images" / f"{digest}.jpg"
                    path.write_bytes(encoded)
                    occurrence = record.get("key")
                    candidate = {
                        "candidateId": f"gbif-media:{occurrence}:{digest[:16]}",
                        "speciesId": target["speciesId"],
                        "scientificName": target["scientificName"],
                        "chineseName": target["chineseName"],
                        "gbifQueryTaxonKey": query_key, "gbifAcceptedTaxonKey": key,
                        "gbifOccurrenceKey": occurrence,
                        "datasetKey": record.get("datasetKey"),
                        "basisOfRecord": record.get("basisOfRecord"),
                        "sourceReferences": record.get("references"),
                        "mediaCreator": media.get("creator"),
                        "mediaRightsHolder": media.get("rightsHolder"),
                        "mediaLicense": media.get("license"),
                        "sourceImageUrl": url, "imageSha256": digest,
                        "differenceHash": f"{perceptual:016x}",
                        "path": f"images/{path.name}",
                        "labelEvidence": "GBIF exact species occurrence; additional licensed media URL is image-level disjoint, while occurrence overlap is allowed and reported.",
                        "trainingEligible": False,
                    }
                    candidates.append(candidate)
                    seen_urls.add(url)
                    seen_hashes.add(digest)
                    seen_dhashes.add(perceptual)
                    author_counts[author_group] += 1
                    accepted += 1
                    if accepted >= per_species:
                        break
                if accepted >= per_species:
                    break
            if page.get("endOfRecords") or accepted >= per_species:
                break
            offset += len(rows)
        print(json.dumps({"scientificName": name, "accepted": accepted}), flush=True)
        write_json(output / "progress.json", {
            "updatedAt": now(), "stage": "collecting",
            "completedSpecies": position, "totalSpecies": len(names),
            "candidateImages": len(candidates),
            "counts": dict(Counter(row["speciesId"] for row in candidates)),
            "rejected": dict(rejected), "trainingEligible": False,
            "automaticAdmission": False,
        })
    write_json(output / "candidates.json", candidates)
    write_json(output / "progress.json", {
        "completedAt": now(), "stage": "collection-pass-finished",
        "candidateImages": len(candidates),
        "counts": dict(Counter(row["speciesId"] for row in candidates)),
        "rejected": dict(rejected), "trainingEligible": False,
        "automaticAdmission": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--names-file", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-species", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--author-cap", type=int, default=3)
    args = parser.parse_args()
    collect(args.targets.resolve(), args.names_file.resolve(), args.exclusions.resolve(),
            args.output.resolve(), per_species=args.per_species,
            max_pages=args.max_pages, author_cap=args.author_cap)


if __name__ == "__main__":
    main()
