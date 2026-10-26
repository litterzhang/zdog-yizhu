"""Collect strict scientific-name Commons search results as review-only candidates."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from yizhu_server.commons_plant_collection import API, inspect_page
from yizhu_server.commons_transport import CommonsClient
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import DuplicateIndex, difference_hash, now, write_json

POLICY = "commons-explicit-species-search-v1"


def collect(plan_path, output, *, max_per_species=20, max_pages=3, allow_by_sa=False):
    from PIL import Image

    plan = json.loads(plan_path.read_text())
    species = plan.get("species")
    if (
        plan.get("policy") != POLICY or not isinstance(species, list) or not species
        or not 1 <= max_per_species <= 50 or not 1 <= max_pages <= 20
        or any(
            not row.get("speciesId") or not row.get("scientificName")
            or not row.get("searchScientificName")
            or not row.get("searchQuery", row.get("searchScientificName"))
            for row in species
        )
    ):
        raise ValueError("Require a bounded reviewed Commons species search plan.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    (output / "images").mkdir()
    (output / "metadata").mkdir()
    client, candidates, duplicates, rejected = CommonsClient(), [], DuplicateIndex(), Counter()
    for position, target in enumerate(species, 1):
        query_target = {
            **target,
            "commonsScientificName": target["searchScientificName"],
            "commonsCategory": "Commons search: " + target["searchScientificName"],
        }
        continuation = {}
        accepted = 0
        for page_number in range(max_pages):
            params = {
                "action": "query", "format": "json", "formatversion": 2,
                "generator": "search",
                "gsrsearch": f'"{target.get("searchQuery", target["searchScientificName"])}"',
                "gsrnamespace": 6, "gsrlimit": 50,
                "prop": "imageinfo", "iiprop": "url|extmetadata|sha1|size|mime",
                "iiurlwidth": 768,
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|"
                                          "ImageDescription|Restrictions|AttributionRequired",
                **continuation,
            }
            payload = json.loads(client.get(API, params=params))
            write_json(output / "metadata" / f"{position}-{page_number}.json", payload)
            for page in payload.get("query", {}).get("pages", []):
                candidate, rejection = inspect_page(
                    page, query_target, allow_by_sa=allow_by_sa,
                )
                if rejection:
                    rejected[rejection] += 1
                    continue
                reserve_space(output, 12 * 1024**2)
                content = client.get(candidate["imageUrl"], limit=8 * 1024**2)
                with Image.open(io.BytesIO(content)) as image:
                    if min(image.size) < 100 or image.width * image.height > 16_000_000:
                        rejected["image-dimensions"] += 1
                        continue
                    image = image.convert("RGB")
                    image.thumbnail((768, 768))
                    perceptual = difference_hash(image)
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=92)
                data = buffer.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                if duplicates.contains(digest, perceptual):
                    rejected["duplicate-or-near-duplicate"] += 1
                    continue
                path = output / "images" / f"{digest}.jpg"
                path.write_bytes(data)
                candidates.append({
                    **candidate, "scientificName": target["scientificName"],
                    "sourceScientificName": target["searchScientificName"],
                    "sourceNameEvidence": target.get("sourceNameEvidence", {}),
                    "path": f"images/{path.name}", "imageSha256": digest,
                    "differenceHash": f"{perceptual:016x}", "downloadedAt": now(),
                    "modifications": "RGB JPEG re-encode, max 768px, metadata removed",
                })
                duplicates.add(digest, perceptual)
                accepted += 1
                if accepted >= max_per_species:
                    break
            write_json(output / "candidates.json", candidates)
            continuation = payload.get("continue")
            if not continuation or accepted >= max_per_species:
                break
        print(f"{position}/{len(species)} {target['scientificName']}: {accepted}", flush=True)
    result = {
        "completedAt": now(), "stage": "collection-pass-finished",
        "speciesInPass": len(species), "downloadedCandidateImages": len(candidates),
        "counts": dict(Counter(row["speciesId"] for row in candidates)),
        "rejected": dict(rejected), "trainingEligible": False, "onlineChanged": False,
    }
    write_json(output / "progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-species", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--allow-by-sa", action="store_true")
    args = parser.parse_args()
    collect(
        args.plan.resolve(), args.output.resolve(),
        max_per_species=args.max_per_species, max_pages=args.max_pages,
        allow_by_sa=args.allow_by_sa,
    )


if __name__ == "__main__":
    main()
