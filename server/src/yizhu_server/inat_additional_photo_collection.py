"""Collect extra licensed photos from known or new iNaturalist observations."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from yizhu_server.training_data import PublicClient, difference_hash, now, write_json

LICENSES = {"cc0", "cc-by", "cc-by-sa"}
PHOTO_HOST = "inaturalist-open-data.s3.amazonaws.com"


def collect(plan_path, exclusions_path, output, *, per_species=10, max_pages=30,
            author_cap=5):
    from PIL import Image, ImageOps

    if output.exists() or not 1 <= per_species <= 30 or not 1 <= max_pages <= 100:
        raise ValueError("Use a new bounded iNaturalist additional-photo collection.")
    plan = json.loads(plan_path.read_text())
    species = plan.get("species")
    if (
        plan.get("policy") != "reviewed-inat-additional-photos-v1"
        or not isinstance(species, list) or not species
        or any(type(row.get("inatTaxonId")) is not int for row in species)
    ):
        raise ValueError("Require reviewed exact iNaturalist species IDs.")
    exclusions = json.loads(exclusions_path.read_text())
    excluded_photos = {str(value) for value in exclusions.get("inatPhotoIds", [])}
    excluded_urls = set(exclusions.get("sourceImageUrls", []))
    excluded_hashes = set(exclusions.get("imageSha256", []))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    write_json(output / "plan.json", plan)
    client = PublicClient()
    candidates, rejected = [], Counter()
    seen_photos, seen_hashes, seen_dhashes = set(), set(), set()
    for position, target in enumerate(species, 1):
        accepted, author_counts = 0, Counter()
        for page in range(1, max_pages + 1):
            payload = client.json("observations", {
                "taxon_id": target["inatTaxonId"], "quality_grade": "research",
                "photos": "true", "photo_license": "cc0,cc-by,cc-by-sa",
                "per_page": 100, "page": page, "order_by": "id", "order": "desc",
            })
            observations = payload.get("results", [])
            if not observations:
                break
            for observation in observations:
                taxon = observation.get("taxon") or {}
                user = observation.get("user") or {}
                if (
                    observation.get("quality_grade") != "research"
                    or taxon.get("id") != target["inatTaxonId"]
                    or taxon.get("rank") != "species"
                    or type(observation.get("id")) is not int
                    or type(user.get("id")) is not int
                ):
                    rejected["observation-contract"] += 1
                    continue
                author_group = hashlib.sha256(
                    f"inat-observer:{user['id']}".encode()
                ).hexdigest()
                for photo in observation.get("photos", []):
                    photo_id = str(photo.get("id", ""))
                    license_code = photo.get("license_code")
                    parsed = urlparse(photo.get("url", ""))
                    if (
                        not photo_id.isdigit() or photo_id in excluded_photos
                        or photo_id in seen_photos or license_code not in LICENSES
                        or not photo.get("attribution") or photo.get("hidden")
                        or photo.get("flags") or parsed.scheme != "https"
                        or parsed.netloc != PHOTO_HOST or parsed.query or parsed.fragment
                        or Path(parsed.path).suffix.lower() not in {".jpg", ".jpeg", ".png"}
                    ):
                        rejected["photo-contract-or-known"] += 1
                        continue
                    image_url = (
                        f"https://{PHOTO_HOST}/photos/{photo_id}/medium"
                        f"{Path(parsed.path).suffix}"
                    )
                    if image_url in excluded_urls or author_counts[author_group] >= author_cap:
                        rejected["known-url-or-author-cap"] += 1
                        continue
                    try:
                        raw = client.get(image_url, limit=4 * 1024 * 1024)
                        with Image.open(io.BytesIO(raw)) as source:
                            if min(source.size) < 100:
                                raise ValueError("Image too small.")
                            image = ImageOps.exif_transpose(source).convert("RGB")
                            buffer = io.BytesIO()
                            image.save(buffer, "JPEG", quality=92)
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
                    license_url = (
                        "https://creativecommons.org/publicdomain/zero/1.0/"
                        if license_code == "cc0"
                        else f"https://creativecommons.org/licenses/{license_code[3:]}/4.0/"
                    )
                    candidates.append({
                        "candidateId": f"inat-photo:{photo_id}",
                        "speciesId": target["speciesId"],
                        "scientificName": target["scientificName"],
                        "chineseName": target["chineseName"],
                        "sourceObservationId": observation["id"],
                        "sourceImageUrl": image_url,
                        "sourceReferences": f"https://www.inaturalist.org/observations/{observation['id']}",
                        "basisOfRecord": "HUMAN_OBSERVATION",
                        "authorGroup": author_group,
                        "attribution": photo["attribution"],
                        "license": license_code, "licenseUrl": license_url,
                        "imageSha256": digest, "differenceHash": f"{perceptual:016x}",
                        "path": f"images/{path.name}",
                        "labelEvidence": "iNaturalist research-grade species observation; additional licensed photo is image-level disjoint while observation overlap is allowed and reported.",
                        "trainingEligible": False,
                    })
                    seen_photos.add(photo_id)
                    seen_hashes.add(digest)
                    seen_dhashes.add(perceptual)
                    author_counts[author_group] += 1
                    accepted += 1
                    if accepted >= per_species:
                        break
                if accepted >= per_species:
                    break
            if accepted >= per_species or len(observations) < 100:
                break
        print(json.dumps({"scientificName": target["scientificName"], "accepted": accepted}), flush=True)
        write_json(output / "progress.json", {
            "updatedAt": now(), "stage": "collecting", "completedSpecies": position,
            "totalSpecies": len(species), "candidateImages": len(candidates),
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-species", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--author-cap", type=int, default=5)
    args = parser.parse_args()
    collect(args.plan.resolve(), args.exclusions.resolve(), args.output.resolve(),
            per_species=args.per_species, max_pages=args.max_pages,
            author_cap=args.author_cap)


if __name__ == "__main__":
    main()
