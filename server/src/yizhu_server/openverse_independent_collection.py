"""Collect exact-name licensed images from Openverse for independent evaluation."""

import argparse
import hashlib
import io
import ipaddress
import json
import re
import socket
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

from yizhu_server.training_data import difference_hash, now, write_json

API = "https://api.openverse.org/v1/images/"
LICENSES = {"cc0", "by", "by-sa"}
ALLOWED_HOSTS = {
    "live.staticflickr.com", "upload.wikimedia.org",
    "inaturalist-open-data.s3.amazonaws.com", "ids.si.edu",
}


def normalized(value):
    return re.sub(r"[^a-z]+", " ", value.replace("×", "x").casefold()).strip()


def explicit_name(result, names):
    text = " ".join([
        str(result.get("title") or ""),
        " ".join(tag.get("name", "") for tag in result.get("tags") or []),
    ])
    haystack = normalized(text)
    return any(normalized(name) in haystack for name in names)



def download_variant(value):
    parsed = urlparse(value or "")
    if parsed.hostname == "inaturalist-open-data.s3.amazonaws.com":
        return re.sub(r"/original(?=\.[A-Za-z]+$)", "/medium", value)
    return value


def safe_image_url(value):
    parsed = urlparse(value or "")
    if (
        parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username or parsed.password or parsed.port not in {None, 443}
    ):
        return None
    for address in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            return None
    return value


def collect(plan_path, exclusions_path, output, *, per_species=10, pages=5,
            author_cap=3):
    from PIL import Image, ImageOps

    if output.exists() or not 1 <= per_species <= 30 or not 1 <= pages <= 20:
        raise ValueError("Use a new bounded Openverse collection.")
    plan = json.loads(plan_path.read_text())
    species = plan.get("species")
    if (
        plan.get("policy") != "openverse-exact-species-independent-v1"
        or not isinstance(species, list) or not species
        or any(not row.get("speciesId") or not row.get("scientificName") for row in species)
    ):
        raise ValueError("Require reviewed species queries.")
    exclusions = json.loads(exclusions_path.read_text())
    excluded_urls = set(exclusions.get("sourceImageUrls", []))
    excluded_hashes = set(exclusions.get("imageSha256", []))
    excluded_photos = {str(value) for value in exclusions.get("inatPhotoIds", [])}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    write_json(output / "plan.json", plan)
    session = requests.Session()
    session.headers["User-Agent"] = "Yizhu independent plant evaluation/1.0"
    candidates, rejected = [], Counter()
    seen_ids, seen_urls, seen_hashes, seen_dhashes = set(), set(), set(), set()
    for position, target in enumerate(species, 1):
        accepted, authors = 0, Counter()
        query_names = (
            list(target.get("searchAliases", []))
            if target.get("searchOnlyAliases") is True
            else [target["scientificName"], *target.get("searchAliases", [])]
        )
        for query in query_names:
            for page in range(1, pages + 1):
                time.sleep(1)
                response = session.get(API, params={
                    "q": query, "page": page, "page_size": 20,
                    "license": "cc0,by,by-sa",
                }, timeout=(15, 60))
                if response.status_code in {403, 429}:
                    raise PermissionError(f"Openverse source paused: HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results", [])
                for result in results:
                    identifier = str(result.get("id") or "")
                    image_url = safe_image_url(download_variant(result.get("url")))
                    landing = result.get("foreign_landing_url")
                    creator = str(result.get("creator") or "").strip()
                    license_code = result.get("license")
                    license_url = result.get("license_url")
                    if (
                        not identifier or identifier in seen_ids or not image_url
                        or image_url in excluded_urls or image_url in seen_urls
                        or not creator or license_code not in LICENSES
                        or not isinstance(license_url, str)
                        or not license_url.startswith("https://creativecommons.org/")
                        or not explicit_name(result, query_names)
                    ):
                        rejected["metadata-name-license-or-location"] += 1
                        continue
                    photo = re.search(r"/photos/(\d+)/(?:original|medium|large|small|square)", image_url)
                    if photo and photo.group(1) in excluded_photos:
                        rejected["known-inaturalist-photo"] += 1
                        continue
                    author_group = hashlib.sha256(
                        f"openverse:{result.get('source')}:{creator.casefold()}".encode()
                    ).hexdigest()
                    if authors[author_group] >= author_cap:
                        rejected["author-cap"] += 1
                        continue
                    try:
                        with session.get(image_url, timeout=(15, 90), stream=True) as image_response:
                            image_response.raise_for_status()
                            data = bytearray()
                            for chunk in image_response.iter_content(65536):
                                data.extend(chunk)
                                if len(data) > 12 * 1024 * 1024:
                                    raise ValueError("Image too large.")
                        with Image.open(io.BytesIO(data)) as source:
                            if min(source.size) < 100 or source.width * source.height > 40_000_000:
                                raise ValueError("Image dimensions outside bounds.")
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
                    candidates.append({
                        "candidateId": f"openverse:{identifier}",
                        "speciesId": target["speciesId"],
                        "scientificName": target["scientificName"],
                        "chineseName": target["chineseName"],
                        "basisOfRecord": (
                            "PRESERVED_SPECIMEN"
                            if urlparse(image_url).hostname == "ids.si.edu"
                            else "HUMAN_OBSERVATION"
                        ),
                        "sourceRecordId": identifier,
                        "sourceImageUrl": image_url,
                        "sourceReferences": landing,
                        "authorGroup": author_group, "attribution": creator,
                        "license": license_code, "licenseUrl": license_url,
                        "imageSha256": digest, "differenceHash": f"{perceptual:016x}",
                        "path": f"images/{path.name}",
                        "labelEvidence": "Openverse result with explicit reviewed scientific name in title/tags and CC0/CC BY/CC BY-SA media metadata; not expert re-audited.",
                        "trainingEligible": False,
                    })
                    seen_ids.add(identifier)
                    seen_urls.add(image_url)
                    seen_hashes.add(digest)
                    seen_dhashes.add(perceptual)
                    authors[author_group] += 1
                    accepted += 1
                    if accepted >= per_species:
                        break
                if accepted >= per_species or not payload.get("next") or not results:
                    break
            if accepted >= per_species:
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
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--author-cap", type=int, default=3)
    args = parser.parse_args()
    collect(args.plan.resolve(), args.exclusions.resolve(), args.output.resolve(),
            per_species=args.per_species, pages=args.pages,
            author_cap=args.author_cap)


if __name__ == "__main__":
    main()
