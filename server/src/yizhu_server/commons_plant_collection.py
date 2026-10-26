"""Collect licensed Commons photo candidates separately; never auto-label them as training data."""

import argparse
import hashlib
import html
import io
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from yizhu_server.commons_transport import CommonsClient, SourceRestriction, require_resume
from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import (
    DuplicateIndex,
    dataset_lock,
    difference_hash,
    load_samples,
    now,
    require_dataset_active,
    sha256,
    write_json,
)

API = "https://commons.wikimedia.org/w/api.php"
MEDIA_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}


def plain(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value))).strip()


def photo_license(metadata, *, allow_by_sa=False):
    """Check media-level URL, never infer permissions from the site's general license."""
    url = metadata.get("LicenseUrl", {}).get("value", "")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "creativecommons.org":
        return None
    path = parsed.path.rstrip("/")
    suffix = r"(?:/[a-z]{2}(?:_[A-Z]{2})?)?(?:/deed\.[a-z-]+)?"
    match = re.fullmatch(
        rf"/licenses/(by|by-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0){suffix}", path,
    )
    if match and (match.group(1) == "by" or allow_by_sa):
        kind = "cc-by" if match.group(1) == "by" else "cc-by-sa"
        return kind, f"https://creativecommons.org/licenses/{match.group(1)}/{match.group(2)}/"
    if re.fullmatch(rf"/publicdomain/zero/1\.0{suffix}", path):
        return "cc0", "https://creativecommons.org/publicdomain/zero/1.0/"
    return None


def inspect_page(page, species, *, allow_by_sa=False):
    infos = page.get("imageinfo", [])
    if len(infos) != 1 or type(page.get("pageid")) is not int:
        return None, "missing-image-metadata"
    info = infos[0]
    meta = info.get("extmetadata", {})
    permission = photo_license(meta, allow_by_sa=allow_by_sa)
    if not permission:
        return None, "license-not-allowlisted"
    if plain(meta.get("Restrictions", {}).get("value", "")):
        return None, "additional-restrictions"
    artist = plain(meta.get("Artist", {}).get("value", ""))
    if not artist or artist.lower() in {"unknown", "anonymous", "unknown author"}:
        return None, "missing-attribution"
    description = plain(meta.get("ImageDescription", {}).get("value", ""))
    text = plain(page.get("title", "") + " " + description).replace("_", " ")
    source_name = species.get("commonsScientificName", species["scientificName"])
    normalized_text = re.sub(r"\s+", " ", text.replace("×", "x")).casefold()
    normalized_name = re.sub(r"\s+", " ", source_name.replace("×", "x")).casefold()
    scientific_name_present = re.search(
        r"(?<![a-z])" + re.escape(normalized_name) + r"(?![a-z])",
        normalized_text,
    ) is not None
    chinese_names = [
        value.strip() for value in [
            species.get("chineseName"), *species.get("commonsChineseNames", []),
        ]
        if isinstance(value, str) and value.strip()
    ]
    chinese_name_present = any(value in text for value in chinese_names)
    if not scientific_name_present and not chinese_name_present:
        return None, "no-explicit-scientific-name"
    if re.search(
        r"\b(etching|engraving|drawing|illustration|herbarium|specimen|distribution map|"
        r"dried|tea|painting|stamp|pollen|microscope|seed packet)\b|标本|绘画|插画|干花|茶叶",
        text, re.I,
    ):
        return None, "not-a-live-plant-photo"
    url = info.get("thumburl", "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or parsed.hostname not in MEDIA_HOSTS
        or parsed.username or parsed.password or parsed.port not in {None, 443}
        or not parsed.path.startswith("/wikipedia/commons/")
        or info.get("thumbmime", info.get("mime")) not in {"image/jpeg", "image/png"}
    ):
        return None, "unsupported-image-location-or-format"
    category = species.get("commonsCategory", "Category:" + species["scientificName"])
    return {
        "id": f"commons:{page['pageid']}", "speciesId": species["speciesId"],
        "scientificName": species["scientificName"], "chineseName": species["chineseName"],
        "sourceScientificName": source_name,
        **({"sourceNameEvidence": species["commonsSourceNameEvidence"]}
           if species.get("commonsSourceNameEvidence") else {}),
        "source": "Wikimedia Commons", "sourceUrl": info["descriptionurl"],
        "imageUrl": url, "sourceSha1": info.get("sha1"),
        "license": permission[0], "licenseUrl": permission[1], "attribution": artist,
        "sourceTitle": page["title"], "sourceCategory": category,
        "authorGroup": hashlib.sha256(("commons-author:" + artist.casefold()).encode()).hexdigest(),
        "labelEvidence": (
            "Direct species-category membership and explicit "
            + ("scientific name" if scientific_name_present else "target Chinese name")
            + " in title/description; community metadata, not expert-audited labels."
        ),
        "sourceNameMatch": (
            "scientific-name" if scientific_name_present else "target-chinese-name"
        ),
        "labelReviewStatus": "metadata-only", "trainingEligible": False,
        "split": "unassigned", "observedAt": now(),
    }, None


def collect(targets_path, primary, output, *, max_per_species=100, max_pages=5, class_limit=None,
            resume_after_cooldown=False, resume_transport_error=False,
            allow_by_sa=False, names=None, author_cap=3):
    from PIL import Image

    if (
        not 1 <= max_per_species <= 500 or not 1 <= max_pages <= 20
        or not 1 <= author_cap <= 10
    ):
        raise ValueError("Use bounded photo/page limits.")
    require_dataset_active(primary)
    target = json.loads(targets_path.read_text())
    rows = load_samples(primary)
    if not rows and (primary / "train-validation.json").is_file():
        training = json.loads((primary / "train-validation.json").read_text())
        heldout = json.loads((primary / "test-heldout.json").read_text())
        rows = training["samples"] + heldout["samples"]
    counts = {}
    for row in rows:
        counts.setdefault(row["speciesId"], Counter())[row["split"]] += 1
    species = [
        row for row in target["classes"]
        if any(counts.get(row["speciesId"], {}).get(k, 0) < v
               for k, v in {"train": 70, "validation": 15, "test": 15}.items())
    ]
    if names is not None:
        if len(names) != len(set(names)):
            raise ValueError("Commons target names must be distinct.")
        requested = set(names)
        species = [row for row in species if row["scientificName"] in requested]
        if {row["scientificName"] for row in species} != requested:
            raise ValueError("Commons target names are absent or not eligible for supplementation.")
    species.sort(key=lambda r: (r["scientificName"] != "Osmanthus fragrans",
                               sum(counts.get(r["speciesId"], {}).values()),
                               r["scientificName"]))
    identity = {
        "targetManifestSha256": sha256(targets_path), "primarySpeciesSha256": sha256(primary / "species.json"),
        "maxPages": max_pages, "maxPerSpecies": max_per_species,
        "authorCap": author_cap,
    }
    with dataset_lock(output):
        require_dataset_active(output)
        stopped_path = output / "stopped.json"
        stopped = json.loads(stopped_path.read_text()) if stopped_path.exists() else None
        require_resume(stopped, requested=resume_after_cooldown or resume_transport_error,
                       allow_transport_retry=resume_transport_error)
        plan_path = output / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan["identity"] != identity:
                raise ValueError("Collection policy changed; use a separate dataset.")
            species = plan["species"]
        else:
            if class_limit is not None:
                if class_limit < 1:
                    raise ValueError("Class limit must be positive.")
                species = species[:class_limit]
            write_json(plan_path, {
                "createdAt": now(), "identity": identity, "species": species,
                "purpose": "Supplemental raw-photo candidates, not trained model support",
                "automaticTraining": False, "primaryDatasetChanged": False,
            })
        candidate_path = output / "candidates.json"
        candidates = json.loads(candidate_path.read_text()) if candidate_path.exists() else []
        duplicates = DuplicateIndex()
        for row in rows:
            duplicates.add(row["imageSha256"], int(row["differenceHash"], 16))
        for row in candidates:
            if sha256(output / row["path"]) != row["imageSha256"]:
                raise ValueError("Existing supplemental photo changed.")
            duplicates.add(row["imageSha256"], int(row["differenceHash"], 16))
        seen = {row["id"] for row in candidates}
        author_counts = Counter((row["speciesId"], row["authorGroup"]) for row in candidates)
        photo_counts = Counter(row["speciesId"] for row in candidates)
        progress_path = output / "progress.json"
        previous_progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        rejected = Counter(previous_progress.get("rejected", {}))
        if stopped:
            write_json(output / "last-resume.json", {
                "resumedAt": now(), "previousStop": stopped,
                "requestIntervalSeconds": 10, "newRunExplicitlyRequested": True,
                "transportFailureInspected": resume_transport_error,
            })
            # Preserve the actual old stop in last-resume; a subsequent failure writes a new stop.
            stopped_path.unlink()
        client = CommonsClient()
        for position, sp in enumerate(species, 1):
            require_dataset_active(output)
            stem = hashlib.sha256(sp["speciesId"].encode()).hexdigest()
            completion = output / "completed" / (stem + ".json")
            if completion.exists():
                continue
            continuation = {}
            for page_number in range(max_pages):
                cache = output / "metadata" / f"{stem}-{page_number}.json"
                if cache.exists():
                    payload = json.loads(cache.read_text())
                else:
                    params = {
                        "action": "query", "format": "json", "formatversion": 2,
                        "generator": "categorymembers",
                        "gcmtitle": sp.get(
                            "commonsCategory", "Category:" + sp["scientificName"]
                        ),
                        "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo",
                        "iiprop": "url|extmetadata|sha1|size|mime", "iiurlwidth": 768,
                        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|"
                                              "ImageDescription|Restrictions|AttributionRequired",
                        **continuation,
                    }
                    payload = json.loads(client.get(API, params=params))
                    if "error" in payload:
                        raise ValueError("Commons API returned an error.")
                    write_json(cache, payload)
                for page in payload.get("query", {}).get("pages", []):
                    candidate, rejection = inspect_page(
                        page, sp, allow_by_sa=allow_by_sa,
                    )
                    if rejection:
                        rejected[rejection] += 1
                        continue
                    if candidate["id"] in seen or photo_counts[sp["speciesId"]] >= max_per_species:
                        continue
                    author_key = (sp["speciesId"], candidate["authorGroup"])
                    if author_counts[author_key] >= author_cap:
                        rejected["author-cap"] += 1
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
                    relative = f"images/{digest}.jpg"
                    path = output / relative
                    path.parent.mkdir(exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(data)
                    candidates.append({
                        **candidate, "path": relative, "imageSha256": digest,
                        "differenceHash": f"{perceptual:016x}", "downloadedAt": now(),
                        "modifications": "RGB JPEG re-encode, max 768px, metadata removed",
                    })
                    duplicates.add(digest, perceptual)
                    seen.add(candidate["id"])
                    author_counts[author_key] += 1
                    photo_counts[sp["speciesId"]] += 1
                    write_json(candidate_path, candidates)
                write_json(output / "progress.json", {
                    "updatedAt": now(), "stage": "collecting", "position": position,
                    "speciesInPass": len(species), "currentScientificName": sp["scientificName"],
                    "downloadedCandidateImages": len(candidates),
                    "currentSpeciesImages": photo_counts[sp["speciesId"]],
                    "rejected": dict(rejected), "trainingEligible": False,
                })
                continuation = payload.get("continue")
                if not continuation or photo_counts[sp["speciesId"]] >= max_per_species:
                    break
            write_json(completion, {"completedAt": now(), "speciesId": sp["speciesId"],
                                    "candidateImages": photo_counts[sp["speciesId"]]})
            print(f"{position}/{len(species)} {sp['scientificName']}: "
                  f"{photo_counts[sp['speciesId']]} candidate photos", flush=True)
        result = {
            "completedAt": now(), "stage": "collection-pass-finished",
            "speciesInPass": len(species), "downloadedCandidateImages": len(candidates),
            "coveredSpecies": len(photo_counts), "counts": dict(photo_counts),
            "rejected": dict(rejected), "trainingEligible": False, "onlineChanged": False,
            "nextStep": "Verify species labels and cross-source author/split leakage before "
                        "admitting photos into a later training snapshot.",
        }
        write_json(output / "progress.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-limit", type=int)
    parser.add_argument("--max-per-species", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--author-cap", type=int, default=3)
    parser.add_argument("--allow-by-sa", action="store_true")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--resume-after-cooldown", action="store_true")
    parser.add_argument("--resume-transport-error", action="store_true",
                        help="Explicitly resume an inspected connection failure after cooldown")
    args = parser.parse_args()
    if args.names and args.names_file:
        parser.error("Use --names or --names-file, not both.")
    names = args.names
    if args.names_file:
        names = json.loads(args.names_file.read_text())["scientificNames"]
    # A rejected early resume must not overwrite the original server cooldown.
    stopped_path = args.output / "stopped.json"
    require_resume(
        json.loads(stopped_path.read_text()) if stopped_path.exists() else None,
        requested=args.resume_after_cooldown or args.resume_transport_error,
        allow_transport_retry=args.resume_transport_error,
    )
    try:
        collect(args.targets, args.primary, args.output,
                class_limit=args.class_limit, max_pages=args.max_pages,
                max_per_species=args.max_per_species,
                allow_by_sa=args.allow_by_sa,
                names=names,
                author_cap=args.author_cap,
                resume_after_cooldown=args.resume_after_cooldown,
                resume_transport_error=args.resume_transport_error)
    except Exception as exc:
        stopped = {
            "stoppedAt": now(), "errorType": type(exc).__name__, "automaticRetry": False,
            "sourceRestriction": isinstance(exc, PermissionError),
        }
        if isinstance(exc, SourceRestriction):
            stopped.update({
                "httpStatus": exc.status, "retryAfter": exc.retry_after,
                "retryNotBefore": exc.not_before,
            })
        write_json(args.output / "stopped.json", stopped)
        progress_path = args.output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        write_json(progress_path, {
            **progress, **stopped, "updatedAt": now(),
            "stage": "stopped-source-restriction" if stopped["sourceRestriction"] else "failed",
        })
        raise


if __name__ == "__main__":
    main()
