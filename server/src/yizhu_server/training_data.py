"""Offline, resumable public-data collection. Never reads journal/user storage."""

import argparse
import csv
import fcntl
import hashlib
import json
import shutil
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / ".cache/training/china-common-1000"
CATALOG = ROOT / ".cache/catalog/treeoflife-10m/catalog.sqlite3"
API = "https://api.inaturalist.org/v1"
POLICY = "china-inat-research-visible-exact-v1"
PHOTO_HOST = "inaturalist-open-data.s3.amazonaws.com"
LICENSES = {"cc0", "cc-by"}
SPLIT_TARGETS = {"train": 20, "validation": 5, "test": 5}


def now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_dataset_active(root):
    if (root / "paused.json").exists():
        raise ValueError(
            "Dataset paused by user: revise and approve the species selection before "
            "resuming collection/training. Existing images and models are preserved."
        )


@contextmanager
def dataset_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another collector/trainer is using this dataset.") from None
        yield


class PublicClient:
    """One worker, <= 30 requests/minute, bounded response size and retries."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Yizhu-public-plant-training/1.0"
        self.last_request = 0.0

    def get(self, url, *, params=None, limit=12 * 1024 * 1024):
        for attempt in range(3):
            time.sleep(max(0, 2.0 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with self.session.get(
                    url, params=params, timeout=(15, 90), stream=True, allow_redirects=False,
                ) as response:
                    # Never bypass access restrictions or hammer a rate-limited endpoint.
                    if response.status_code in {403, 429}:
                        raise PermissionError(f"Public source paused: HTTP {response.status_code}")
                    if response.is_redirect:
                        raise ValueError("Unexpected redirect from public source.")
                    response.raise_for_status()
                    content = bytearray()
                    for chunk in response.iter_content(65536):
                        content.extend(chunk)
                        if len(content) > limit:
                            raise ValueError("Public response exceeds size limit.")
                    return bytes(content)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 2 or (status is not None and status < 500):
                    raise
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("Unreachable request state.")

    def json(self, endpoint, params):
        return json.loads(self.get(f"{API}/{endpoint}", params=params))


def catalog_names(path):
    """Ambiguous scientific-name matches are excluded, not guessed."""
    names = {}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        for name, raw in db.execute(
            "SELECT scientific_name,data FROM species WHERE is_visible=1 AND has_chinese_name=1"
        ):
            names.setdefault(name, []).append(json.loads(raw))
    return {name: rows[0] for name, rows in names.items() if len(rows) == 1}


def select_species(counts, names, target):
    selected, skipped, seen = [], [], set()
    for row in sorted(counts, key=lambda r: (-r["count"], r["taxon"]["id"])):
        taxon = row["taxon"]
        species = names.get(taxon["name"])
        if taxon.get("rank") != "species" or species is None:
            skipped.append({"scientificName": taxon["name"], "count": row["count"]})
            continue
        if species["id"] in seen:
            continue
        seen.add(species["id"])
        selected.append({
            "rank": len(selected) + 1, "speciesId": species["id"],
            "scientificName": taxon["name"], "chineseName": species["chineseName"],
            "chineseAliases": species.get("chineseAliases", []),
            "taxonomyChinese": species.get("taxonomyChinese", {}),
            "inatTaxonId": taxon["id"], "chinaResearchObservations": row["count"],
        })
        if len(selected) == target:
            break
    return selected, skipped


def shortlist(root, client, *, target=1000, catalog=CATALOG):
    destination = root / "species.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["policy"] != POLICY or len(result["species"]) != target:
            raise ValueError("Existing shortlist has a different policy/size; use a new directory.")
        return result
    names, counts = catalog_names(catalog), []
    selected, skipped = [], []
    for page in range(1, 21):
        cache = root / "counts" / f"{page}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            params = {
                "place_id": 6903, "taxon_id": 47126, "quality_grade": "research",
                "rank": "species", "verifiable": "true", "per_page": 500, "page": page,
            }
            payload = client.json("observations/species_counts", params)
            write_json(cache, payload)
        counts.extend(payload["results"])
        selected, skipped = select_species(counts, names, target)
        if len(selected) == target or not payload["results"]:
            break
    if len(selected) != target:
        raise ValueError(f"Only {len(selected)} exact visible matches; target is {target}.")
    result = {
        "createdAt": now(), "policy": POLICY, "catalogSha256": sha256(catalog),
        "source": f"{API}/observations/species_counts",
        "placeId": 6903, "placeName": "China (iNaturalist place boundary)",
        "countScope": "research-grade species observations; not abundance or national ranking",
        "trainingImageScope": "worldwide public CC0/CC BY photographs, not private user photos",
        "limitations": [
            "Citizen-science geographic and observer bias; cultivated plants underrepresented.",
            "Only exact scientific-name matches with existing visible Chinese names included.",
            "Counts are a live paginated snapshot, not an atomic census.",
        ],
        "species": selected, "skippedBeforeCutoff": skipped,
    }
    write_json(destination, result)
    with (root / "species.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["排名", "物种ID", "中文名", "别名", "学名", "科", "属", "中国观察数", "iNat ID"])
        for row in selected:
            writer.writerow([
                row["rank"], row["speciesId"], row["chineseName"],
                "、".join(row["chineseAliases"]), row["scientificName"],
                row["taxonomyChinese"].get("family", ""),
                row["taxonomyChinese"].get("genus", ""),
                row["chinaResearchObservations"], row["inatTaxonId"],
            ])
    return result


def observer_group(user_id):
    return hashlib.sha256(f"inat-observer:{user_id}".encode()).hexdigest()


def group_split(group):
    # Global observer grouping: same photographer never spans splits or species.
    bucket = int(group[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def public_samples(observations, species):
    """Project only needed public provenance; discard coordinates/comments/profiles."""
    source_name = species.get("sourceScientificName", species["scientificName"])
    evidence = species.get("sourceNameEvidence", {})
    source_rank = species.get("sourceTaxonRank", "species")
    if source_rank == "hybrid":
        from yizhu_server.named_hybrid_sources import validate_hybrid_evidence

        validate_hybrid_evidence(species)
    elif source_rank != "species":
        raise ValueError("Unsupported source rank; do not import infrahybrids or cultivars.")
    elif evidence.get("matchType") == "gbif-exact-synonym":
        from yizhu_server.source_taxonomy_bridge import validate_bridge

        validate_bridge(species)
    elif source_name != species["scientificName"] and (
        evidence.get("queryScientificName") != species["scientificName"]
        or evidence.get("sourceScientificName") != source_name
        or evidence.get("sourceTaxonId") != species["inatTaxonId"]
        or evidence.get("matchType") != "listed-scientific-synonym"
        or not any(
            row.get("name") == species["scientificName"]
            and row.get("lexicon") == "scientific-names"
            and type(row.get("is_valid")) is bool
            for row in evidence.get("linkedScientificNames", [])
        )
    ):
        raise ValueError("Different source scientific name requires explicit source evidence.")
    rows = []
    for observation in observations:
        taxon = observation.get("taxon") or {}
        user_id = (observation.get("user") or {}).get("id")
        if (
            observation.get("quality_grade") != "research"
            or taxon.get("id") != species["inatTaxonId"]
            or taxon.get("name") != source_name
            or taxon.get("rank") != source_rank
            or not isinstance(user_id, int)
        ):
            continue
        row = licensed_photo(
            observation, species, label_source="iNaturalist research grade (not expert-audited gold label)",
        )
        if row is not None:
            rows.append(row)
    return rows


def licensed_photo(observation, species, *, label_source):
    """Project one licensed photo; caller must independently validate the taxon/label policy."""
    user_id = (observation.get("user") or {}).get("id")
    if type(user_id) is not int or user_id < 1 or type(observation.get("id")) is not int:
        return None
    for photo in observation.get("photos", []):
        url = urlparse(photo.get("url", ""))
        if (
            photo.get("license_code") not in LICENSES or photo.get("hidden")
            or photo.get("flags") or not photo.get("attribution")
            or type(photo.get("id")) is not int or photo["id"] <= 0
            or url.scheme != "https" or url.netloc != PHOTO_HOST or url.query or url.fragment
            or Path(url.path).suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            continue
        group = observer_group(user_id)
        source_rank = species.get("sourceTaxonRank", "species")
        evidence = species.get("sourceNameEvidence", {})
        return {
            "sampleId": f"inat:{observation['id']}:{photo['id']}",
            "speciesId": species["speciesId"], "scientificName": species["scientificName"],
            **({"sourceTaxonRank": source_rank} if source_rank != "species" else {}),
            **({"sourceScientificName": species.get("sourceScientificName", species["scientificName"]),
                "sourceNameEvidence": evidence} if evidence else {}),
            "observationId": observation["id"], "photoId": photo["id"],
            "observerGroup": group, "split": group_split(group),
            "sourceUrl": f"https://www.inaturalist.org/observations/{observation['id']}",
            "imageUrl": f"https://{PHOTO_HOST}/photos/{photo['id']}/medium{Path(url.path).suffix}",
            "license": photo["license_code"], "attribution": photo["attribution"],
            "licenseUrl": ("https://creativecommons.org/publicdomain/zero/1.0/"
                           if photo["license_code"] == "cc0"
                           else "https://creativecommons.org/licenses/by/4.0/"),
            "labelSource": label_source, "retrievedAt": now(),
        }
    return None


def difference_hash(image):
    from PIL import Image

    values = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (values[y * 9 + x] > values[y * 9 + x + 1])
    return result


class DuplicateIndex:
    """Exact and approximate deduplication across ALL splits/classes."""

    def __init__(self):
        self.exact = set()
        self.buckets = {}

    @staticmethod
    def keys(value):
        # Five disjoint blocks: <=4 changed bits share at least one block.
        return [(i, (value >> (i * 13)) & (4095 if i == 4 else 8191)) for i in range(5)]

    def contains(self, digest, perceptual):
        if digest in self.exact:
            return True
        candidates = set()
        for key in self.keys(perceptual):
            candidates.update(self.buckets.get(key, ()))
        return any((value ^ perceptual).bit_count() <= 4 for value in candidates)

    def add(self, digest, perceptual):
        self.exact.add(digest)
        for key in self.keys(perceptual):
            self.buckets.setdefault(key, set()).add(perceptual)


def load_samples(root):
    rows = []
    for path in sorted((root / "samples").glob("*.json")):
        rows.extend(json.loads(path.read_text())["samples"])
    return rows


def collect(root, client, species_document, *, class_limit=1000,
            split_targets=None, max_pages=5, resolve_species=None):
    require_dataset_active(root)
    from yizhu_server.recognition import read_photo

    targets = dict(SPLIT_TARGETS if split_targets is None else split_targets)
    if (
        set(targets) != set(SPLIT_TARGETS)
        or any(type(value) is not int or value < 1 for value in targets.values())
        or type(max_pages) is not int or not 1 <= max_pages <= 100
    ):
        raise ValueError("Invalid collection split targets or page limit.")
    root.mkdir(parents=True, exist_ok=True)
    index = DuplicateIndex()
    existing = load_samples(root)
    for row in existing:
        index.add(row["imageSha256"], int(row["differenceHash"], 16))
    collected_ids = {row["sampleId"] for row in existing}
    config_path = root / "collection.json"
    if not config_path.exists():
        write_json(config_path, {
            "createdBefore": now(), "splitTargets": targets,
            "speciesSha256": sha256(root / "species.json"), "maxPagesPerSpecies": max_pages,
            "imageSize": "medium-500px", "policy": "public-cc0-ccby-observer-split-dhash4-v1",
        })
    config = json.loads(config_path.read_text())
    if config["speciesSha256"] != sha256(root / "species.json"):
        raise ValueError("Shortlist changed after image collection started.")
    if config["splitTargets"] != targets or config["maxPagesPerSpecies"] != max_pages:
        raise ValueError("Collection policy changed; use a new dataset directory.")
    for position, species in enumerate(species_document["species"][:class_limit], 1):
        require_dataset_active(root)
        write_json(root / "progress.json", {
            "updatedAt": now(), "stage": "resolving" if resolve_species else "collecting",
            "currentSpeciesPosition": position,
            "currentScientificName": species["scientificName"],
            "targetSpecies": len(species_document["species"]),
            "downloadedImages": len(collected_ids),
        })
        if resolve_species:
            species = resolve_species(species)
            if species is None:
                continue
        taxon_id = species["inatTaxonId"]
        path = root / "samples" / f"{taxon_id}.json"
        state = json.loads(path.read_text()) if path.exists() else {"samples": [], "complete": False}
        if state["complete"]:
            continue
        counts = Counter(row["split"] for row in state["samples"])
        observer_counts = Counter(row["observerGroup"] for row in state["samples"])
        for page in range(1, config["maxPagesPerSpecies"] + 1):
            require_dataset_active(root)
            cache = root / "photo-metadata" / f"{taxon_id}-{page}.json"
            if cache.exists():
                metadata = json.loads(cache.read_text())
            else:
                payload = client.json("observations", {
                    "taxon_id": taxon_id, "quality_grade": "research",
                    "photos": "true", "photo_license": "cc0,cc-by", "per_page": 100,
                    "page": page, "order_by": "id", "order": "desc",
                    "created_d2": config["createdBefore"],
                })
                metadata = {
                    "samples": public_samples(payload["results"], species),
                    "hasMore": len(payload["results"]) == 100,
                }
                write_json(cache, metadata)
            for row in metadata["samples"]:
                if (
                    counts[row["split"]] >= targets[row["split"]]
                    or row["sampleId"] in collected_ids
                    or observer_counts[row["observerGroup"]] >= 3
                ):
                    continue
                if shutil.disk_usage(root).free < 4 * 1024**3:
                    raise OSError("Paused: keep at least 4 GiB disk free.")
                relative = f"images/{row['photoId']}.jpg"
                image_path = root / relative
                image_path.parent.mkdir(exist_ok=True)
                temporary = image_path.with_suffix(".download")
                try:
                    content = client.get(row["imageUrl"], limit=4 * 1024 * 1024)
                    temporary.write_bytes(content)
                    with read_photo(temporary) as image:
                        if min(image.size) < 100:
                            continue
                        perceptual = difference_hash(image)
                        # Pixel-only encoding strips EXIF and other metadata.
                        image.save(temporary, format="JPEG", quality=92)
                    digest = sha256(temporary)
                    if index.contains(digest, perceptual):
                        continue
                    temporary.replace(image_path)
                except requests.HTTPError as exc:
                    if exc.response.status_code not in {404, 410}:
                        raise
                    continue
                except ValueError:
                    continue
                finally:
                    temporary.unlink(missing_ok=True)
                sample = {
                    **row, "path": relative, "imageSha256": digest,
                    "differenceHash": f"{perceptual:016x}",
                    "modifications": "500px source; RGB JPEG re-encode; metadata stripped",
                }
                state["samples"].append(sample)
                index.add(digest, perceptual)
                collected_ids.add(row["sampleId"])
                counts[row["split"]] += 1
                observer_counts[row["observerGroup"]] += 1
                write_json(path, state)
                if len(state["samples"]) % 10 == 0:
                    write_json(root / "progress.json", {
                        "updatedAt": now(), "stage": "collecting",
                        "currentSpeciesPosition": position,
                        "currentScientificName": species["scientificName"],
                        "targetSpecies": len(species_document["species"]),
                        "downloadedImages": len(collected_ids),
                        "currentSpeciesCounts": dict(counts),
                    })
            if all(counts[key] >= value for key, value in targets.items()):
                break
            if not metadata["hasMore"]:
                break
        state.update({"complete": True, "counts": dict(counts)})
        write_json(path, state)
        write_json(root / "progress.json", {
            "updatedAt": now(), "stage": "collecting", "lastSpeciesRank": position,
            "targetSpecies": len(species_document["species"]),
            "downloadedImages": len(collected_ids), "lastSpeciesCounts": dict(counts),
        })
        print(f"species {position}/{class_limit}: {dict(counts)}", flush=True)
    all_rows = load_samples(root)
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    coverage = {}
    for row in all_rows:
        coverage.setdefault(row["speciesId"], Counter())[row["split"]] += 1
    ready = [
        species["speciesId"] for species in species_document["species"]
        if all(coverage.get(species["speciesId"], {}).get(key, 0) >= value
               for key, value in targets.items())
    ]
    report = {
        "updatedAt": now(), "stage": "collection-pass-finished",
        "targetSpecies": len(species_document["species"]), "downloadedImages": len(all_rows),
        "readySpecies": ready, "readySpeciesCount": len(ready),
        "splitTargets": targets, "counts": coverage,
        "shortfalls": [
            {
                "speciesId": species["speciesId"],
                "scientificName": species["scientificName"],
                "missing": {
                    key: max(0, value - coverage.get(species["speciesId"], {}).get(key, 0))
                    for key, value in targets.items()
                },
            }
            for species in species_document["species"] if species["speciesId"] not in ready
        ],
        "note": "Missing classes are NOT padded with pseudo-labels or text embeddings.",
    }
    write_json(root / "coverage.json", report)
    write_json(root / "progress.json", {
        k: v for k, v in report.items() if k not in {"counts", "shortfalls"}
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "collect"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--class-limit", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.class_limit <= 1000:
        parser.error("--class-limit must be 1–1000.")
    with dataset_lock(args.data):
        client = PublicClient()
        document = shortlist(args.data, client)
        print(f"Shortlist: {len(document['species'])} species; {args.data / 'species.csv'}", flush=True)
        if args.action == "collect":
            collect(args.data, client, document, class_limit=args.class_limit)


if __name__ == "__main__":
    main()
