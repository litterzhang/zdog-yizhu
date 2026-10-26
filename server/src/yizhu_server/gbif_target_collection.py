"""Collect review-only CC0/CC-BY GBIF image candidates for exact target taxa."""

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from yizhu_server.training_data import difference_hash, now, sha256, write_json

API = "https://api.gbif.org/v1"
LICENSES = ("CC0_1_0", "CC_BY_4_0")
BY_SA_LICENSE = "CC_BY_SA_4_0"
ALLOWED_MEDIA_LICENSES = (
  "creativecommons.org/publicdomain/zero/1.0",
  "creativecommons.org/licenses/by/4.0",
)


def request_json(path, parameters=None, attempts=4):
  url = f"{API}/{path}"
  if parameters:
    url += "?" + urllib.parse.urlencode(parameters)
  for attempt in range(attempts):
    try:
      request = urllib.request.Request(url, headers={"User-Agent": "yizhu-training/1.0"})
      with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)
    except Exception:
      if attempt + 1 == attempts:
        raise
      time.sleep(min(20, 2 ** attempt))


def download(url, attempts=1):
  limit = 16 * 1024 * 1024
  for attempt in range(attempts):
    try:
      request = urllib.request.Request(url, headers={"User-Agent": "yizhu-training/1.0"})
      with urllib.request.urlopen(request, timeout=10) as response:
        if response.headers.get_content_type() not in {"image/jpeg", "image/png"}:
          raise ValueError("Unexpected image content type.")
        length = response.headers.get("Content-Length")
        if length and int(length) > limit:
          raise ValueError("Image size outside collection bounds.")
        deadline = time.monotonic() + 20
        chunks, size = [], 0
        while True:
          if time.monotonic() > deadline:
            raise TimeoutError("Image transfer exceeded total time limit.")
          chunk = response.read(min(64 * 1024, limit + 1 - size))
          if not chunk:
            break
          chunks.append(chunk)
          size += len(chunk)
          if size > limit:
            raise ValueError("Image size outside collection bounds.")
        data = b"".join(chunks)
        if not 1024 <= len(data) <= limit:
          raise ValueError("Image size outside collection bounds.")
        return data
    except Exception:
      if attempt + 1 == attempts:
        raise
      time.sleep(min(20, 2 ** attempt))


def media_row(record, *, allow_by_sa=False):
  for media in record.get("media", []):
    license_url = (media.get("license") or "").lower().rstrip("/")
    identifier = media.get("identifier")
    attribution = (
      media.get("creator") or media.get("rightsHolder")
      or record.get("rightsHolder") or record.get("recordedBy")
      or media.get("publisher") or record.get("datasetName")
    )
    if (
      media.get("type") == "StillImage" and isinstance(identifier, str)
      and identifier.startswith("https://")
      and (
        any(token in license_url for token in ALLOWED_MEDIA_LICENSES)
        or (allow_by_sa and "creativecommons.org/licenses/by-sa/4.0" in license_url)
      )
      and attribution
    ):
      return media, str(attribution).strip()
  return None


def collect(targets_path, output, *, names, per_species=100, author_cap=5,
            exclusions_path=None, max_pages=20, skip_unresolved=False,
            allow_by_sa=False):
  from PIL import Image, ImageOps

  if (
    output.exists() or not names or per_species > 150 or author_cap > 10
    or not 1 <= max_pages <= 100
  ):
    raise ValueError("Use a new output and bounded GBIF collection settings.")
  targets = json.loads(targets_path.read_text())
  exclusions = (
    json.loads(exclusions_path.read_text()) if exclusions_path is not None else {}
  )
  excluded_occurrences = set(exclusions.get("gbifOccurrenceKeys", []))
  excluded_urls = set(exclusions.get("sourceImageUrls", []))
  excluded_inat_photos = {str(value) for value in exclusions.get("inatPhotoIds", [])}
  excluded_inat_observations = {
    str(value) for value in exclusions.get("inatObservationIds", [])
  }
  indexed = {row["scientificName"]: row for row in targets["classes"]}
  if len(names) != len(set(names)) or not set(names) <= set(indexed):
    raise ValueError("Require distinct reviewed target scientific names.")
  output.mkdir(parents=True)
  (output / "images").mkdir()
  (output / "records").mkdir()
  licenses = (*LICENSES, None) if allow_by_sa else LICENSES
  write_json(output / "plan.json", {
    "createdAt": now(), "purpose": "target162-missing-GBIF-candidates",
    "targetSha256": sha256(targets_path), "scientificNames": names,
    "licenses": [value or "UNFILTERED_MEDIA_LEVEL_BY_SA_CHECK" for value in licenses],
    "perSpeciesLimit": per_species,
    "authorCap": author_cap, "maxPagesPerLicense": max_pages,
    "trainingEligible": False,
    "preciseCoordinatesStored": False, "automaticAdmission": False,
    "exclusionsSha256": sha256(exclusions_path) if exclusions_path else None,
    "excludedOccurrenceCount": len(excluded_occurrences),
    "excludedSourceImageUrlCount": len(excluded_urls),
    "excludedInaturalistPhotoCount": len(excluded_inat_photos),
    "excludedInaturalistObservationCount": len(excluded_inat_observations),
  })
  candidates, rejected = [], Counter()
  image_hashes, difference_hashes = set(), set()
  for position, name in enumerate(names, 1):
    target = indexed[name]
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
      if skip_unresolved:
        rejected["species-resolution"] += 1
        print(json.dumps({
          "scientificName": name, "accepted": 0, "resolutionSkipped": True,
        }), flush=True)
        write_json(output / "progress.json", {
          "updatedAt": now(), "stage": "collecting",
          "completedSpecies": position, "totalSpecies": len(names),
          "currentScientificName": name, "candidateImages": len(candidates),
          "counts": dict(Counter(row["speciesId"] for row in candidates)),
          "rejected": dict(rejected), "trainingEligible": False,
          "automaticAdmission": False,
        })
        continue
      raise ValueError(f"GBIF exact species resolution failed for {name}.")
    author_counts, accepted = Counter(), 0
    seen_occurrences = set()
    for license_name in licenses:
      offset = 0
      pages = 0
      while accepted < per_species and pages < max_pages:
        parameters = {
          "taxon_key": key, "media_type": "StillImage",
          "limit": 100, "offset": offset,
        }
        if license_name is not None:
          parameters["license"] = license_name
        page = request_json("occurrence/search", parameters)
        rows = page.get("results", [])
        pages += 1
        if not rows:
          break
        for record in rows:
          occurrence_key = record.get("key")
          if occurrence_key in excluded_occurrences:
            rejected["excluded-existing-occurrence"] += 1
            continue
          if occurrence_key in seen_occurrences:
            continue
          seen_occurrences.add(occurrence_key)
          if (
            record.get("speciesKey") != key or record.get("acceptedTaxonKey") != key
            or record.get("taxonRank") != "SPECIES"
            or record.get("occurrenceStatus") != "PRESENT"
          ):
            rejected["taxon-contract"] += 1
            continue
          media_result = media_row(record, allow_by_sa=allow_by_sa)
          if media_result is None:
            rejected["media-license-or-attribution"] += 1
            continue
          media, author = media_result
          if media["identifier"] in excluded_urls:
            rejected["excluded-existing-source-image"] += 1
            continue
          photo_match = re.search(r"/photos/(\d+)/(?:original|medium|large|small|square)", media["identifier"])
          references = str(record.get("references") or record.get("occurrenceID") or "")
          observation_match = re.search(r"/observations/(\d+)", references)
          if photo_match and photo_match.group(1) in excluded_inat_photos:
            rejected["excluded-existing-inaturalist-photo"] += 1
            continue
          if (
            observation_match
            and observation_match.group(1) in excluded_inat_observations
          ):
            rejected["excluded-existing-inaturalist-observation"] += 1
            continue
          author_group = f"{record.get('datasetKey')}:{author.casefold()}"
          if author_counts[author_group] >= author_cap:
            rejected["author-cap"] += 1
            continue
          try:
            raw = download(media["identifier"])
            with Image.open(io.BytesIO(raw)) as source:
              if min(source.size) < 100 or source.width * source.height > 40_000_000:
                raise ValueError("Image dimensions outside collection bounds.")
              image = ImageOps.exif_transpose(source).convert("RGB")
              image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
              buffer = io.BytesIO()
              image.save(buffer, "JPEG", quality=90, optimize=True)
              encoded = buffer.getvalue()
              dhash = difference_hash(image)
          except Exception:
            rejected["download-or-decode"] += 1
            continue
          digest = __import__("hashlib").sha256(encoded).hexdigest()
          if digest in image_hashes or dhash in difference_hashes:
            rejected["duplicate-image"] += 1
            continue
          image_hashes.add(digest)
          difference_hashes.add(dhash)
          author_counts[author_group] += 1
          path = output / "images" / f"{digest}.jpg"
          path.write_bytes(encoded)
          row = {
            "candidateId": f"gbif:{occurrence_key}:{digest[:16]}",
            "speciesId": target["speciesId"],
            "scientificName": target["scientificName"],
            "chineseName": target["chineseName"],
            "gbifQueryTaxonKey": query_key, "gbifAcceptedTaxonKey": key,
            "gbifMatchStatus": match.get("status"),
            "gbifMatchedScientificName": match.get("scientificName"),
            "gbifAcceptedSpeciesName": match.get("species"),
            "gbifOccurrenceKey": occurrence_key,
            "datasetKey": record.get("datasetKey"),
            "basisOfRecord": record.get("basisOfRecord"),
            "eventDate": record.get("eventDate"),
            "recordedBy": record.get("recordedBy"),
            "identifiedBy": record.get("identifiedBy"),
            "sourceReferences": record.get("references"),
            "mediaReferences": media.get("references"),
            "mediaCreator": media.get("creator"),
            "mediaRightsHolder": media.get("rightsHolder"),
            "mediaLicense": media.get("license"),
            "sourceImageUrl": media.get("identifier"),
            "imageSha256": digest, "differenceHash": dhash,
            "path": f"images/{path.name}",
            "labelEvidence": (
              "GBIF exact 100-confidence species match (including an explicit "
              "acceptedUsageKey synonym link where applicable) and occurrence "
              "speciesKey/acceptedTaxonKey equality; review-only, not expert re-audited."
            ),
            "trainingEligible": False,
          }
          candidates.append(row)
          write_json(output / "records" / f"{row['candidateId'].replace(':', '-')}.json", row)
          accepted += 1
          if accepted >= per_species:
            break
        if page.get("endOfRecords") or accepted >= per_species:
          break
        offset += len(rows)
    print(json.dumps({"scientificName": name, "accepted": accepted}), flush=True)
    write_json(output / "progress.json", {
      "updatedAt": now(), "stage": "collecting",
      "completedSpecies": position, "totalSpecies": len(names),
      "currentScientificName": name, "candidateImages": len(candidates),
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
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--names", nargs="+")
  parser.add_argument("--names-file", type=Path)
  parser.add_argument("--per-species", type=int, default=100)
  parser.add_argument("--author-cap", type=int, default=5)
  parser.add_argument("--max-pages", type=int, default=20)
  parser.add_argument("--exclusions", type=Path)
  parser.add_argument("--skip-unresolved", action="store_true")
  parser.add_argument("--allow-by-sa", action="store_true")
  args = parser.parse_args()
  if bool(args.names) == bool(args.names_file):
    parser.error("Provide exactly one of --names or --names-file")
  names = args.names
  if args.names_file:
    document = json.loads(args.names_file.read_text())
    names = document["scientificNames"]
  collect(
    args.targets.resolve(), args.output.resolve(), names=names,
    per_species=args.per_species, author_cap=args.author_cap,
    exclusions_path=args.exclusions.resolve() if args.exclusions else None,
    max_pages=args.max_pages, skip_unresolved=args.skip_unresolved,
    allow_by_sa=args.allow_by_sa,
  )


if __name__ == "__main__":
  main()
