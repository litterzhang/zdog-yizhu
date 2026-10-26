"""Supplemental cultivated-photo candidates with explicit multi-account identification evidence."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from yizhu_server.snapshot_files import reserve_space
from yizhu_server.training_data import (
    DuplicateIndex,
    PublicClient,
    dataset_lock,
    difference_hash,
    licensed_photo,
    load_samples,
    now,
    observer_group,
    require_dataset_active,
    sha256,
    write_json,
)

WITHIN_SPECIES = {"species", "subspecies", "variety", "form"}


def belongs(taxon, species_id):
    return (
        isinstance(taxon, dict) and taxon.get("rank") in WITHIN_SPECIES
        and (taxon.get("id") == species_id or species_id in taxon.get("ancestor_ids", []))
    )


def consensus_candidate(observation, species):
    source_id = species["inatTaxonId"]
    taxon = observation.get("taxon") or {}
    community = observation.get("community_taxon_id")
    if (
        observation.get("quality_grade") != "casual" or observation.get("captive") is not True
        or observation.get("identifications_most_agree") is not True
        or observation.get("identifications_most_disagree") is True
        or not observation.get("observed_on") or not observation.get("location")
        or not belongs(taxon, source_id)
        or community not in {source_id, taxon.get("id")}
        or any(q.get("agree") is False and q.get("metric") != "wild"
               for q in observation.get("quality_metrics", []))
    ):
        return None, "missing-cultivated-species-consensus-or-quality"
    votes = {}
    for identification in observation.get("identifications", []):
        if identification.get("current") is not True:
            continue
        user_id = (identification.get("user") or {}).get("id")
        named = identification.get("taxon") or {}
        if identification.get("disagreement") is True:
            return None, "current-identification-disagrees"
        if named.get("rank") in WITHIN_SPECIES | {"hybrid", "infrahybrid"} and not belongs(named, source_id):
            return None, "conflicting-current-taxon"
        if (
            belongs(named, source_id) and type(user_id) is int and user_id > 0
            and identification.get("category") in {"leading", "improving", "supporting"}
        ):
            votes[observer_group(user_id)] = {
                "taxonId": named["id"], "rank": named["rank"], "name": named.get("name"),
                "ancestorIds": named.get("ancestor_ids", []),
                "identificationId": identification.get("id"),
            }
    if len(votes) < 2:
        return None, "fewer-than-two-agreeing-accounts"
    row = licensed_photo(
        observation, species,
        label_source="iNaturalist cultivated casual observation; multi-account consensus candidate",
    )
    if row is None:
        return None, "no-allowlisted-licensed-photo"
    row.update({
        "proposedSplit": row.pop("split"), "trainingEligible": False,
        "sourceQualityGrade": "casual", "sourceCaptive": True,
        "labelReviewStatus": "cultivated-consensus-candidate-not-research-grade",
        "consensusEvidence": {
            "communityTaxonId": community, "targetSourceSpeciesId": source_id,
            "observationTaxonId": taxon["id"], "observationTaxonRank": taxon["rank"],
            "observationAncestorIds": taxon.get("ancestor_ids", []),
            "agreeingAccounts": votes,
            "qualityMetrics": [{"metric": q.get("metric"), "agree": q.get("agree")}
                               for q in observation.get("quality_metrics", [])],
        },
    })
    return row, None


def collect(primary, output, *, names):
    from PIL import Image

    require_dataset_active(primary)
    primary_doc = json.loads((primary / "species.json").read_text())
    indexed = {r["scientificName"]: r for r in primary_doc["species"]}
    if not names or len(names) != len(set(names)) or not set(names) <= indexed.keys():
        raise ValueError("Require distinct approved target scientific names.")
    members = []
    for name in names:
        sp = indexed[name]
        path = primary / "taxon-matches" / (hashlib.sha256(sp["speciesId"].encode()).hexdigest() + ".json")
        match = json.loads(path.read_text())
        if match["status"] != "matched" or match.get("sourceScientificName", name) != name:
            raise ValueError("This initial cultivated pass requires an exact source species match.")
        members.append({**sp, "inatTaxonId": match["inatTaxonId"]})
    identity = {"primarySpeciesSha256": sha256(primary / "species.json"), "species": members,
                "policy": "cultivated-multi-account-candidates-v1", "maxPages": 10,
                "maxPhotosPerSpecies": 100, "maxPhotosPerObserverSpecies": 3}
    with dataset_lock(output):
        require_dataset_active(output)
        plan = output / "plan.json"
        if plan.exists():
            existing = json.loads(plan.read_text())
            if {k: existing.get(k) for k in identity} != identity:
                raise ValueError("Collection scope changed.")
            identity = existing
        else:
            identity["createdBefore"] = now()
            write_json(plan, identity)
        saved = output / "candidates.json"
        rows = json.loads(saved.read_text()) if saved.exists() else []
        primary_rows = load_samples(primary)
        duplicates = DuplicateIndex()
        for r in [*primary_rows, *rows]:
            duplicates.add(r["imageSha256"], int(r["differenceHash"], 16))
        seen = {r["sampleId"] for r in [*primary_rows, *rows]}
        counts = Counter(r["speciesId"] for r in rows)
        authors = Counter((r["speciesId"], r["observerGroup"]) for r in [*primary_rows, *rows])
        rejected = Counter()
        client = PublicClient()
        for position, sp in enumerate(members, 1):
            for page in range(1, 11):
                require_dataset_active(output)
                metadata = output / "metadata" / f"{sp['inatTaxonId']}-{page}.json"
                if metadata.exists():
                    batch = json.loads(metadata.read_text())
                else:
                    payload = client.json("observations", {
                        "taxon_id": sp["inatTaxonId"], "quality_grade": "casual",
                        "photos": "true", "photo_license": "cc0,cc-by",
                        "per_page": 100, "page": page, "order_by": "id", "order": "desc",
                        "created_d2": identity["createdBefore"],
                    })
                    eligible = []
                    for observation in payload["results"]:
                        row, reason = consensus_candidate(observation, sp)
                        if row:
                            eligible.append(row)
                        else:
                            rejected[reason] += 1
                    batch = {"fetchedAt": now(), "rows": eligible,
                             "hasMore": len(payload["results"]) == 100,
                             "totalObservations": payload["total_results"]}
                    write_json(metadata, batch)
                for row in batch["rows"]:
                    author = (row["speciesId"], row["observerGroup"])
                    if row["sampleId"] in seen or counts[sp["speciesId"]] >= 100:
                        continue
                    if authors[author] >= 3:
                        rejected["observer-cap-across-primary-and-supplement"] += 1
                        continue
                    reserve_space(output, 8 * 1024**2)
                    content = client.get(row["imageUrl"], limit=4 * 1024**2)
                    with Image.open(io.BytesIO(content)) as image:
                        if min(image.size) < 100 or image.width * image.height > 16_000_000:
                            rejected["image-dimensions"] += 1
                            continue
                        image = image.convert("RGB")
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
                    rows.append({**row, "path": relative, "imageSha256": digest,
                                 "differenceHash": f"{perceptual:016x}",
                                 "modifications": "500px source; RGB JPEG; metadata removed"})
                    duplicates.add(digest, perceptual)
                    seen.add(row["sampleId"])
                    counts[sp["speciesId"]] += 1
                    authors[author] += 1
                    write_json(saved, rows)
                write_json(output / "progress.json", {
                    "updatedAt": now(), "stage": "collecting", "position": position,
                    "speciesInPass": len(members), "scientificName": sp["scientificName"],
                    "candidateImages": len(rows), "counts": dict(counts),
                    "rejected": dict(rejected), "trainingEligible": False,
                })
                if not batch["hasMore"] or counts[sp["speciesId"]] >= 100:
                    break
        report = {"completedAt": now(), "stage": "collection-pass-finished",
                  "candidateImages": len(rows), "counts": dict(counts),
                  "rejected": dict(rejected), "trainingEligible": False,
                  "currentTrainingChanged": False, "sourceQualityGradeChanged": False}
        write_json(output / "progress.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()
    try:
        collect(args.primary, args.output, names=args.names)
    except Exception as exc:
        write_json(args.output / "stopped.json", {
            "stoppedAt": now(), "errorType": type(exc).__name__, "automaticRetry": False,
        })
        raise


if __name__ == "__main__":
    main()
