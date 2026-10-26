"""Collect licensed public photos for the fixed regional-evidence candidate set."""

import argparse
import hashlib
import json
from pathlib import Path

from yizhu_server.training_data import (
    ROOT,
    PublicClient,
    collect,
    dataset_lock,
    now,
    require_dataset_active,
    sha256,
    write_json,
)

DEFAULT_POOL = ROOT / ".cache/research/everyday-candidate-pool-20260922-v3/candidates.json"
DEFAULT_DATA = ROOT / ".cache/training/china-everyday-972"
POLICY = "fixed-priority-species-public-photos-v1"
TARGETS = {"train": 70, "validation": 15, "test": 15}


def select_targets(pool):
    """Preserve the requested candidates, including missing Chinese display names."""
    selected = []
    ids, names = set(), set()
    for item in pool:
        match = item.get("catalogMatch") or {}
        if not (
            item.get("priorityReview") and item.get("rankStatus") == "种"
            and match.get("speciesId")
        ):
            continue
        name = match["scientificName"]
        if item["scientificLabel"] != name:
            raise ValueError("Candidate scientific name differs from its catalog match.")
        if match["speciesId"] in ids or name in names:
            raise ValueError("Duplicate target species ID or scientific name.")
        ids.add(match["speciesId"])
        names.add(name)
        selected.append({
            "speciesId": match["speciesId"], "scientificName": name,
            "chineseName": match.get("chineseName"),
            "candidateId": item["candidateId"],
            "sourceNames": item.get("sourceNames", []),
            "regionalEvidence": {
                key: item.get(key, [])
                for key in ("sourceRegions", "sourceRecordIds", "sourceWorks", "sourceTypes")
            },
            "frequencyEvidence": item.get("frequencies", {}),
        })
    return selected


def prepare(root, pool_path):
    require_dataset_active(root)
    digest = sha256(pool_path)
    destination = root / "species.json"
    if destination.exists():
        document = json.loads(destination.read_text())
        if document.get("policy") != POLICY or document.get("candidatePoolSha256") != digest:
            raise ValueError("Existing dataset has a different selection; do not overwrite.")
        return document
    species = select_targets(json.loads(pool_path.read_text()))
    if len(species) != 972:
        raise ValueError(f"Expected the agreed 972 targets, found {len(species)}.")
    document = {
        "createdAt": now(), "policy": POLICY, "candidatePoolSha256": digest,
        "requestedClassCount": len(species), "species": species,
        "splitTargets": TARGETS,
        "trainingImageScope": "Worldwide public CC0/CC BY photos of the fixed target species",
        "limitations": [
            "Regional-evidence priority candidates, not a national commonness ranking.",
            "Original candidate pool is not modified or globally marked reviewed.",
            "Exact active species matches only; unresolved synonyms remain gaps.",
            "Research-grade labels are not expert-audited; cultivated plants may be scarce.",
            "Photo collection does not mean a trained or deployed 972-class model.",
        ],
    }
    write_json(destination, document)
    return document


def exact_taxon(payload, name):
    matches = [
        row for row in payload["results"]
        if row.get("name") == name and row.get("rank") == "species"
        and row.get("is_active") is True and 47126 in row.get("ancestor_ids", [])
        and type(row.get("id")) is int and row["id"] > 0
    ]
    # Do not accept a unique-looking match from a truncated/ambiguous search.
    if payload.get("total_results", len(payload["results"])) > len(payload["results"]):
        return None, "search-truncated"
    if len(matches) != 1:
        return None, "ambiguous" if matches else "no-exact-active-species-match"
    return matches[0]["id"], "matched"


def source_name_taxon(payload, name):
    """Require a unique active plant species with an exact scientific-name link."""
    if payload.get("total_results", len(payload["results"])) > len(payload["results"]):
        return None, "search-truncated"
    matches = []
    for row in payload["results"]:
        if (
            row.get("rank") != "species" or row.get("is_active") is not True
            or 47126 not in row.get("ancestor_ids", [])
            or type(row.get("id")) is not int or row["id"] <= 0
            or not isinstance(row.get("name"), str) or not row["name"]
        ):
            continue
        linked = [
            item for item in row.get("names", [])
            if item.get("name") == name and item.get("lexicon") == "scientific-names"
            and type(item.get("is_valid")) is bool
        ]
        if row["name"] != name and not linked:
            continue
        matches.append({
            "inatTaxonId": row["id"], "sourceScientificName": row["name"],
            "sourceNameEvidence": {
                "queryScientificName": name, "sourceScientificName": row["name"],
                "sourceTaxonId": row["id"],
                "matchType": "exact" if row["name"] == name else "listed-scientific-synonym",
                "linkedScientificNames": linked,
                "listedScientificNames": sorted({
                    row["name"],
                    *(item["name"] for item in row.get("names", [])
                      if item.get("lexicon") == "scientific-names"
                      and isinstance(item.get("name"), str)),
                }),
            },
        })
    if len(matches) != 1:
        return None, "ambiguous" if matches else "no-exact-scientific-name-evidence"
    return matches[0], "matched"


class TaxonResolver:
    def __init__(self, root, client, *, allow_source_names=False, target_names=None):
        self.root, self.client = root, client
        self.allow_source_names = allow_source_names
        self.target_names = target_names or {}
        self.records = {}
        for path in sorted((root / "taxon-matches").glob("*.json")):
            record = json.loads(path.read_text())
            self.records[record["speciesId"]] = record

    def __call__(self, species):
        require_dataset_active(self.root)
        identifier = species["speciesId"]
        name = species["scientificName"]
        record = self.records.get(identifier)
        if record is None:
            payload = self.client.json("taxa", {
                "q": name, "rank": "species", "is_active": "true",
                "taxon_id": 47126, "per_page": 100,
                **({"all_names": "true"} if self.allow_source_names else {}),
            })
            details = {}
            if self.allow_source_names:
                details, status = source_name_taxon(payload, name)
                details = details or {}
                taxon_id = details.get("inatTaxonId")
            else:
                taxon_id, status = exact_taxon(payload, name)
            if (
                taxon_id is not None
                and any(
                    self.target_names.get(source_name, identifier) != identifier
                    for source_name in details.get("sourceNameEvidence", {}).get(
                        "listedScientificNames", [details.get("sourceScientificName")],
                    )
                )
            ):
                taxon_id, status, details = None, "source-name-is-another-target", {}
            if taxon_id is not None and any(
                row.get("inatTaxonId") == taxon_id for row in self.records.values()
            ):
                taxon_id, status, details = None, "taxon-already-mapped", {}
            record = {
                **details,
                "speciesId": identifier, "scientificName": name,
                "inatTaxonId": taxon_id, "status": status, "retrievedAt": now(),
                "source": "https://api.inaturalist.org/v1/taxa",
            }
            filename = hashlib.sha256(identifier.encode()).hexdigest() + ".json"
            write_json(self.root / "taxon-matches" / filename, record)
            self.records[identifier] = record
        if record["scientificName"] != name:
            raise ValueError("Cached taxon mapping does not match the target scientific name.")
        write_json(self.root / "resolution.json", {
            "updatedAt": now(), "checkedSpecies": len(self.records),
            "matchedSpecies": sum(r["status"] == "matched" for r in self.records.values()),
            "unresolved": [r for r in self.records.values() if r["status"] != "matched"],
        })
        print(f"taxon {name}: {record['status']}", flush=True)
        return {
            **species, "inatTaxonId": record["inatTaxonId"],
            **{key: record[key] for key in ("sourceScientificName", "sourceNameEvidence")
               if key in record},
        } if (
            record["status"] == "matched"
        ) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--class-limit", type=int, default=972)
    args = parser.parse_args()
    if not 1 <= args.class_limit <= 972:
        parser.error("--class-limit must be 1–972.")
    with dataset_lock(args.data):
        document = prepare(args.data, args.pool)
        print(f"Fixed targets: {len(document['species'])}; {args.data}", flush=True)
        if args.action == "collect":
            client = PublicClient()
            try:
                collect(
                    args.data, client, document, class_limit=args.class_limit,
                    split_targets=TARGETS, max_pages=20,
                    resolve_species=TaxonResolver(args.data, client),
                )
            except Exception as exc:
                progress_path = args.data / "progress.json"
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                write_json(progress_path, {
                    **progress, "updatedAt": now(), "stage": "stopped",
                    "errorType": type(exc).__name__,
                    "note": "Inspect local log; rerun to resume. No automatic retry loop.",
                })
                raise


if __name__ == "__main__":
    main()
