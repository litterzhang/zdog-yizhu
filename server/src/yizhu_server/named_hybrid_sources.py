"""Explicit source-name bridges for already approved named hybrid targets."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def binomial_notation(name):
    """Normalize only a standalone hybrid marker; never drop subspecies or parent names."""
    if not isinstance(name, str):
        return None
    value = re.sub(r"^×\s*", "", name.strip())
    value = re.sub(r"(?<=\s)[×x](?=\s)", " ", value)
    parts = value.split()
    if len(parts) != 2 or not re.fullmatch(r"[A-Z][a-z]+", parts[0]) or not re.fullmatch(
        r"[a-z][a-z-]*", parts[1]
    ):
        return None
    return " ".join(parts)


def resolve_named_hybrid(payload, species):
    wanted = binomial_notation(species["scientificName"])
    if wanted is None or payload.get("total_results", 0) > len(payload.get("results", [])):
        raise ValueError("Invalid target or truncated source results.")
    candidates = []
    for row in payload.get("results", []):
        if (
            row.get("rank") != "hybrid" or row.get("is_active") is not True
            or 47126 not in row.get("ancestor_ids", [])
            or type(row.get("id")) is not int or row["id"] < 1
            or binomial_notation(row.get("name")) is None
        ):
            continue
        links = [
            r for r in row.get("names", [])
            if r.get("lexicon") == "scientific-names" and type(r.get("is_valid")) is bool
            and binomial_notation(r.get("name")) == wanted
        ]
        if not links:
            continue
        candidates.append({
            **species, "sourceTaxonRank": "hybrid", "inatTaxonId": row["id"],
            "sourceScientificName": row["name"],
            "sourceNameEvidence": {
                "queryScientificName": species["scientificName"],
                "sourceScientificName": row["name"], "sourceTaxonId": row["id"],
                "sourceTaxonRank": "hybrid", "matchType": "named-hybrid-source",
                "linkedScientificNames": links,
                "note": "Explicit source scientific-name link; only standalone ×/x normalized.",
            },
        })
    if len(candidates) != 1:
        raise ValueError("No unique approved named hybrid source match.")
    validate_hybrid_evidence(candidates[0])
    return candidates[0]


def validate_hybrid_evidence(species):
    evidence = species.get("sourceNameEvidence", {})
    normalized = binomial_notation(species["scientificName"])
    if (
        normalized is None or species.get("sourceTaxonRank") != "hybrid"
        or evidence.get("sourceTaxonRank") != "hybrid"
        or evidence.get("matchType") != "named-hybrid-source"
        or evidence.get("queryScientificName") != species["scientificName"]
        or evidence.get("sourceScientificName") != species.get("sourceScientificName")
        or evidence.get("sourceTaxonId") != species.get("inatTaxonId")
        or binomial_notation(species.get("sourceScientificName")) is None
        or not any(
            r.get("lexicon") == "scientific-names" and type(r.get("is_valid")) is bool
            and binomial_notation(r.get("name")) == normalized
            for r in evidence.get("linkedScientificNames", [])
        )
    ):
        raise ValueError("Named hybrid images require an explicit, matching source bridge.")


def main():
    from yizhu_server.training_data import (
        PublicClient,
        collect,
        dataset_lock,
        now,
        sha256,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text())
    approved = {r["speciesId"]: r for r in targets["classes"]}
    reviewed = json.loads((args.review / "review-summary.json").read_text())["matches"]
    members, bindings = [], {}
    for row in reviewed:
        species = approved[row["speciesId"]]
        cache = args.review / (hashlib.sha256(species["speciesId"].encode()).hexdigest() + ".json")
        record = json.loads(cache.read_text())
        if record["scientificName"] != species["scientificName"]:
            raise ValueError("Reviewed source name differs from target.")
        resolved = resolve_named_hybrid(record["response"], species)
        resolved["sourceNameEvidence"]["queryResponseSha256"] = sha256(cache)
        if resolved["inatTaxonId"] != row["sourceTaxonId"]:
            raise ValueError("Reviewed source identity changed.")
        if resolved["inatTaxonId"] in bindings:
            raise ValueError("Multiple target IDs share the same source taxon.")
        bindings[resolved["inatTaxonId"]] = species["speciesId"]
        members.append(resolved)
    # Prevent two target IDs owning the same source taxon already used by the main pool.
    for path in (args.primary / "taxon-matches").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("status") == "matched" and record["inatTaxonId"] in bindings:
            if bindings[record["inatTaxonId"]] != record["speciesId"]:
                raise ValueError("Source taxon already belongs to another main-pool target.")
    members.sort(key=lambda r: r["scientificName"])
    identity = {
        "policy": "approved-named-hybrid-source-bridge-v1",
        "targetManifestSha256": sha256(args.targets),
        "reviewSha256": sha256(args.review / "review-summary.json"),
        "species": members,
    }
    with dataset_lock(args.data):
        path = args.data / "species.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("Named-hybrid collection scope changed.")
        else:
            write_json(path, identity)
        try:
            report = collect(
                args.data, PublicClient(), identity, class_limit=len(members),
                split_targets={"train": 70, "validation": 15, "test": 15}, max_pages=20,
            )
            write_json(args.data / "completion.json", {
                "completedAt": now(), "images": report["downloadedImages"],
                "readySpeciesCount": report["readySpeciesCount"],
                "sourceSpeciesCount": len(members), "mainDatasetChanged": False,
                "currentTrainingChanged": False, "crossPoolDedupStillRequired": True,
            })
        except Exception as exc:
            write_json(args.data / "stopped.json", {
                "stoppedAt": now(), "errorType": type(exc).__name__, "automaticRetry": False,
            })
            raise


if __name__ == "__main__":
    main()
