"""Admit audited GBIF/Commons candidates into a new reference-only target snapshot."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from yizhu_server.training_data import (
    ROOT,
    DuplicateIndex,
    group_split,
    now,
    sha256,
    write_json,
)

ALLOWED_LICENSES = {"cc0", "cc-by"}


def _perceptual(value):
    return int(value, 16) if isinstance(value, str) else int(value)


def _normalized_hybrid_name(value):
    return " ".join(re.sub(r"(?<=\s)[×x](?=\s)", " ", value).split())


def _candidate_image(root, row):
    path = (root / row["path"]).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or sha256(path) != row["imageSha256"]
    ):
        raise ValueError("Candidate image is missing, escapes its source, or changed.")
    return path


def _gbif_row(root, candidate, target, completed_at):
    if (
        candidate["speciesId"] != target["speciesId"]
        or candidate["scientificName"] != target["scientificName"]
        or candidate.get("trainingEligible") is not False
    ):
        raise ValueError("GBIF candidate target identity changed.")
    license_url = (candidate.get("mediaLicense") or "").lower()
    if "publicdomain/zero/1.0" in license_url:
        license_name = "cc0"
    elif "licenses/by/4.0" in license_url:
        license_name = "cc-by"
    else:
        raise ValueError("GBIF candidate license is not approved for training.")
    author = str(
        candidate.get("mediaCreator") or candidate.get("mediaRightsHolder")
        or candidate.get("recordedBy") or candidate.get("datasetKey")
    ).strip()
    if not author:
        raise ValueError("GBIF candidate attribution is missing.")
    observer = hashlib.sha256(
        f"gbif:{candidate.get('datasetKey')}:{author.casefold()}".encode()
    ).hexdigest()
    occurrence = candidate["gbifOccurrenceKey"]
    return {
        "sampleId": candidate["candidateId"],
        "speciesId": target["speciesId"],
        "scientificName": target["scientificName"],
        "sourceScientificName": candidate.get(
            "gbifAcceptedSpeciesName", target["scientificName"]
        ),
        "sourceNameEvidence": {
            "matchType": "gbif-exact-species-or-explicit-accepted-synonym",
            "queryScientificName": target["scientificName"],
            "matchedScientificName": candidate.get("gbifMatchedScientificName"),
            "acceptedScientificName": candidate.get("gbifAcceptedSpeciesName"),
            "queryTaxonKey": candidate.get("gbifQueryTaxonKey"),
            "acceptedTaxonKey": candidate.get("gbifAcceptedTaxonKey"),
            "confidence": 100,
        },
        "observationId": f"gbif-occurrence:{occurrence}",
        "photoId": candidate["candidateId"],
        "observerGroup": observer,
        "split": group_split(observer),
        "sourceUrl": candidate.get("sourceReferences"),
        "imageUrl": candidate["sourceImageUrl"],
        "license": license_name,
        "attribution": author,
        "licenseUrl": candidate["mediaLicense"],
        "labelSource": candidate["labelEvidence"],
        "retrievedAt": completed_at,
        "path": candidate["path"],
        "imageSha256": candidate["imageSha256"],
        "differenceHash": f"{_perceptual(candidate['differenceHash']):016x}",
        "modifications": "EXIF removed; RGB JPEG; maximum edge 2048px",
        "basisOfRecord": candidate.get("basisOfRecord"),
        "datasetKey": candidate.get("datasetKey"),
        "trainingEligible": True,
        "admissionStatus": "admitted-after-exact-taxonomy-license-dedup-audit",
    }


def _commons_row(candidate, target):
    if (
        candidate["speciesId"] != target["speciesId"]
        or candidate["scientificName"] != target["scientificName"]
        or candidate.get("trainingEligible") is not False
        or candidate.get("labelReviewStatus") != "metadata-only"
        or candidate["license"] not in ALLOWED_LICENSES
    ):
        raise ValueError("Commons candidate identity, review state, or license changed.")
    observer = candidate["authorGroup"]
    source_name = candidate.get("sourceScientificName", target["scientificName"])
    source_evidence = candidate.get("sourceNameEvidence")
    if (
        _normalized_hybrid_name(source_name)
        != _normalized_hybrid_name(target["scientificName"])
        and not (
            isinstance(source_evidence, dict)
            and source_evidence.get("queryScientificName") == target["scientificName"]
            and source_evidence.get("sourceScientificName") == source_name
            and source_evidence.get("matchType")
            == "gbif-explicit-accepted-species-synonym"
        )
    ):
        raise ValueError("Commons source-name bridge is missing or does not match.")
    return {
        "sampleId": candidate["id"],
        "speciesId": target["speciesId"],
        "scientificName": target["scientificName"],
        "sourceScientificName": source_name,
        "sourceNameEvidence": {
            "matchType": "commons-species-category-and-explicit-scientific-name",
            "queryScientificName": target["scientificName"],
            "sourceScientificName": source_name,
            "sourceCategory": candidate["sourceCategory"],
            "sourceTitle": candidate["sourceTitle"],
            **({"taxonomyBridge": source_evidence} if source_evidence else {}),
        },
        "observationId": candidate["id"],
        "photoId": candidate["id"],
        "observerGroup": observer,
        "split": group_split(observer),
        "sourceUrl": candidate["sourceUrl"],
        "imageUrl": candidate["imageUrl"],
        "license": candidate["license"],
        "attribution": candidate["attribution"],
        "licenseUrl": candidate["licenseUrl"],
        "labelSource": candidate["labelEvidence"],
        "retrievedAt": candidate["downloadedAt"],
        "path": candidate["path"],
        "imageSha256": candidate["imageSha256"],
        "differenceHash": f"{_perceptual(candidate['differenceHash']):016x}",
        "modifications": candidate["modifications"],
        "trainingEligible": True,
        "admissionStatus": "admitted-after-commons-metadata-license-dedup-audit",
    }


def load_candidates(roots, class_by_id):
    rows = []
    evidence = []
    for root in roots:
        progress_path = root / "progress.json"
        progress = json.loads(progress_path.read_text())
        if progress.get("stage") != "collection-pass-finished":
            raise ValueError("Candidate collection is not complete.")
        candidate_path = root / "candidates.json"
        candidates = json.loads(candidate_path.read_text())
        is_commons = root.name.startswith("commons-")
        if progress.get("trainingEligible") is not False:
            raise ValueError("Candidates were unexpectedly pre-approved.")
        for candidate in candidates:
            target = class_by_id.get(candidate["speciesId"])
            if target is None:
                raise ValueError("Candidate species is outside the reviewed target list.")
            image = _candidate_image(root, candidate)
            row = (
                _commons_row(candidate, target) if is_commons
                else _gbif_row(root, candidate, target, progress["completedAt"])
            )
            rows.append((row, image, root.name))
        evidence.append({
            "path": str(root), "progressSha256": sha256(progress_path),
            "candidatesSha256": sha256(candidate_path), "candidateCount": len(candidates),
        })
    return rows, evidence


def admit(base_rows, candidates, classes):
    duplicates = DuplicateIndex()
    keys = set()
    counts = defaultdict(Counter)
    for row in base_rows:
        duplicates.add(row["imageSha256"], _perceptual(row["differenceHash"]))
        for key in ("sampleId", "photoId", "observationId"):
            keys.add((key, str(row[key])))
        counts[row["speciesId"]][row["split"]] += 1
    admitted, rejected = [], Counter()
    author_counts = Counter()
    for row, image, source in candidates:
        perceptual = _perceptual(row["differenceHash"])
        row_keys = [(key, str(row[key])) for key in ("sampleId", "photoId", "observationId")]
        if any(key in keys for key in row_keys):
            rejected[f"{source}:duplicate-identity"] += 1
            continue
        if duplicates.contains(row["imageSha256"], perceptual):
            rejected[f"{source}:exact-or-near-duplicate"] += 1
            continue
        author_key = row["speciesId"], row["observerGroup"]
        if author_counts[author_key] >= 5:
            rejected[f"{source}:author-cap"] += 1
            continue
        admitted.append((row, image, source))
        duplicates.add(row["imageSha256"], perceptual)
        keys.update(row_keys)
        author_counts[author_key] += 1

    # Keep author groups intact. Only classes with no prior training evidence receive
    # one deterministic group override, ensuring the classifier cannot claim a class
    # merely because validation/test candidates exist.
    by_species = defaultdict(list)
    for row, image, source in admitted:
        by_species[row["speciesId"]].append((row, image, source))
    overrides = set()
    for target in classes:
        sid = target["speciesId"]
        if counts[sid]["train"] or any(row["split"] == "train" for row, _, _ in by_species[sid]):
            continue
        groups = sorted({row["observerGroup"] for row, _, _ in by_species[sid]})
        if groups:
            overrides.add(groups[0])
    for row, _, _ in admitted:
        if row["observerGroup"] in overrides:
            row["split"] = "train"
        counts[row["speciesId"]][row["split"]] += 1
    return admitted, rejected, overrides, counts


def build(base, candidate_roots, output):
    if output.exists():
        raise ValueError("Use a new admitted snapshot directory.")
    training_path = base / "train-validation.json"
    test_path = base / "test-heldout.json"
    training = json.loads(training_path.read_text())
    test = json.loads(test_path.read_text())
    classes = training["classes"]
    if (
        len(classes) != 249
        or training.get("testImagesIncluded") is not False
        or test.get("classes") != classes
        or test.get("trainingManifestSha256") != sha256(training_path)
    ):
        raise ValueError("Require the audited 249-class reference snapshot.")
    class_by_id = {row["speciesId"]: row for row in classes}
    base_rows = training["samples"] + test["samples"]
    candidates, evidence = load_candidates(candidate_roots, class_by_id)
    admitted, rejected, overrides, counts = admit(base_rows, candidates, classes)
    combined = list(base_rows)
    for row, image, source in admitted:
        combined.append({
            **row,
            "sourceDataset": str(image.parent.parent.relative_to(ROOT.parent)),
            "sourcePath": str(image.relative_to(ROOT.parent)),
        })
    combined.sort(key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])))
    no_train = [
        row for row in classes if counts[row["speciesId"]]["train"] == 0
    ]
    output.mkdir(parents=True)
    write_json(output / "species.json", {
        "createdAt": now(), "policy": "reviewed-target-candidate-admission-v1",
        "species": classes, "requestedClassCount": len(classes),
    })
    write_json(output / "train-validation.json", {
        **{key: value for key, value in training.items() if key != "samples"},
        "createdAt": now(), "samples": [row for row in combined if row["split"] != "test"],
        "scope": "249 reviewed species plus audited GBIF/Commons supplemental candidates.",
    })
    write_json(output / "test-heldout.json", {
        **{key: value for key, value in test.items()
           if key not in {"samples", "trainingManifestSha256"}},
        "createdAt": now(), "samples": [row for row in combined if row["split"] == "test"],
        "trainingManifestSha256": sha256(output / "train-validation.json"),
    })
    write_json(output / "admission.json", {
        "createdAt": now(), "status": "candidates-admitted-reference-snapshot",
        "requestedClassCount": len(classes),
        "trainingClassCount": len(classes) - len(no_train),
        "missingTrainingClasses": no_train,
        "baseRows": len(base_rows), "candidateRows": len(candidates),
        "admittedRows": len(admitted), "combinedRows": len(combined),
        "rejected": dict(rejected), "groupOverridesToTrain": sorted(overrides),
        "splitCounts": dict(Counter(row["split"] for row in combined)),
        "perSpecies": [{
            **{key: row[key] for key in ("speciesId", "scientificName", "chineseName")},
            "counts": dict(counts[row["speciesId"]]),
        } for row in classes],
        "sources": evidence, "imagesCopied": False, "productionChanged": False,
    })
    return json.loads((output / "admission.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.base.resolve(), [path.resolve() for path in args.candidate],
        args.output.resolve(),
    )
    print(json.dumps({
        key: report[key] for key in (
            "requestedClassCount", "trainingClassCount", "candidateRows",
            "admittedRows", "combinedRows", "splitCounts",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
