"""Merge a validated base snapshot with reviewed new-species image collections."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from yizhu_server.training_data import (
    ROOT,
    DuplicateIndex,
    group_split,
    load_samples,
    now,
    sha256,
    write_json,
)


def _perceptual(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return int(value, 16)
    raise ValueError("Invalid perceptual hash.")


def _relative(path):
    return str(path.resolve().relative_to(ROOT.parent.resolve()))


def _source_row(row, image):
    return {**row, "sourceDataset": _relative(image.parent.parent), "sourcePath": _relative(image)}


def _base_rows(base):
    training_path, test_path = base / "train-validation.json", base / "test-heldout.json"
    training, test = json.loads(training_path.read_text()), json.loads(test_path.read_text())
    if (
        training.get("testImagesIncluded") is not False
        or test.get("classes") != training.get("classes")
        or test.get("trainingManifestSha256") != sha256(training_path)
    ):
        raise ValueError("Base snapshot manifests differ.")
    rows = training["samples"] + test["samples"]
    for row in rows:
        image = (ROOT.parent / row["sourcePath"]).resolve()
        if not image.is_file() or sha256(image) != row["imageSha256"]:
            raise ValueError("Base snapshot image changed.")
    return rows, training, test, training_path, test_path


def _inat_rows(root):
    result = []
    for row in load_samples(root):
        image = root / row["path"]
        if not image.is_file() or sha256(image) != row["imageSha256"]:
            raise ValueError("iNaturalist expansion image changed.")
        result.append(_source_row(row, image))
    return result


def _inat_additional_rows(root):
    candidates = json.loads((root / "candidates.json").read_text())
    result = []
    for row in candidates:
        image = root / row["path"]
        if (
            row.get("trainingEligible") is not False
            or not image.is_file() or sha256(image) != row["imageSha256"]
        ):
            raise ValueError("iNaturalist additional candidate contract changed.")
        projected = {
            "sampleId": row["candidateId"], "speciesId": row["speciesId"],
            "scientificName": row["scientificName"],
            "sourceScientificName": row["scientificName"],
            "sourceNameEvidence": {
                "queryScientificName": row["scientificName"],
                "matchType": "inaturalist-research-grade-exact-species",
            },
            "observationId": f"inat:{row['sourceObservationId']}",
            "photoId": row["candidateId"], "observerGroup": row["authorGroup"],
            "split": group_split(row["authorGroup"]),
            "sourceUrl": row["sourceReferences"], "imageUrl": row["sourceImageUrl"],
            "license": row["license"], "attribution": row["attribution"],
            "licenseUrl": row["licenseUrl"], "labelSource": row["labelEvidence"],
            "retrievedAt": now(), "path": row["path"],
            "imageSha256": row["imageSha256"],
            "differenceHash": f"{_perceptual(row['differenceHash']):016x}",
            "modifications": "RGB JPEG re-encode; metadata stripped",
        }
        result.append(_source_row(projected, image))
    return result


def _gbif_rows(root):
    candidates = json.loads((root / "candidates.json").read_text())
    result = []
    for row in candidates:
        image = root / row["path"]
        if (
            row.get("trainingEligible") is not False
            or not image.is_file() or sha256(image) != row["imageSha256"]
        ):
            raise ValueError("GBIF candidate contract changed.")
        author = (
            row.get("mediaCreator") or row.get("mediaRightsHolder")
            or row.get("recordedBy") or row.get("datasetKey")
        )
        observer = hashlib.sha256(
            f"gbif:{row.get('datasetKey')}:{str(author).casefold()}".encode()
        ).hexdigest()
        license_url = row["mediaLicense"].lower()
        license_name = "cc0" if "/zero/1.0" in license_url else "cc-by"
        projected = {
            "sampleId": row["candidateId"], "speciesId": row["speciesId"],
            "scientificName": row["scientificName"],
            "sourceScientificName": row.get("gbifMatchedScientificName", row["scientificName"]),
            "sourceNameEvidence": {
                "queryScientificName": row["scientificName"],
                "sourceScientificName": row.get("gbifMatchedScientificName"),
                "acceptedScientificName": row.get("gbifAcceptedSpeciesName"),
                "queryTaxonKey": row.get("gbifQueryTaxonKey"),
                "acceptedTaxonKey": row.get("gbifAcceptedTaxonKey"),
                "matchType": "gbif-exact-species",
            },
            "observationId": f"gbif:{row['gbifOccurrenceKey']}",
            "photoId": row["candidateId"], "observerGroup": observer,
            "split": group_split(observer), "sourceUrl": row.get("sourceReferences"),
            "imageUrl": row["sourceImageUrl"], "license": license_name,
            "attribution": str(author), "licenseUrl": row["mediaLicense"],
            "labelSource": row["labelEvidence"], "retrievedAt": now(),
            "path": row["path"], "imageSha256": row["imageSha256"],
            "differenceHash": f"{_perceptual(row["differenceHash"]):016x}",
            "modifications": "RGB JPEG re-encode; max 2048px; metadata stripped",
        }
        result.append(_source_row(projected, image))
    return result



def _commons_rows(root):
    candidates = json.loads((root / "candidates.json").read_text())
    result = []
    for row in candidates:
        image = root / row["path"]
        if (
            row.get("trainingEligible") is not False
            or row.get("split") != "unassigned"
            or not image.is_file() or sha256(image) != row["imageSha256"]
        ):
            raise ValueError("Commons candidate contract changed.")
        projected = {
            "sampleId": row["id"], "speciesId": row["speciesId"],
            "scientificName": row["scientificName"],
            "sourceScientificName": row.get("sourceScientificName", row["scientificName"]),
            "observationId": row["id"], "photoId": row["id"],
            "observerGroup": row["authorGroup"],
            "split": group_split(row["authorGroup"]),
            "sourceUrl": row["sourceUrl"], "imageUrl": row["imageUrl"],
            "license": row["license"], "attribution": row["attribution"],
            "licenseUrl": row["licenseUrl"], "labelSource": row["labelEvidence"],
            "retrievedAt": row["downloadedAt"], "path": row["path"],
            "imageSha256": row["imageSha256"],
            "differenceHash": f"{_perceptual(row['differenceHash']):016x}",
            "modifications": row["modifications"],
        }
        result.append(_source_row(projected, image))
    return result


def build(base, targets_path, inat_roots, inat_additional_roots, gbif_roots, commons_roots, output):
    if output.exists():
        raise ValueError("Use a new expansion snapshot directory.")
    targets = json.loads(targets_path.read_text())
    classes = targets["classes"]
    base_rows, training, test, training_path, test_path = _base_rows(base)
    base_ids = {row["speciesId"] for row in training["classes"]}
    class_by_id = {row["speciesId"]: row for row in classes}
    if (
        len(base_ids) != len(training["classes"])
        or len(class_by_id) != len(classes) or not base_ids < set(class_by_id)
        or len(classes) != targets.get("targetClassCount")
    ):
        raise ValueError("Target classes must be a strict species expansion of the base snapshot.")
    pools = [("base", base_rows)]
    pools.extend((f"inat-{index}", _inat_rows(root)) for index, root in enumerate(inat_roots, 1))
    pools.extend(
        (f"inat-additional-{index}", _inat_additional_rows(root))
        for index, root in enumerate(inat_additional_roots, 1)
    )
    pools.extend((f"gbif-{index}", _gbif_rows(root)) for index, root in enumerate(gbif_roots, 1))
    pools.extend((f"commons-{index}", _commons_rows(root)) for index, root in enumerate(commons_roots, 1))
    duplicates, identities, selected, rejected = DuplicateIndex(), set(), [], Counter()
    for pool_name, rows in pools:
        for row in rows:
            target = class_by_id.get(row["speciesId"])
            if target is None:
                rejected[f"{pool_name}:outside-target"] += 1
                continue
            if row["scientificName"] != target["scientificName"]:
                raise ValueError("Sample scientific name differs from reviewed target.")
            keys = tuple((key, str(row[key])) for key in ("sampleId", "photoId", "observationId"))
            if any(key in identities for key in keys):
                rejected[f"{pool_name}:duplicate-identity"] += 1
                continue
            perceptual = _perceptual(row["differenceHash"])
            if duplicates.contains(row["imageSha256"], perceptual):
                rejected[f"{pool_name}:exact-or-near-duplicate"] += 1
                continue
            identities.update(keys)
            duplicates.add(row["imageSha256"], perceptual)
            selected.append(row)
    base_groups = {row["observerGroup"] for row in base_rows}
    preliminary = defaultdict(Counter)
    for row in selected:
        preliminary[row["speciesId"]][row["split"]] += 1
    overrides = set()
    for target in classes:
        sid = target["speciesId"]
        if preliminary[sid]["train"]:
            continue
        groups = sorted({
            row["observerGroup"] for row in selected
            if row["speciesId"] == sid and row["observerGroup"] not in base_groups
        })
        if groups:
            overrides.add(groups[0])
    for row in selected:
        if row["observerGroup"] in overrides:
            row["split"] = "train"
    counts = defaultdict(Counter)
    for row in selected:
        counts[row["speciesId"]][row["split"]] += 1
    no_train = [row for row in classes if not counts[row["speciesId"]]["train"]]
    output.mkdir(parents=True)
    write_json(output / "species.json", {
        "createdAt": now(), "policy": "reviewed-species-expansion-snapshot-v1",
        "species": classes, "requestedClassCount": len(classes),
        "targetManifestSha256": sha256(targets_path),
    })
    train_validation = sorted(
        (row for row in selected if row["split"] != "test"),
        key=lambda row: (row["split"], row["speciesId"], str(row["sampleId"])),
    )
    heldout = sorted(
        (row for row in selected if row["split"] == "test"),
        key=lambda row: (row["speciesId"], str(row["sampleId"])),
    )
    write_json(output / "train-validation.json", {
        "createdAt": now(), "classes": classes,
        "targetSpeciesIds": [row["speciesId"] for row in classes],
        "samples": train_validation, "testImagesIncluded": False,
        "pretrainedWeightsAllowed": True, "pretrainedClassifierRowsAllowed": True,
        "automaticDeployment": False,
        "scope": f"{len(classes)} reviewed species; source images referenced, not copied.",
    })
    write_json(output / "test-heldout.json", {
        "createdAt": now(), "classes": classes, "samples": heldout,
        "trainingManifestSha256": sha256(output / "train-validation.json"),
        "independentFinalBenchmark": False,
        "note": "Observer-group development holdout; not final independent acceptance data.",
    })
    audit = {
        "createdAt": now(), "status": "reference-snapshot-audited",
        "requestedClassCount": len(classes),
        "trainingClassCount": len(classes) - len(no_train),
        "missingTrainingClassCount": len(no_train),
        "missingTrainingClasses": [{
            "speciesId": row["speciesId"], "scientificName": row["scientificName"],
            "chineseName": row["chineseName"], "counts": dict(counts[row["speciesId"]]),
        } for row in no_train],
        "baseRows": len(base_rows), "mergedDistinctRows": len(selected),
        "sourceRows": {name: len(rows) for name, rows in pools},
        "splitCounts": dict(Counter(row["split"] for row in selected)),
        "perSpecies": [{
            "speciesId": row["speciesId"], "scientificName": row["scientificName"],
            "chineseName": row["chineseName"], "counts": dict(counts[row["speciesId"]]),
        } for row in classes],
        "rejected": dict(rejected), "groupOverridesToTrain": sorted(overrides),
        "targetManifestSha256": sha256(targets_path),
        "baseTrainingSha256": sha256(training_path), "baseTestSha256": sha256(test_path),
        "imagesCopied": False, "modelWeightsOrFeaturesRead": False,
        "productionChanged": False,
    }
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--inat", type=Path, action="append", default=[])
    parser.add_argument("--inat-additional", type=Path, action="append", default=[])
    parser.add_argument("--gbif", type=Path, action="append", default=[])
    parser.add_argument("--commons", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.base.resolve(), args.targets.resolve(),
        [path.resolve() for path in args.inat],
        [path.resolve() for path in args.inat_additional],
        [path.resolve() for path in args.gbif],
        [path.resolve() for path in args.commons], args.output.resolve(),
    )
    print(json.dumps({
        key: report[key] for key in (
            "requestedClassCount", "trainingClassCount", "missingTrainingClassCount",
            "mergedDistinctRows", "splitCounts",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
