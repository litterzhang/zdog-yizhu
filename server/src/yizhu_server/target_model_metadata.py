"""Freeze self-contained metadata for a reviewed target-only BioCLIP classifier."""

import argparse
import json
from pathlib import Path

from yizhu_server.plantnet_transfer import DEFAULT_PRESENCE
from yizhu_server.recognition_rerank import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
)
from yizhu_server.training_data import now, sha256, write_json


def build(model, features, snapshot, solution_id, presence_gate_path=None):
    from safetensors import safe_open

    destination = model / "model.metadata.json"
    if destination.exists():
        raise ValueError("Model metadata is immutable; use a new model directory.")
    results_path = model / "results.json"
    plan_path = model / "plan.json"
    classifier_path = model / "classifier.safetensors"
    results = json.loads(results_path.read_text())
    feature_index = json.loads((features / "index.json").read_text())
    training_path = snapshot / "train-validation.json"
    test_path = snapshot / "test-heldout.json"
    training = json.loads(training_path.read_text())
    test = json.loads(test_path.read_text())
    classes = results["classes"]
    class_count = results["classCount"]
    if (
        results["requestedClassCount"] != len(training["classes"])
        or class_count != len(classes)
        or class_count != len(training["classes"])
        or results.get("coverageGoalMet") is not True
        or feature_index.get("architecture") != MODEL_ID
        or feature_index.get("checkpointSha256") != MODEL_SHA256
        or sha256(features / "features.safetensors") != feature_index["featuresSha256"]
        or test.get("trainingManifestSha256") != sha256(training_path)
    ):
        raise ValueError("Complete target model, features, or snapshot contract is missing.")
    if [row["speciesId"] for row in classes] != [
        row["speciesId"] for row in training["classes"]
    ]:
        raise ValueError("Classifier order differs from the reviewed target order.")
    with safe_open(classifier_path, framework="pt") as stream:
        tensor_keys = set(stream.keys())
    transform = (
        "zscore" if {"feature_mean", "feature_std"} <= tensor_keys
        else "center-l2" if "feature_mean" in tensor_keys
        else "none"
    )
    if not {"weight", "bias"} <= tensor_keys:
        raise ValueError("Classifier tensor contract is incomplete.")
    plants = [{
        "classIndex": index,
        "id": row["speciesId"], "catalogSpeciesId": row["speciesId"],
        "scientificName": row["scientificName"],
        "chineseName": row.get("chineseName"),
        "chineseAliases": row.get("chineseAliases", []),
    } for index, row in enumerate(classes)]
    evidence_path = next(
        (snapshot / name for name in ("admission.json", "audit.json", "snapshot.json")
         if (snapshot / name).is_file()),
        None,
    )
    if evidence_path is None:
        raise ValueError("Snapshot evidence is missing.")
    server = Path(__file__).resolve().parents[2]
    gate_model_path = server / ".artifacts/models/plantnet-china-replay-v2/model.json"
    if presence_gate_path is None:
        gate_model = json.loads(gate_model_path.read_text())
        presence_gate = gate_model["presenceGate"]
        gate_source = {
            "sourceModelFile": str(gate_model_path.relative_to(server)),
            "sourceModelSha256": sha256(gate_model_path),
        }
    else:
        presence_gate = json.loads(presence_gate_path.read_text())
        gate_source = {
            "sourceContractFile": str(presence_gate_path),
            "sourceContractSha256": sha256(presence_gate_path),
        }
    if (
        sha256(DEFAULT_PRESENCE) != presence_gate.get("headSha256")
        or not isinstance(presence_gate.get("threshold"), (int, float))
        or isinstance(presence_gate.get("threshold"), bool)
        or not 0 <= presence_gate["threshold"] <= 1
    ):
        raise ValueError("Pinned plant-presence gate differs from its validated metadata.")
    metadata = {
        "schemaVersion": 1, "createdAt": now(), "solutionId": solution_id,
        "task": "species-identification",
        "status": "trained-offline-development-candidate",
        "productionApproved": False,
        "plantSet": {
            "kind": "finite", "unit": "catalog-species",
            "count": class_count, "plants": plants,
        },
        "encoder": {
            "model": MODEL_ID, "revision": MODEL_REVISION,
            "weightsSha256": MODEL_SHA256,
            "dimension": feature_index["dimension"],
            "normalized": feature_index["normalized"],
        },
        "classifier": {
            "file": classifier_path.name, "sha256": sha256(classifier_path),
            "head": f"{transform}-linear-768-to-{class_count}",
            "featureTransform": transform,
        },
        "presenceGate": {
            **presence_gate,
            "headFile": str(DEFAULT_PRESENCE.relative_to(server)),
            **gate_source,
        },
        "data": {
            "snapshot": str(snapshot),
            "trainingManifestSha256": sha256(training_path),
            "developmentTestManifestSha256": sha256(test_path),
            "snapshotEvidenceFile": evidence_path.name,
            "snapshotEvidenceSha256": sha256(evidence_path),
            "featureIndexSha256": sha256(features / "index.json"),
            "featureWeightsSha256": sha256(features / "features.safetensors"),
            "featureRows": len(feature_index["samples"]),
        },
        "developmentMetrics": results["metrics"],
        "coverageGoalMet": results["coverageGoalMet"],
        "validationGoalMet": results["validationGoalMet"],
        "finalIndependentGoalMet": False,
        "trainingPlanSha256": sha256(plan_path),
        "trainingResultsSha256": sha256(results_path),
        "limitations": [
            "Development validation/test data cannot prove final independent accuracy.",
            "Output-label eligibility is not a per-species accuracy guarantee.",
            "Production remains unchanged until a sealed independent evaluation passes.",
        ],
    }
    write_json(destination, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--solution-id", required=True)
    parser.add_argument("--presence-gate-contract", type=Path)
    args = parser.parse_args()
    result = build(
        args.model.resolve(), args.features.resolve(), args.snapshot.resolve(),
        args.solution_id,
        args.presence_gate_contract.resolve() if args.presence_gate_contract else None,
    )
    print(json.dumps({
        "solutionId": result["solutionId"],
        "classCount": result["plantSet"]["count"],
        "validationGoalMet": result["validationGoalMet"],
        "finalIndependentGoalMet": result["finalIndependentGoalMet"],
    }))


if __name__ == "__main__":
    main()
