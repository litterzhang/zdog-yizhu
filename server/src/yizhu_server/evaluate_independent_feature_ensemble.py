"""Evaluate one frozen BioCLIP/PlantCLEF ensemble on sealed independent features."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def evaluate(manifest_path, ensemble_dir, bioclip_features, plantclef_features,
             output):
  import torch
  from safetensors.torch import load_file

  if output.exists():
    raise ValueError("Independent ensemble result already exists; never overwrite.")
  manifest = json.loads(manifest_path.read_text())
  metadata = json.loads((ensemble_dir / "model.metadata.json").read_text())
  spec = json.loads((ensemble_dir / "ensemble.json").read_text())
  if (
    manifest.get("status") != "complete" or manifest.get("imagesVerified") is not True
    or manifest.get("representedSpecies") != 162
    or manifest.get("modelOutputsReadDuringSelection") is not False
    or metadata.get("status") != "trained-offline-development-candidate"
    or metadata.get("productionApproved") is not False
    or metadata.get("classifier", {}).get("sha256")
       != sha256(ensemble_dir / "ensemble.json")
    or spec.get("classCount") != 162 or spec.get("testUsedForSelection") is not False
  ):
    raise ValueError("Independent manifest or ensemble freeze contract is incomplete.")
  plants = sorted(metadata["plantSet"]["plants"], key=lambda row: row["classIndex"])
  class_ids = [row["id"] for row in plants]
  if set(class_ids) != {row["speciesId"] for row in manifest["samples"]}:
    raise ValueError("Independent set does not exactly represent the ensemble plant set.")

  bio_index = json.loads((bioclip_features / "index.json").read_text())
  plant_index = json.loads((plantclef_features / "index.json").read_text())
  bio_rows, plant_rows = bio_index["samples"], plant_index["samples"]
  if (
    len(bio_rows) != len(manifest["samples"])
    or len(plant_rows) != len(manifest["samples"])
  ):
    raise ValueError("Independent feature counts differ from sealed manifest.")
  expected = {
    row["sampleId"]: (row["speciesId"], row["imageSha256"])
    for row in manifest["samples"]
  }
  for rows in (bio_rows, plant_rows):
    if set(row["sampleId"] for row in rows) != set(expected):
      raise ValueError("Independent feature sample IDs differ.")
    if any(
      (row["speciesId"], row["imageSha256"]) != expected[row["sampleId"]]
      for row in rows
    ):
      raise ValueError("Independent feature label/image identity changed.")

  bio_vectors_raw = load_file(bioclip_features / "features.safetensors")["features"]
  plant_vectors_raw = load_file(plantclef_features / "features.safetensors")["features"]
  bio_by_id = {row["sampleId"]: bio_vectors_raw[i] for i, row in enumerate(bio_rows)}
  plant_by_id = {row["sampleId"]: plant_vectors_raw[i] for i, row in enumerate(plant_rows)}
  ordered = manifest["samples"]
  bio_vectors = torch.stack([bio_by_id[row["sampleId"]] for row in ordered])
  plant_vectors = torch.stack([plant_by_id[row["sampleId"]] for row in ordered])

  bio_model = Path(metadata["components"]["bioClip"]["modelDir"])
  plant_model = Path(metadata["components"]["plantClef"]["modelDir"])
  if (
    sha256(bio_model / "classifier.safetensors")
    != spec["bioClipClassifierSha256"]
    or sha256(plant_model / "classifier.safetensors")
    != spec["plantClefClassifierSha256"]
  ):
    raise ValueError("Frozen ensemble component changed.")
  bio_state = load_file(bio_model / "classifier.safetensors")
  plant_state = load_file(plant_model / "classifier.safetensors")
  bio_logits = bio_vectors @ bio_state["weight"].T + bio_state["bias"]
  plant_normalized = torch.nn.functional.normalize(
    plant_vectors - plant_state["feature_mean"], dim=1,
  )
  plant_logits = plant_normalized @ plant_state["weight"].T + plant_state["bias"]
  bio_logp = torch.log_softmax(
    bio_logits / spec["bioClipTemperature"], dim=1,
  )
  plant_logp = torch.log_softmax(
    plant_logits / spec["plantClefTemperature"], dim=1,
  )
  alpha = spec["bioClipProbabilityWeight"]
  logits = torch.logaddexp(
    bio_logp + math.log(alpha), plant_logp + math.log1p(-alpha),
  )
  ranking = logits.topk(5, 1).indices.tolist()
  per = defaultdict(lambda: {"samples": 0, "top1Correct": 0, "top5Correct": 0})
  top1 = top5 = 0
  for sample, ranked in zip(ordered, ranking, strict=True):
    expected_id = sample["speciesId"]
    predicted = [class_ids[index] for index in ranked]
    one, five = predicted[0] == expected_id, expected_id in predicted
    top1 += one
    top5 += five
    row = per[expected_id]
    row["samples"] += 1
    row["top1Correct"] += one
    row["top5Correct"] += five
  for row in per.values():
    row["top1"] = row["top1Correct"] / row["samples"]
    row["top5"] = row["top5Correct"] / row["samples"]
  macro_top1 = sum(row["top1"] for row in per.values()) / len(per)
  macro_top5 = sum(row["top5"] for row in per.values()) / len(per)
  output.mkdir(parents=True)
  write_json(output / "results.json", {
    "createdAt": now(), "status": "final-independent-ensemble-evaluation",
    "manifestSha256": sha256(manifest_path),
    "ensembleMetadataSha256": sha256(ensemble_dir / "model.metadata.json"),
    "ensembleSpecSha256": sha256(ensemble_dir / "ensemble.json"),
    "samples": len(ordered), "representedSpecies": len(per),
    "top1": top1 / len(ordered), "top5": top5 / len(ordered),
    "macroTop1": macro_top1, "macroTop5": macro_top5,
    "perClass": dict(per),
    "goalMet": (
      len(per) == 162 and macro_top1 > .8 and macro_top5 > .9
    ),
    "evaluatedOnce": True, "productionChanged": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--ensemble", type=Path, required=True)
  parser.add_argument("--bioclip-features", type=Path, required=True)
  parser.add_argument("--plantclef-features", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  evaluate(
    args.manifest.resolve(), args.ensemble.resolve(),
    args.bioclip_features.resolve(), args.plantclef_features.resolve(),
    args.output.resolve(),
  )


if __name__ == "__main__":
  main()
