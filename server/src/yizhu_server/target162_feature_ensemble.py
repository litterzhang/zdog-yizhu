"""Select a fixed BioCLIP/PlantCLEF probability blend on development validation only."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def metrics(logits, expected, class_ids):
  ranking = logits.topk(5, 1).indices.tolist()
  per = defaultdict(lambda: [0, 0, 0])
  top1 = top5 = 0
  for target, ranked in zip(expected, ranking, strict=True):
    predicted = [class_ids[index] for index in ranked]
    one, five = predicted[0] == target, target in predicted
    top1 += one
    top5 += five
    row = per[target]
    row[0] += 1
    row[1] += one
    row[2] += five
  return {
    "samples": len(expected), "representedSpecies": len(per),
    "top1": top1 / len(expected), "top5": top5 / len(expected),
    "macroTop1": sum(one / count for count, one, _ in per.values()) / len(per),
    "macroTop5": sum(five / count for count, _, five in per.values()) / len(per),
  }


def blend(logits_a, logits_b, alpha, temperature_a, temperature_b):
  import torch

  a = torch.log_softmax(logits_a / temperature_a, dim=1)
  b = torch.log_softmax(logits_b / temperature_b, dim=1)
  if alpha == 0:
    return b
  if alpha == 1:
    return a
  return torch.logaddexp(a + math.log(alpha), b + math.log1p(-alpha))


def run(bioclip_features, bioclip_model, plantclef_features, plantclef_model,
        manifest_path, output):
  import torch
  from safetensors.torch import load_file

  if output.exists():
    raise ValueError("Use a new ensemble output.")
  manifest = json.loads(manifest_path.read_text())
  if len(manifest["classes"]) != 162:
    raise ValueError("Require the full reviewed 162-class manifest.")
  bio_index = json.loads((bioclip_features / "index.json").read_text())
  plant_index = json.loads((plantclef_features / "index.json").read_text())
  bio_rows, plant_rows = bio_index["samples"], plant_index["samples"]
  identity_keys = (
    "sampleId", "speciesId", "imageSha256", "observerGroup",
    "photoId", "observationId", "split",
  )
  if (
    len(bio_rows) != len(plant_rows)
    or any(
      any(a[key] != b[key] for key in identity_keys)
      for a, b in zip(bio_rows, plant_rows, strict=True)
    )
  ):
    raise ValueError("Feature sources do not represent the same frozen samples.")
  bio_vectors = load_file(bioclip_features / "features.safetensors")["features"]
  plant_vectors = load_file(plantclef_features / "features.safetensors")["features"]
  bio_result = json.loads((bioclip_model / "results.json").read_text())
  plant_result = json.loads((plantclef_model / "results.json").read_text())
  bio_ids = [row["speciesId"] for row in bio_result["classes"]]
  plant_ids = [row["speciesId"] for row in plant_result["classes"]]
  if bio_ids != plant_ids or len(bio_ids) != 162:
    raise ValueError("Classifier class orders differ.")
  bio_state = load_file(bioclip_model / "classifier.safetensors")
  plant_state = load_file(plantclef_model / "classifier.safetensors")
  bio_logits = bio_vectors @ bio_state["weight"].T + bio_state["bias"]
  centered = torch.nn.functional.normalize(
    plant_vectors - plant_state["feature_mean"], dim=1,
  )
  plant_logits = centered @ plant_state["weight"].T + plant_state["bias"]

  splits = {}
  for split in ("validation", "test"):
    indices = [i for i, row in enumerate(bio_rows) if row["split"] == split]
    splits[split] = {
      "bio": bio_logits[indices], "plant": plant_logits[indices],
      "expected": [bio_rows[i]["speciesId"] for i in indices],
    }
  temperatures = (.5, .75, 1., 1.5, 2., 3.)
  alphas = tuple(index / 20 for index in range(21))
  best = None
  history = []
  for temperature_a in temperatures:
    for temperature_b in temperatures:
      for alpha in alphas:
        combined = blend(
          splits["validation"]["bio"], splits["validation"]["plant"],
          alpha, temperature_a, temperature_b,
        )
        result = metrics(combined, splits["validation"]["expected"], bio_ids)
        key = (
          result["macroTop1"], result["macroTop5"],
          result["top1"], result["top5"],
        )
        row = {
          "bioClipProbabilityWeight": alpha,
          "bioClipTemperature": temperature_a,
          "plantClefTemperature": temperature_b,
          **result,
        }
        history.append(row)
        if best is None or key > best[0]:
          best = key, row
  selected = best[1]
  results = {}
  for split, rows in splits.items():
    results[split] = metrics(
      blend(
        rows["bio"], rows["plant"],
        selected["bioClipProbabilityWeight"],
        selected["bioClipTemperature"], selected["plantClefTemperature"],
      ),
      rows["expected"], bio_ids,
    )
  output.mkdir(parents=True)
  write_json(output / "plan.json", {
    "createdAt": now(), "selectionSplit": "validation",
    "testUsedForSelection": False,
    "bioClipModelSha256": sha256(bioclip_model / "classifier.safetensors"),
    "plantClefModelSha256": sha256(plantclef_model / "classifier.safetensors"),
    "temperatureGrid": list(temperatures), "alphaGrid": list(alphas),
    "candidateCount": len(history), "productionChanged": False,
  })
  write_json(output / "results.json", {
    "createdAt": now(), "classCount": 162, "selected": selected,
    "metrics": results, "coverageGoalMet": True,
    "validationGoalMet": (
      results["validation"]["macroTop1"] > .8
      and results["validation"]["macroTop5"] > .9
    ),
    "finalIndependentGoalMet": False, "testUsedForSelection": False,
    "productionApproved": False,
  })
  write_json(output / "search.json", history)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--bioclip-features", type=Path, required=True)
  parser.add_argument("--bioclip-model", type=Path, required=True)
  parser.add_argument("--plantclef-features", type=Path, required=True)
  parser.add_argument("--plantclef-model", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  run(
    args.bioclip_features.resolve(), args.bioclip_model.resolve(),
    args.plantclef_features.resolve(), args.plantclef_model.resolve(),
    args.manifest.resolve(), args.output.resolve(),
  )


if __name__ == "__main__":
  main()
