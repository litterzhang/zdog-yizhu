"""Run one frozen BioCLIP 2 classifier evaluation on a sealed independent manifest."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from yizhu_server.recognition import read_photo
from yizhu_server.recognition_rerank import MODEL_SHA256, BioClipReranker
from yizhu_server.training_data import now, sha256, write_json


def evaluate(manifest_path, model_dir, output, *, threads=4, batch_size=8):
  import torch
  from safetensors.torch import load_file
  from torch.utils.data import DataLoader, Dataset

  from yizhu_server.plant_presence import PlantPresenceClassifier

  if output.exists():
    raise ValueError("Independent evaluation output already exists; never overwrite it.")
  manifest = json.loads(manifest_path.read_text())
  metadata = json.loads((model_dir / "model.metadata.json").read_text())
  samples = manifest.get("samples", [])
  plants = metadata.get("plantSet", {}).get("plants", [])
  class_count = metadata.get("plantSet", {}).get("count")
  if (
    manifest.get("status") != "complete" or manifest.get("imagesVerified") is not True
    or manifest.get("requestedSpecies") != class_count
    or manifest.get("representedSpecies") != class_count
    or manifest.get("modelOutputsReadDuringSelection") is not False
    or metadata.get("status") != "trained-offline-development-candidate"
    or metadata.get("productionApproved") is not False
    or metadata.get("encoder", {}).get("weightsSha256") != MODEL_SHA256
    or metadata.get("classifier", {}).get("sha256")
       != sha256(model_dir / "classifier.safetensors")
    or not isinstance(class_count, int) or class_count < 2
    or len(plants) != class_count or not samples
  ):
    raise ValueError("Independent manifest or frozen model contract is incomplete.")
  class_ids = [row["id"] for row in sorted(plants, key=lambda row: row["classIndex"])]
  class_index = {sid: index for index, sid in enumerate(class_ids)}
  if set(class_ids) != {row["speciesId"] for row in samples}:
    raise ValueError("Independent manifest does not exactly represent the frozen plant set.")
  for row in samples:
    raw_path = Path(row["path"])
    image = raw_path if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
    if (
      row["speciesId"] not in class_index
      or (not raw_path.is_absolute() and not image.is_relative_to(manifest_path.parent.resolve()))
      or not image.is_file() or sha256(image) != row["imageSha256"]
    ):
      raise ValueError("Independent image or label identity changed.")
    row["_resolvedPath"] = str(image)

  encoder = BioClipReranker(threads=threads)
  server = Path(__file__).resolve().parents[2]
  gate = metadata.get("presenceGate", {})
  gate_head = server / gate.get("headFile", "")
  if (
    not gate_head.is_file() or sha256(gate_head) != gate.get("headSha256")
    or not isinstance(gate.get("threshold"), (int, float))
    or isinstance(gate.get("threshold"), bool)
    or not 0 <= gate["threshold"] <= 1
  ):
    raise ValueError("Frozen plant-presence gate contract is incomplete.")
  presence = PlantPresenceClassifier(gate_head, threads=threads)
  if presence.encoder.identity != gate.get("encoder"):
    raise ValueError("Plant-presence encoder identity changed.")
  weights = load_file(model_dir / "classifier.safetensors")
  if set(weights) not in ({"weight", "bias"}, {"weight", "bias", "feature_mean"}):
    raise ValueError("Unsupported frozen BioCLIP classifier transform.")
  head = torch.nn.Linear(768, class_count)
  head.load_state_dict(
    {"weight": weights["weight"], "bias": weights["bias"]}, strict=True,
  )
  head.eval().requires_grad_(False)

  class Photos(Dataset):
    def __len__(self):
      return len(samples)

    def __getitem__(self, index):
      with read_photo(Path(samples[index]["_resolvedPath"])) as image:
        return encoder.preprocess(image), index

  loader = DataLoader(Photos(), batch_size=batch_size, shuffle=False, num_workers=0)
  predictions = [None] * len(samples)
  presence_scores = [None] * len(samples)
  with torch.inference_mode():
    for images, indices in loader:
      features = torch.nn.functional.normalize(encoder.model.encode_image(images), dim=1)
      if "feature_mean" in weights:
        features = torch.nn.functional.normalize(
          features - weights["feature_mean"], dim=1,
        )
      ranking = head(features).topk(5, 1).indices.tolist()
      for source_index, ranked in zip(indices.tolist(), ranking, strict=True):
        predictions[source_index] = [class_ids[index] for index in ranked]
    for start in range(0, len(samples), batch_size):
      for source_index in range(start, min(start + batch_size, len(samples))):
        with read_photo(Path(samples[source_index]["_resolvedPath"])) as image:
          presence_scores[source_index] = presence.predict(image)["score"]
  if any(row is None for row in predictions) or any(
    score is None for score in presence_scores
  ):
    raise ValueError("Independent inference did not cover every sample.")

  per_class = defaultdict(lambda: {"samples": 0, "top1Correct": 0, "top5Correct": 0})
  top1 = top5 = end_to_end_top1 = end_to_end_top5 = passed = 0
  for sample, ranked, presence_score in zip(
    samples, predictions, presence_scores, strict=True,
  ):
    expected = sample["speciesId"]
    one = ranked[0] == expected
    five = expected in ranked
    gate_passed = presence_score >= gate["threshold"]
    top1 += one
    top5 += five
    passed += gate_passed
    end_to_end_top1 += gate_passed and one
    end_to_end_top5 += gate_passed and five
    row = per_class[expected]
    row["samples"] += 1
    row["top1Correct"] += one
    row["top5Correct"] += five
    row["plantGatePassed"] = row.get("plantGatePassed", 0) + gate_passed
    row["endToEndTop1Correct"] = row.get("endToEndTop1Correct", 0) + (
      gate_passed and one
    )
    row["endToEndTop5Correct"] = row.get("endToEndTop5Correct", 0) + (
      gate_passed and five
    )
  for sid, row in per_class.items():
    row["top1"] = row["top1Correct"] / row["samples"]
    row["top5"] = row["top5Correct"] / row["samples"]
    row["plantPassRate"] = row["plantGatePassed"] / row["samples"]
    row["endToEndTop1"] = row["endToEndTop1Correct"] / row["samples"]
    row["endToEndTop5"] = row["endToEndTop5Correct"] / row["samples"]
  macro_top1 = sum(row["top1"] for row in per_class.values()) / len(per_class)
  macro_top5 = sum(row["top5"] for row in per_class.values()) / len(per_class)
  macro_end_to_end_top1 = sum(
    row["endToEndTop1"] for row in per_class.values()
  ) / len(per_class)
  macro_end_to_end_top5 = sum(
    row["endToEndTop5"] for row in per_class.values()
  ) / len(per_class)
  result = {
    "createdAt": now(), "status": "final-independent-evaluation",
    "manifestSha256": sha256(manifest_path),
    "modelMetadataSha256": sha256(model_dir / "model.metadata.json"),
    "classifierSha256": sha256(model_dir / "classifier.safetensors"),
    "samples": len(samples), "representedSpecies": len(per_class),
    "top1": top1 / len(samples), "top5": top5 / len(samples),
    "macroTop1": macro_top1, "macroTop5": macro_top5,
    "plantPassRate": passed / len(samples),
    "endToEndTop1": end_to_end_top1 / len(samples),
    "endToEndTop5": end_to_end_top5 / len(samples),
    "macroEndToEndTop1": macro_end_to_end_top1,
    "macroEndToEndTop5": macro_end_to_end_top5,
    "presenceGate": {
      "headSha256": gate["headSha256"], "threshold": gate["threshold"],
      "confidenceCalibrated": False,
    },
    "perClass": dict(per_class),
    "goalMet": (
      len(per_class) == class_count
      and macro_end_to_end_top1 > .8 and macro_end_to_end_top5 > .9
    ),
    "productionChanged": False, "evaluatedOnce": True,
  }
  output.mkdir(parents=True)
  write_json(output / "results.json", result)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=8)
  args = parser.parse_args()
  evaluate(
    args.manifest.resolve(), args.model.resolve(), args.output.resolve(),
    threads=args.threads, batch_size=args.batch_size,
  )


if __name__ == "__main__":
  main()
