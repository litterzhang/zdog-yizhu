"""Merge audited feature partitions into the exact frozen-manifest row order."""

import argparse
import json
import os
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def merge(training_path, test_path, partitions, output):
  import torch
  from safetensors.torch import load_file, save_file

  if output.exists() or len(partitions) < 2:
    raise ValueError("Use a new output and at least two feature partitions.")
  training = json.loads(training_path.read_text())
  test = json.loads(test_path.read_text())
  if (
    training.get("testImagesIncluded") is not False
    or test.get("trainingManifestSha256") != sha256(training_path)
    or test.get("classes") != training.get("classes")
  ):
    raise ValueError("Frozen target manifests differ.")
  rows = sorted(
    training["samples"] + test["samples"],
    key=lambda row: (row["split"], row["speciesId"], row["sampleId"]),
  )
  expected = {row["sampleId"]: row for row in rows}
  gathered, identities = {}, set()
  for root in partitions:
    index = json.loads((root / "index.json").read_text())
    vectors = load_file(root / "features.safetensors")["features"]
    identity = (
      index.get("architecture"), index.get("checkpointSha256"),
      index.get("dimension"), index.get("normalized"),
    )
    identities.add(identity)
    if (
      index.get("status") != "features-extracted"
      or sha256(root / "features.safetensors") != index.get("featuresSha256")
      or vectors.shape != (len(index.get("samples", [])), index.get("dimension"))
    ):
      raise ValueError("Invalid feature partition.")
    for position, sample in enumerate(index["samples"]):
      sample_id = sample["sampleId"]
      source = expected.get(sample_id)
      if (
        source is None or sample_id in gathered
        or any(sample[key] != source[key] for key in (
          "speciesId", "imageSha256", "observerGroup", "photoId",
          "observationId", "split",
        ))
      ):
        raise ValueError("Feature partition sample identity changed or duplicated.")
      gathered[sample_id] = vectors[position]
  if len(identities) != 1 or set(gathered) != set(expected):
    raise ValueError("Feature partitions do not exactly cover the frozen manifest.")
  architecture, checkpoint, dimension, normalized = identities.pop()
  merged = torch.stack([gathered[row["sampleId"]] for row in rows])
  output.mkdir(parents=True)
  temporary = output / "features.safetensors.tmp"
  save_file({"features": merged.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  write_json(output / "index.json", {
    "createdAt": now(), "status": "features-extracted",
    "architecture": architecture, "checkpointSha256": checkpoint,
    "dimension": dimension, "normalized": normalized,
    "trainingManifestSha256": sha256(training_path),
    "testManifestSha256": sha256(test_path),
    "samples": [{
      "row": index, **{key: row[key] for key in (
        "sampleId", "speciesId", "imageSha256", "observerGroup", "photoId",
        "observationId", "split",
      )},
    } for index, row in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "sourcePartitions": [{
      "path": str(root), "indexSha256": sha256(root / "index.json"),
    } for root in partitions],
    "trainingStarted": False, "testUsedForTraining": False,
    "productionChanged": False,
  })
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows),
    "total": len(rows), "trainingStarted": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--training", type=Path, required=True)
  parser.add_argument("--test", type=Path, required=True)
  parser.add_argument("--partition", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  merge(
    args.training.resolve(), args.test.resolve(),
    [path.resolve() for path in args.partition], args.output.resolve(),
  )


if __name__ == "__main__":
  main()
