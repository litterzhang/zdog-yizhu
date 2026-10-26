"""Extract Pl@ntNet ResNet50 features from a frozen train/validation/test manifest."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from yizhu_server.distributed_plantnet_features import (
  DIMENSION,
  WEIGHTS_SHA256,
  augmented_preprocess,
  load_checkpoint,
  preprocess,
)
from yizhu_server.recognition import read_photo
from yizhu_server.training_data import now, sha256, write_json


def extract(data, training_path, test_path, weights, output, *, threads=4, batch_size=16,
            view_seed=None):
  import torch
  from safetensors.torch import save_file
  from torch.utils.data import DataLoader, Dataset

  if output.exists() or not 1 <= threads <= 8 or not 1 <= batch_size <= 64:
    raise ValueError("Use a new output and bounded CPU settings.")
  training = json.loads(training_path.read_text())
  test = json.loads(test_path.read_text())
  if (
    training.get("testImagesIncluded") is not False
    or test.get("trainingManifestSha256") != sha256(training_path)
    or test.get("classes") != training.get("classes")
    or any(r["split"] not in {"train", "validation"} for r in training["samples"])
    or any(r["split"] != "test" for r in test["samples"])
  ):
    raise ValueError("Require matching frozen train/validation and development-test manifests.")
  rows = sorted(
    training["samples"] + test["samples"],
    key=lambda r: (r["split"], r["speciesId"], r["sampleId"]),
  )
  if view_seed is not None:
    rows = [row for row in rows if row["split"] == "train"]
  if (
    len({r["sampleId"] for r in rows}) != len(rows)
    or len({r["imageSha256"] for r in rows}) != len(rows)
  ):
    raise ValueError("Duplicate sample or image in frozen manifests.")
  for row in rows:
    image = (data / row["path"]).resolve()
    if (
      not image.is_relative_to((data / "images").resolve())
      or not image.is_file() or sha256(image) != row["imageSha256"]
    ):
      raise ValueError("Frozen image missing or changed.")
  transform = augmented_preprocess() if view_seed is not None else preprocess()

  class Photos(Dataset):
    def __len__(self):
      return len(rows)

    def __getitem__(self, index):
      with read_photo(data / rows[index]["path"]) as image:
        if view_seed is None:
          tensor = transform(image)
        else:
          seed = int.from_bytes(
            __import__("hashlib").sha256(
              f"{view_seed}:{rows[index]['sampleId']}".encode(),
            ).digest()[:8], "big",
          ) % (2**63)
          with torch.random.fork_rng():
            torch.manual_seed(seed)
            tensor = transform(image)
        return tensor, index

  torch.set_num_threads(threads)
  torch.set_num_interop_threads(1)
  model = load_checkpoint(weights)
  loader = DataLoader(Photos(), batch_size=batch_size, shuffle=False, num_workers=0)
  output.mkdir(parents=True)
  vectors = torch.empty((len(rows), DIMENSION), dtype=torch.float32)
  started, offset = time.monotonic(), 0
  with torch.inference_mode(), torch.backends.nnpack.flags(enabled=False):
    for images, indices in loader:
      encoded = model(images)
      if (
        encoded.shape != (len(indices), DIMENSION) or not torch.isfinite(encoded).all()
        or not torch.equal(indices, torch.arange(offset, offset + len(indices)))
      ):
        raise ValueError("Invalid or reordered feature batch.")
      vectors[offset:offset + len(indices)].copy_(encoded)
      offset += len(indices)
      status = {
        "updatedAt": now(), "stage": "extracting", "completed": offset, "total": len(rows),
        "elapsedSeconds": time.monotonic() - started,
        "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      }
      write_json(output / "progress.json", status)
      if offset % 160 == 0 or offset == len(rows):
        print(json.dumps(status), flush=True)
  temporary = output / "features.safetensors.tmp"
  save_file({"features": vectors.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  index = {
    "createdAt": now(), "status": "features-extracted",
    "trainingManifestSha256": sha256(training_path),
    "testManifestSha256": sha256(test_path),
    "checkpointSha256": WEIGHTS_SHA256, "architecture": "plantnet-resnet50",
    "dimension": DIMENSION, "normalized": False, "samples": [{
      "row": i,
      "sampleId": (
        f"{r['sampleId']}:feature-view:{view_seed}"
        if view_seed is not None else r["sampleId"]
      ),
      **({"baseSampleId": r["sampleId"],
          "featureView": f"augmented-seed-{view_seed}"} if view_seed is not None else {}),
      **{k: r[k] for k in (
        "speciesId", "imageSha256", "observerGroup", "photoId",
        "observationId", "split",
      )},
    } for i, r in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "splitCounts": {
      split: sum(r["split"] == split for r in rows)
      for split in ("train", "validation", "test")
    },
    "viewSeed": view_seed, "testUsedForTraining": False,
    "trainingStarted": False, "productionChanged": False,
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  }
  write_json(output / "index.json", index)
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows), "total": len(rows),
    "featuresSha256": index["featuresSha256"], "trainingStarted": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--training", type=Path, required=True)
  parser.add_argument("--test", type=Path, required=True)
  parser.add_argument("--weights", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--view-seed", type=int)
  args = parser.parse_args()
  extract(
    args.data.resolve(), args.training.resolve(), args.test.resolve(),
    args.weights.resolve(), args.output.resolve(), threads=args.threads,
    batch_size=args.batch_size, view_seed=args.view_seed,
  )


if __name__ == "__main__":
  main()
