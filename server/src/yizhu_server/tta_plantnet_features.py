"""Extract a fixed horizontal-flip center-crop view for validation/test TTA."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from yizhu_server.distributed_plantnet_features import (
  DIMENSION,
  WEIGHTS_SHA256,
  load_checkpoint,
)
from yizhu_server.recognition import read_photo
from yizhu_server.training_data import now, sha256, write_json

VIEWS = {
  "hflip-center", "top-left", "top-right", "bottom-left", "bottom-right",
}


def transform(view):
  from torchvision import transforms
  from torchvision.transforms import InterpolationMode, functional

  if view not in VIEWS:
    raise ValueError("Unsupported fixed TTA view.")

  def crop(image):
    width, height = image.size
    if min(width, height) < 224:
      raise ValueError("Resized image is unexpectedly smaller than crop.")
    positions = {
      "hflip-center": ((height - 224) // 2, (width - 224) // 2),
      "top-left": (0, 0),
      "top-right": (0, width - 224),
      "bottom-left": (height - 224, 0),
      "bottom-right": (height - 224, width - 224),
    }
    top, left = positions[view]
    result = functional.crop(image, top, left, 224, 224)
    return functional.hflip(result) if view == "hflip-center" else result

  return transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.Lambda(crop),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])


def extract(data, training_path, test_path, weights, output, *, view="hflip-center",
            threads=4, batch_size=16):
  import torch
  from safetensors.torch import save_file
  from torch.utils.data import DataLoader, Dataset

  if output.exists():
    raise ValueError("Use a new TTA feature output.")
  training = json.loads(training_path.read_text())
  test = json.loads(test_path.read_text())
  if (
    training.get("testImagesIncluded") is not False
    or test.get("trainingManifestSha256") != sha256(training_path)
    or test.get("classes") != training.get("classes")
  ):
    raise ValueError("Frozen manifests differ.")
  rows = sorted(
    [r for r in training["samples"] if r["split"] == "validation"] + test["samples"],
    key=lambda r: (r["split"], r["speciesId"], r["sampleId"]),
  )
  if any(r["split"] not in {"validation", "test"} for r in rows):
    raise ValueError("TTA extraction must not include training images.")
  for row in rows:
    image = (data / row["path"]).resolve()
    if (
      not image.is_relative_to((data / "images").resolve())
      or not image.is_file() or sha256(image) != row["imageSha256"]
    ):
      raise ValueError("Frozen image missing or changed.")
  apply = transform(view)

  class Photos(Dataset):
    def __len__(self):
      return len(rows)

    def __getitem__(self, index):
      with read_photo(data / rows[index]["path"]) as image:
        return apply(image), index

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
        raise ValueError("Invalid or reordered TTA features.")
      vectors[offset:offset + len(indices)].copy_(encoded)
      offset += len(indices)
      write_json(output / "progress.json", {
        "updatedAt": now(), "stage": "extracting", "completed": offset, "total": len(rows),
        "elapsedSeconds": time.monotonic() - started,
      })
  temporary = output / "features.safetensors.tmp"
  save_file({"features": vectors.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  write_json(output / "index.json", {
    "createdAt": now(), "status": "features-extracted",
    "checkpointSha256": WEIGHTS_SHA256, "dimension": DIMENSION, "normalized": False,
    "featureView": f"resize256-{view}",
    "samples": [{
      "row": i, "sampleId": f"{r['sampleId']}:feature-view:{view}",
      "baseSampleId": r["sampleId"], "featureView": view,
      **{k: r[k] for k in (
        "speciesId", "imageSha256", "observerGroup", "photoId", "observationId", "split",
      )},
    } for i, r in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "testUsedForTraining": False, "trainingStarted": False,
    "productionChanged": False,
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  })
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows), "total": len(rows),
    "trainingStarted": False,
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
  parser.add_argument("--view", choices=sorted(VIEWS), default="hflip-center")
  args = parser.parse_args()
  extract(
    args.data.resolve(), args.training.resolve(), args.test.resolve(),
    args.weights.resolve(), args.output.resolve(), threads=args.threads,
    batch_size=args.batch_size, view=args.view,
  )


if __name__ == "__main__":
  main()
