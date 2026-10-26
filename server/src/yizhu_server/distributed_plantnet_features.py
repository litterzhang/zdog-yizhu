"""Extract pinned Pl@ntNet ResNet50 features for one audited remote image partition."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from yizhu_server.recognition import read_photo
from yizhu_server.training_data import now, sha256, write_json

WEIGHTS_BYTES = 206_149_183
WEIGHTS_SHA256 = "4bdaa41d62d0fb9640f6ba137812560d2f2d587ecabe810e3c3571625f72305c"
DIMENSION = 2048


def load_checkpoint(path, class_count=1081):
  import torch
  from torchvision.models import resnet50

  if path.stat().st_size != WEIGHTS_BYTES or sha256(path) != WEIGHTS_SHA256:
    raise ValueError("Pinned Pl@ntNet ResNet50 checkpoint missing or changed.")
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
  if (
    not isinstance(state, dict) or state.get("fc.weight") is None
    or tuple(state["fc.weight"].shape) != (class_count, DIMENSION)
  ):
    raise ValueError("Unexpected pretrained checkpoint structure.")
  model = resnet50(weights=None, num_classes=class_count)
  model.load_state_dict(state, strict=True)
  model.fc = torch.nn.Identity()
  return model.eval().requires_grad_(False)


def preprocess():
  from torchvision import transforms
  from torchvision.transforms import InterpolationMode

  return transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])


def augmented_preprocess():
  from torchvision import transforms
  from torchvision.transforms import InterpolationMode

  return transforms.Compose([
    transforms.RandomResizedCrop(
      224, scale=(.7, 1.0), ratio=(.75, 4 / 3),
      interpolation=InterpolationMode.BILINEAR, antialias=True,
    ),
    transforms.RandomHorizontalFlip(.5),
    transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])


def extract(collection, accepted_path, weights, output, *, threads=2, batch_size=16,
            view_seed=None):
  import torch
  from safetensors.torch import save_file
  from torch.utils.data import DataLoader, Dataset

  if output.exists() or not 1 <= threads <= 8 or not 1 <= batch_size <= 64:
    raise ValueError("Use a new output and bounded CPU settings.")
  accepted_doc = json.loads(accepted_path.read_text())
  if (
    accepted_doc.get("sourceCollection") != collection.name
    or accepted_doc.get("trainingAdmitted") is not False
    or not accepted_doc.get("samples")
  ):
    raise ValueError("Accepted partition metadata differs.")
  expected = {r["sampleId"]: r for r in accepted_doc["samples"]}
  if len(expected) != len(accepted_doc["samples"]):
    raise ValueError("Duplicate accepted sample IDs.")
  metadata = {}
  for path in sorted((collection / "samples").glob("*.json")):
    rows = json.loads(path.read_text()).get("samples", [])
    if len(rows) != 1:
      raise ValueError("Invalid collection sample file.")
    metadata[rows[0]["sampleId"]] = rows[0]
  if not set(expected) <= set(metadata):
    raise ValueError("Accepted samples missing from collection.")
  rows = []
  for sample_id in sorted(expected):
    row, allowed = metadata[sample_id], expected[sample_id]
    if (
      any(row[k] != allowed[k] for k in (
        "sampleId", "speciesId", "photoId", "observationId", "observerGroup",
        "imageSha256", "differenceHash", "preEncodingDifferenceHash", "path",
      ))
      or row.get("split") != "train" or row.get("trainingEligible") is not False
    ):
      raise ValueError("Accepted metadata identity changed.")
    image = collection / row["path"]
    if not image.is_file() or sha256(image) != row["imageSha256"]:
      raise ValueError("Accepted image missing or changed.")
    rows.append(row)
  transform = augmented_preprocess() if view_seed is not None else preprocess()

  class Photos(Dataset):
    def __len__(self):
      return len(rows)

    def __getitem__(self, index):
      with read_photo(collection / rows[index]["path"]) as image:
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
  loader = DataLoader(
    Photos(), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
  )
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
      print(json.dumps(status), flush=True)
  if offset != len(rows) or not torch.isfinite(vectors).all():
    raise ValueError("Incomplete feature extraction.")
  temporary = output / "features.safetensors.tmp"
  save_file({"features": vectors.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  index = {
    "createdAt": now(), "status": "features-extracted", "sourceCollection": collection.name,
    "acceptedManifestSha256": sha256(accepted_path),
    "checkpointSha256": WEIGHTS_SHA256, "architecture": "plantnet-resnet50",
    "preprocessing": {
      "resizeShortEdge": 256, "centerCrop": 224, "interpolation": "bilinear",
      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
    "dimension": DIMENSION, "normalized": False, "samples": [{
      "row": i,
      "sampleId": (
        f"{r['sampleId']}:feature-view:{view_seed}"
        if view_seed is not None else r["sampleId"]
      ),
      **({"baseSampleId": r["sampleId"],
          "featureView": f"augmented-seed-{view_seed}"} if view_seed is not None else {}),
      **{k: r[k] for k in (
        "speciesId", "imageSha256", "observerGroup", "photoId", "observationId",
      )},
    } for i, r in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "viewSeed": view_seed, "trainingStarted": False,
    "testImagesRead": False, "productionChanged": False,
  }
  write_json(output / "index.json", index)
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows), "total": len(rows),
    "featuresSha256": index["featuresSha256"], "trainingStarted": False,
  })
  print(json.dumps({"stage": "complete", "samples": len(rows)}))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--collection", type=Path, required=True)
  parser.add_argument("--accepted", type=Path, required=True)
  parser.add_argument("--weights", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--view-seed", type=int)
  args = parser.parse_args()
  extract(
    args.collection.resolve(), args.accepted.resolve(), args.weights.resolve(),
    args.output.resolve(), threads=args.threads, batch_size=args.batch_size,
    view_seed=args.view_seed,
  )


if __name__ == "__main__":
  main()
