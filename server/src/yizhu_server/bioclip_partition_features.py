"""Extract normalized BioCLIP 2 features for a machine-local image partition."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from yizhu_server.recognition import read_photo
from yizhu_server.recognition_rerank import MODEL_SHA256, BioClipReranker
from yizhu_server.training_data import now, sha256, write_json

DIMENSION = 768


def extract(partition_path, output, *, threads=4, batch_size=8):
  import torch
  from safetensors.torch import save_file
  from torch.utils.data import DataLoader, Dataset

  if output.exists() or not 1 <= threads <= 4 or not 1 <= batch_size <= 16:
    raise ValueError("Use a new output and bounded BioCLIP CPU settings.")
  document = json.loads(partition_path.read_text())
  rows = document.get("samples", [])
  if (
    document.get("purpose") not in {
      "target162-bioclip-feature-partition",
      "target-species-bioclip-feature-partition",
    }
    or not rows or len({row["sampleId"] for row in rows}) != len(rows)
  ):
    raise ValueError("Invalid BioCLIP feature partition.")
  for row in rows:
    image = Path(row["sourcePath"])
    if not image.is_absolute() or not image.is_file() or sha256(image) != row["imageSha256"]:
      raise ValueError("Partition image missing or changed.")
  encoder = BioClipReranker(threads=threads)

  class Photos(Dataset):
    def __len__(self):
      return len(rows)

    def __getitem__(self, index):
      with read_photo(Path(rows[index]["sourcePath"])) as image:
        return encoder.preprocess(image), index

  loader = DataLoader(Photos(), batch_size=batch_size, shuffle=False, num_workers=0)
  output.mkdir(parents=True)
  vectors = torch.empty((len(rows), DIMENSION), dtype=torch.float32)
  started, offset = time.monotonic(), 0
  with torch.inference_mode():
    for images, indices in loader:
      encoded = torch.nn.functional.normalize(encoder.model.encode_image(images), dim=1)
      if (
        encoded.shape != (len(indices), DIMENSION) or not torch.isfinite(encoded).all()
        or not torch.equal(indices, torch.arange(offset, offset + len(indices)))
      ):
        raise ValueError("Invalid or reordered BioCLIP features.")
      vectors[offset:offset + len(indices)].copy_(encoded)
      offset += len(indices)
      if offset % 40 == 0 or offset == len(rows):
        status = {
          "updatedAt": now(), "stage": "extracting", "completed": offset,
          "total": len(rows), "elapsedSeconds": time.monotonic() - started,
          "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        write_json(output / "progress.json", status)
        print(json.dumps(status), flush=True)
  temporary = output / "features.safetensors.tmp"
  save_file({"features": vectors.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  index = {
    "createdAt": now(), "status": "features-extracted",
    "architecture": "imageomics/bioclip-2", "checkpointSha256": MODEL_SHA256,
    "dimension": DIMENSION, "normalized": True,
    "partitionSha256": sha256(partition_path),
    "samples": [{
      "row": position,
      **{key: row[key] for key in (
        "sampleId", "speciesId", "imageSha256", "observerGroup", "photoId",
        "observationId", "split",
      )},
    } for position, row in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "trainingStarted": False, "testUsedForTraining": False,
    "productionChanged": False, "elapsedSeconds": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  }
  write_json(output / "index.json", index)
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows),
    "total": len(rows), "trainingStarted": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--partition", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=8)
  args = parser.parse_args()
  extract(
    args.partition.resolve(), args.output.resolve(),
    threads=args.threads, batch_size=args.batch_size,
  )


if __name__ == "__main__":
  main()
