"""Extract PlantCLEF DINOv2 logits for an audited machine-local image partition."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy

from yizhu_server.plantclef_manifest_features import DIMENSION, IMAGE_SIZE, _preprocess
from yizhu_server.training_data import now, sha256, write_json


def extract(partition_path, model_path, output, *, threads=4, batch_size=1):
  import onnxruntime
  import torch
  from safetensors.torch import save_file

  if output.exists() or not 1 <= threads <= 8 or not 1 <= batch_size <= 4:
    raise ValueError("Use a new output and bounded PlantCLEF CPU settings.")
  document = json.loads(partition_path.read_text())
  rows = document.get("samples", [])
  if (
    document.get("purpose") != "target162-plantclef-feature-partition"
    or not rows or len({row["sampleId"] for row in rows}) != len(rows)
  ):
    raise ValueError("Invalid PlantCLEF feature partition.")
  for row in rows:
    image = Path(row["sourcePath"])
    if not image.is_absolute() or not image.is_file() or sha256(image) != row["imageSha256"]:
      raise ValueError("Partition image missing or changed.")
  options = onnxruntime.SessionOptions()
  options.intra_op_num_threads = threads
  options.inter_op_num_threads = 1
  session = onnxruntime.InferenceSession(
    str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
  )
  if (
    [(item.name, item.shape) for item in session.get_inputs()]
    != [("input", ["batch", 3, IMAGE_SIZE, IMAGE_SIZE])]
    or [(item.name, item.shape) for item in session.get_outputs()]
    != [("logits", ["batch", DIMENSION])]
  ):
    raise ValueError("Unexpected PlantCLEF ONNX contract.")
  output.mkdir(parents=True)
  vectors = torch.empty((len(rows), DIMENSION), dtype=torch.float32)
  started = time.monotonic()
  for offset in range(0, len(rows), batch_size):
    selected = rows[offset:offset + batch_size]
    images = numpy.stack([_preprocess(Path(row["sourcePath"])) for row in selected])
    logits = session.run(["logits"], {"input": images})[0]
    if logits.shape != (len(selected), DIMENSION) or not numpy.isfinite(logits).all():
      raise ValueError("Invalid PlantCLEF logits.")
    vectors[offset:offset + len(selected)].copy_(torch.from_numpy(logits))
    completed = offset + len(selected)
    if completed % 20 == 0 or completed == len(rows):
      status = {
        "updatedAt": now(), "stage": "extracting", "completed": completed,
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
    "architecture": "PlantCLEF-2024-DINOv2-7806-logits",
    "checkpointSha256": sha256(model_path), "dimension": DIMENSION,
    "normalized": False, "partitionSha256": sha256(partition_path),
    "samples": [{
      "row": index,
      **{key: row[key] for key in (
        "sampleId", "speciesId", "imageSha256", "observerGroup", "photoId",
        "observationId", "split",
      )},
    } for index, row in enumerate(rows)],
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
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=1)
  args = parser.parse_args()
  extract(
    args.partition.resolve(), args.model.resolve(), args.output.resolve(),
    threads=args.threads, batch_size=args.batch_size,
  )


if __name__ == "__main__":
  main()
