"""Extract PlantCLEF DINOv2 logits as frozen features for a reviewed manifest."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy

from yizhu_server.training_data import now, sha256, write_json

DIMENSION = 7806
IMAGE_SIZE = 518


def _preprocess(path):
  from PIL import Image

  with Image.open(path) as source:
    image = source.convert("RGB")
    width, height = image.size
    scale = IMAGE_SIZE / min(width, height)
    resized = image.resize(
      (round(width * scale), round(height * scale)),
      resample=Image.Resampling.BICUBIC,
    )
    left = (resized.width - IMAGE_SIZE) // 2
    top = (resized.height - IMAGE_SIZE) // 2
    array = numpy.asarray(
      resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)),
      dtype=numpy.float32,
    )
  array = array.transpose(2, 0, 1) / 255.0
  mean = numpy.asarray((0.485, 0.456, 0.406), dtype=numpy.float32)[:, None, None]
  std = numpy.asarray((0.229, 0.224, 0.225), dtype=numpy.float32)[:, None, None]
  return (array - mean) / std


def extract(data, training_path, test_path, model_path, output, *, threads=4,
            batch_size=1):
  import onnxruntime
  import torch
  from safetensors.torch import save_file

  if not 1 <= threads <= 8 or not 1 <= batch_size <= 4:
    raise ValueError("Use bounded PlantCLEF CPU settings.")
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
  for row in rows:
    image = (data / row["path"]).resolve()
    if (
      not image.is_relative_to((data / "images").resolve())
      or not image.is_file() or sha256(image) != row["imageSha256"]
    ):
      raise ValueError("Frozen image missing or changed.")
  model_hash = sha256(model_path)
  output.mkdir(parents=True, exist_ok=True)
  partial_path = output / "features.partial.npy"
  resume_path = output / "resume.json"
  completed = 0
  if partial_path.exists() or resume_path.exists():
    if not partial_path.exists() or not resume_path.exists():
      raise ValueError("Incomplete PlantCLEF resume state.")
    resume = json.loads(resume_path.read_text())
    if (
      resume.get("modelSha256") != model_hash
      or resume.get("trainingManifestSha256") != sha256(training_path)
      or resume.get("testManifestSha256") != sha256(test_path)
      or resume.get("total") != len(rows)
    ):
      raise ValueError("PlantCLEF resume identity changed.")
    completed = resume["completed"]
    vectors = numpy.lib.format.open_memmap(partial_path, mode="r+")
    if vectors.shape != (len(rows), DIMENSION) or vectors.dtype != numpy.float32:
      raise ValueError("Invalid PlantCLEF partial feature array.")
  else:
    vectors = numpy.lib.format.open_memmap(
      partial_path, mode="w+", dtype=numpy.float32,
      shape=(len(rows), DIMENSION),
    )
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
  started = time.monotonic()
  for offset in range(completed, len(rows), batch_size):
    selected = rows[offset:offset + batch_size]
    images = numpy.stack([_preprocess(data / row["path"]) for row in selected])
    logits = session.run(["logits"], {"input": images})[0]
    if logits.shape != (len(selected), DIMENSION) or not numpy.isfinite(logits).all():
      raise ValueError("Invalid PlantCLEF logits.")
    vectors[offset:offset + len(selected)] = logits
    completed = offset + len(selected)
    if completed % 20 == 0 or completed == len(rows):
      vectors.flush()
      status = {
        "updatedAt": now(), "stage": "extracting", "completed": completed,
        "total": len(rows), "elapsedSecondsThisRun": time.monotonic() - started,
        "modelSha256": model_hash,
        "trainingManifestSha256": sha256(training_path),
        "testManifestSha256": sha256(test_path),
        "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
      }
      write_json(resume_path, status)
      write_json(output / "progress.json", status)
      print(json.dumps(status), flush=True)
  tensor = torch.from_numpy(numpy.asarray(vectors))
  temporary = output / "features.safetensors.tmp"
  save_file({"features": tensor.contiguous()}, temporary)
  os.replace(temporary, output / "features.safetensors")
  write_json(output / "index.json", {
    "createdAt": now(), "status": "features-extracted",
    "architecture": "PlantCLEF-2024-DINOv2-7806-logits",
    "checkpointSha256": model_hash, "dimension": DIMENSION,
    "normalized": False,
    "trainingManifestSha256": sha256(training_path),
    "testManifestSha256": sha256(test_path),
    "samples": [{
      "row": index, **{key: row[key] for key in (
        "sampleId", "speciesId", "imageSha256", "observerGroup", "photoId",
        "observationId", "split",
      )},
    } for index, row in enumerate(rows)],
    "featuresSha256": sha256(output / "features.safetensors"),
    "trainingStarted": False, "testUsedForTraining": False,
    "productionChanged": False,
    "elapsedSecondsThisRun": time.monotonic() - started,
    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
  })
  partial_path.unlink()
  resume_path.unlink()
  write_json(output / "progress.json", {
    "updatedAt": now(), "stage": "complete", "completed": len(rows),
    "total": len(rows), "trainingStarted": False,
  })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--training", type=Path, required=True)
  parser.add_argument("--test", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=1)
  args = parser.parse_args()
  extract(
    args.data.resolve(), args.training.resolve(), args.test.resolve(),
    args.model.resolve(), args.output.resolve(), threads=args.threads,
    batch_size=args.batch_size,
  )


if __name__ == "__main__":
  main()
