"""Pinned PlantCLEF DINOv2 assets and local closed-set inference; no hub downloads."""

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

RECORD = "10.5281/zenodo.10848263"
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
CLASS_COUNT = 7806
ARCHIVE_BYTES = 2_271_216_640
ARCHIVE_MD5 = "42bc1bf3725ebe0bc93a2167c6b3deda"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache/models/plantclef-2024/prepared"
MEMBERS = {
  "class_mapping.txt": (
    "pretrained_models/class_mapping.txt", 62448,
    "85e16f8f9fd4cbbe8d0e3d268a9180a37defe4a320066e85af3521915a5cdfa7",
  ),
  "species_id_to_name.txt": (
    "pretrained_models/species_id_to_name.txt", 324013,
    "995b9bbfa02675069eea12e2b646f3629baf4fb54df7987836a93f71f61965cd",
  ),
  "checkpoint.pth.tar": (
    "pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar",
    1_481_641_544, None,
  ),
}


def checksum(path: Path, algorithm="sha256") -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, algorithm).hexdigest()


def prepare_archive(archive: Path, output: Path) -> dict:
  """Verify the complete publisher archive before reading any serialized checkpoint."""
  if output.exists() or output.is_symlink():
    raise ValueError("Prepared directory exists; use a new directory.")
  if archive.stat().st_size != ARCHIVE_BYTES or checksum(archive, "md5") != ARCHIVE_MD5:
    raise ValueError("PlantCLEF archive does not match the published size/checksum.")
  import tarfile

  import torch
  from safetensors.torch import save_file

  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="plantclef-", dir=output.parent) as temporary:
    staging = Path(temporary)
    with tarfile.open(archive, mode="r:") as tar:
      for filename, (member_name, size, sha) in MEMBERS.items():
        member = tar.getmember(member_name)
        if not member.isfile() or member.size != size:
          raise ValueError("Unexpected model archive member.")
        path = staging / filename
        with tar.extractfile(member) as source, path.open("xb") as destination:
          shutil.copyfileobj(source, destination, length=1024 * 1024)
        if sha is not None and checksum(path) != sha:
          raise ValueError("PlantCLEF class/name map checksum mismatch.")
    # timm checkpoints may include inert argparse metadata. Permit only that
    # standard-library container, never arbitrary publisher classes/functions.
    # These arguments are ignored: architecture/transforms are pinned below.
    with torch.serialization.safe_globals([argparse.Namespace]):
      checkpoint = torch.load(
        staging / "checkpoint.pth.tar", map_location="cpu", weights_only=True, mmap=True,
      )
    state = checkpoint.get("state_dict_ema") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
      raise ValueError("Expected the published EMA state dictionary.")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
      raise ValueError("Non-tensor value in EMA weights.")
    if tuple(state["head.weight"].shape) != (CLASS_COUNT, 768):
      raise ValueError("Unexpected PlantCLEF classifier shape.")
    save_file({key: value.detach().contiguous() for key, value in state.items()},
              staging / "model.safetensors")
    del state, checkpoint
    gc.collect()
    (staging / "checkpoint.pth.tar").unlink()
    manifest = {
      "format": 1, "record": RECORD, "model": MODEL, "classCount": CLASS_COUNT,
      "weights": "state_dict_ema", "archiveMd5": ARCHIVE_MD5,
      "archiveSha256": checksum(archive), "license": "CC-BY-4.0",
      "attribution": "Goëau, Lombardo, Affouard, Espitalier, Bonnet, Joly (2024)",
      "modifications": "EMA tensors converted to safetensors; no retraining.",
      "files": {
        name: {"size": (staging / name).stat().st_size, "sha256": checksum(staging / name)}
        for name in ("class_mapping.txt", "species_id_to_name.txt", "model.safetensors")
      },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
      raise ValueError("Prepared directory appeared during installation.")
    os.rename(staging, output)
  return manifest


def verify_assets(directory: Path) -> dict:
  path = directory / "manifest.json"
  if not path.is_file() or path.stat().st_size > 16_384:
    raise ValueError("Prepare local PlantCLEF weights before selecting this solution.")
  manifest = json.loads(path.read_text())
  if (manifest.get("format") != 1 or manifest.get("record") != RECORD
      or manifest.get("model") != MODEL or manifest.get("classCount") != CLASS_COUNT
      or manifest.get("archiveMd5") != ARCHIVE_MD5 or manifest.get("weights") != "state_dict_ema"):
    raise ValueError("Unexpected PlantCLEF manifest.")
  for filename in ("class_mapping.txt", "species_id_to_name.txt", "model.safetensors"):
    info = manifest["files"][filename]
    path = directory / filename
    if (not path.is_file() or path.stat().st_size != info["size"]
        or checksum(path) != info["sha256"]):
      raise ValueError("Prepared PlantCLEF file checksum mismatch.")
    if filename in MEMBERS:
      _, size, sha = MEMBERS[filename]
      if info["size"] != size or info["sha256"] != sha:
        raise ValueError("Class/name map is not the published snapshot.")
  return manifest


def scientific_binomial(raw: str) -> str | None:
  match = re.fullmatch(r"([A-Z][a-z]+ [a-z][a-z-]*)(?:\s+(.*))?", raw)
  if not match:
    return None
  name, author = match.groups()
  return name if not author or author[0].isupper() or author.startswith("(") else None


def class_references(class_ids: list[str], names: dict[str, str], catalog_rows: list) -> list[dict]:
  if len(class_ids) != CLASS_COUNT or len(set(class_ids)) != CLASS_COUNT or any(
      identifier not in names for identifier in class_ids):
    raise ValueError("Invalid PlantCLEF output-index mapping.")
  binomials = [scientific_binomial(names[identifier]) for identifier in class_ids]
  counts = Counter(binomials)
  catalog_counts = Counter(name for _, name in catalog_rows)
  catalog = {name: identifier for identifier, name in catalog_rows if catalog_counts[name] == 1}
  return [
    {
      "class_index": index, "source_species_id": source_id,
      "source_scientific_name": names[source_id],
      "scientific_name": name or names[source_id],
      "species_id": catalog.get(name) if name and counts[name] == 1 else None,
    }
    for index, (source_id, name) in enumerate(zip(class_ids, binomials, strict=True))
  ]


def class_result(references: list[dict], probabilities: list[float], *, limit: int, manifest: dict) -> dict:
  import math

  if (len(probabilities) != len(references)
      or any(isinstance(s, bool) or not isinstance(s, (float, int))
             or not math.isfinite(s) or not 0 <= s <= 1 for s in probabilities)):
    raise ValueError("Invalid PlantCLEF class scores.")
  indices = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))[:limit]
  return {
    "source": "PlantCLEF", "model": MODEL, "revision": RECORD + ":ema",
    "weights_sha256": manifest["files"]["model.safetensors"]["sha256"],
    "license": "CC-BY-4.0", "score_type": "closed_set_softmax_not_correctness_probability",
    "scope": "plantclef_2024_7806_classes", "is_demo": False,
    "candidates": [
      {**references[i], "kind": "plant" if references[i]["species_id"] else "unmapped",
       "score": float(probabilities[i]), "confidence": None}
      for i in indices
    ],
  }


class PlantCLEFProvider:
  def __init__(self, catalog, *, directory=DEFAULT_DIRECTORY, threads=2):
    if not 1 <= threads <= 4:
      raise ValueError("Use 1–4 inference threads.")
    self.directory = directory
    self.manifest = verify_assets(directory)
    ids = (directory / "class_mapping.txt").read_text().splitlines()
    with (directory / "species_id_to_name.txt").open() as stream:
      rows = list(csv.DictReader(stream, delimiter=";"))
    names = {row["species_id"]: row["species"] for row in rows}
    if len(names) != len(rows):
      raise ValueError("Ambiguous PlantCLEF source IDs.")
    with catalog.connect() as db:
      catalog_rows = db.execute("SELECT id,scientific_name FROM species").fetchall()
    self.references = class_references(ids, names, catalog_rows)
    self.threads = threads
    self.model = None
    self.preprocess = None

  def _load(self):
    if self.model is not None:
      return
    import timm
    import torch
    from safetensors.torch import load_file
    from timm.data import create_transform

    torch.set_num_threads(self.threads)
    model = timm.create_model(MODEL, pretrained=False, num_classes=CLASS_COUNT, img_size=518)
    state = load_file(self.directory / "model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    model.requires_grad_(False)
    self.preprocess = create_transform(
      input_size=(3, 518, 518), is_training=False, crop_pct=1.0, interpolation="bicubic",
      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
    self.model = model

  def recognize(self, image, *, top_k=20):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
      raise ValueError("PlantCLEF top_k must be between 1 and 20.")
    import torch

    self._load()
    with torch.inference_mode():
      logits = self.model(self.preprocess(image).unsqueeze(0))
      if logits.shape != (1, CLASS_COUNT) or not torch.isfinite(logits).all():
        raise ValueError("Invalid PlantCLEF output tensor.")
      scores = logits.softmax(dim=-1)[0].tolist()
    return class_result(self.references, scores, limit=top_k, manifest=self.manifest)


def main():
  parser = argparse.ArgumentParser(description="Verify official archive and prepare local EMA weights.")
  parser.add_argument("archive", type=Path)
  parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
  args = parser.parse_args()
  try:
    print(json.dumps(prepare_archive(args.archive, args.output), ensure_ascii=False, indent=2))
  except (OSError, ValueError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
    parser.exit(1, f"PlantCLEF preparation failed: {error}\n")


if __name__ == "__main__":
  main()
