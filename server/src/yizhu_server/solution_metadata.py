"""Versioned solution capabilities, derived from local labels; never load a model."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from yizhu_server.catalog import ASSETS, CATALOG_PATH, PlantCatalog
from yizhu_server.plantnet_pretrained import read_class_map, sha256
from yizhu_server.recognition import MODEL_ID, MODEL_REVISION, MODEL_SHA256, load_labels
from yizhu_server.recognition_plantclef import MEMBERS, scientific_binomial
from yizhu_server.recognition_rerank import MODEL_ID as RERANK_MODEL_ID
from yizhu_server.recognition_rerank import MODEL_REVISION as RERANK_REVISION
from yizhu_server.recognition_rerank import MODEL_SHA256 as RERANK_SHA256
from yizhu_server.recognition_solutions import SOLUTIONS
from yizhu_server.training_data import write_json

SERVER = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = SERVER / ".cache/solution-metadata"
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,179}")


def encoded(value):
  return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value):
  return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path, limit=96 * 1024 * 1024):
  if not 0 < path.stat().st_size <= limit:
    raise ValueError("Metadata file has an invalid size.")
  return json.loads(path.read_text())


def immutable(path, value):
  content = encoded(value)
  if path.exists():
    if sha256(path) != hashlib.sha256(content).hexdigest():
      raise ValueError("Immutable metadata snapshot changed.")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("xb") as stream:
    stream.write(content)


def not_applicable():
  return {
    "kind": "not-applicable", "count": None, "plants": None,
    "reason": "Binary plant presence does not identify species or claim coverage of all plants.",
  }


def artifact_metadata(path):
  """Small self-contained sidecar for trained heads; never rewrites model bytes."""
  model = read_json(path, 8 * 1024 * 1024)
  if model.get("task") == "plant-presence" and model.get("labels") == ["nonplant", "plant"]:
    return {
      "schemaVersion": 1, "task": "plant-presence",
      "artifactSha256": sha256(path), "modelFile": path.name,
      "encoder": model["encoder"], "datasetSha256": model["datasetSha256"],
      "classLabels": model["labels"], "plantSet": not_applicable(),
      "accuracyGuarantee": False, "productionApproved": False,
      **({"classifierArchitecture": "2048-32-ReLU-2", "headType": model["headType"]}
         if model.get("format") == 2 and model.get("headType") == "mlp32" else {}),
    }
  if model.get("architecture") in {
    "normalized-bioclip2-linear-v1", "normalized-plantnet-resnet50-linear-v1",
    "plantnet-resnet50-retained-plus-new-v1",
  }:
    classes = model["classes"]
    if len(classes) != model["classCount"] or not classes:
      raise ValueError("Species model class count does not match its label set.")
    plants = [{
      "id": row["speciesId"], "scientificName": row["scientificName"],
      "chineseName": row.get("chineseName"), "chineseAliases": row.get("chineseAliases", []),
      "classIndex": index, "catalogSpeciesId": row["speciesId"],
      **({"classRole": row["classRole"]}
         if model["architecture"] == "plantnet-resnet50-retained-plus-new-v1" else {}),
    } for index, row in enumerate(classes)]
    if len({row["id"] for row in plants}) != len(plants):
      raise ValueError("Duplicate species model output IDs.")
    return {
      "schemaVersion": 1, "task": "species-identification",
      "artifactSha256": sha256(path), "modelFile": path.name,
      "weightsSha256": model["weightsSha256"], "encoder": model["encoder"],
      "plantSet": {"kind": "finite", "unit": "catalog-species", "count": len(plants),
                   "plants": plants, "sha256": digest(plants)},
      "accuracyGuarantee": False, "productionApproved": False,
      **{key: model[key] for key in ("dataPolicy", "trainingScope", "dataLimitations")
         if key in model},
      **({"presenceGate": model["presenceGate"],
          "requestedClassCount": model["requestedClassCount"],
          "trainingMode": model["trainingMode"]}
         if model.get("architecture") in {
           "normalized-plantnet-resnet50-linear-v1", "plantnet-resnet50-retained-plus-new-v1",
         } else {}),
      **({key: model[key] for key in (
        "retainedClassCount", "trainedAdditionalClassCount", "retainedSourceIndices",
        "retainedParametersUnchanged", "retentionAccuracyGuaranteed",
      )} if model["architecture"] == "plantnet-resnet50-retained-plus-new-v1" else {}),
      **({"retainedReplay": model["retainedReplay"]} if "retainedReplay" in model else {}),
    }
  raise ValueError("Unrecognized trained model metadata; declare its plant-set semantics.")


def write_artifact_metadata(path):
  result = artifact_metadata(path)
  destination = path.with_suffix(".metadata.json")
  if destination.exists():
    if read_json(destination) != result:
      raise ValueError("Artifact sidecar disagrees with the model; do not overwrite it.")
  else:
    immutable(destination, result)
  return result


class Exporter:
  def __init__(self, directory, catalog):
    self.directory, self.catalog = directory, catalog
    self.index = {}
    self.catalog_rows, self.reference = {}, []
    self.by_name = defaultdict(list)
    self.catalog_version = None
    if catalog.available:
      with catalog.connect() as db:
        db.execute("BEGIN")  # One consistent read-only catalog snapshot.
        self.catalog_version = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
        for row in db.execute("SELECT id,scientific_name,is_visible,data FROM species ORDER BY id"):
          data = json.loads(row["data"])
          item = {
            "id": row["id"], "catalogSpeciesId": row["id"],
            "scientificName": row["scientific_name"], "chineseName": data.get("chineseName"),
            "chineseAliases": data.get("chineseAliases", []), "visible": bool(row["is_visible"]),
          }
          self.catalog_rows[item["id"]] = item
          self.by_name[item["scientificName"]].append(item)
        indices = defaultdict(list)
        for row in db.execute(
          "SELECT idx,species_id FROM labels WHERE species_id IS NOT NULL ORDER BY idx",
        ):
          if row["species_id"] not in self.catalog_rows:
            raise ValueError("Reference label points to an absent catalog species.")
          indices[row["species_id"]].append(row["idx"])
        self.reference = [
          {**self.catalog_rows[key], "referenceIndices": values}
          for key, values in sorted(indices.items())
        ]

  def plant_set(self, plants, *, unit="catalog-species", reason=None):
    if plants is None:
      return {"kind": "unknown", "count": None, "file": None, "reason": reason}
    members = sorted(plants, key=lambda row: row["id"])
    if len({row["id"] for row in members}) != len(members):
      raise ValueError("Duplicate plant-set member identity.")
    document = {"schemaVersion": 1, "unit": unit, "plants": members}
    key = digest(document)
    relative = Path("sets") / f"{key}.json"
    immutable(self.directory / relative, document)
    return {"kind": "finite", "unit": unit, "count": len(members),
            "sha256": key, "file": str(relative)}

  def add(self, identifier, title, task, plants, *, source, runtime=False,
          artifact=None, unit="catalog-species", stages=None, note=None, declared_support=None):
    if not IDENTIFIER.fullmatch(identifier) or identifier in self.index:
      raise ValueError("Invalid or duplicate solution metadata ID.")
    if task == "plant-presence":
      plant_set = not_applicable()
      app_set = not_applicable()
    else:
      plant_set = self.plant_set(plants, unit=unit, reason="Required source labels unavailable.")
      if not self.catalog.available or plants is None:
        app_set = self.plant_set(None, reason="Catalog mapping or source labels unavailable.")
      else:
        visible = {
          row["catalogSpeciesId"]: self.catalog_rows[row["catalogSpeciesId"]]
          for row in plants
          if row.get("catalogSpeciesId") in self.catalog_rows
          and self.catalog_rows[row["catalogSpeciesId"]]["visible"]
        }
        app_set = self.plant_set(list(visible.values()))
    body = {
      "schemaVersion": 1, "id": identifier, "title": title, "task": task,
      "serverSelectable": runtime, "plantSet": plant_set, "appPlantSet": app_set,
      **({
        "supportedPlantSet": self.plant_set(declared_support),
        "supportUpdatePolicy": "Only actually trained output classes; never planned additions.",
      } if declared_support is not None else {}),
      "source": source, "stages": stages or [], "artifact": artifact,
      "catalogVersion": self.catalog_version,
      "accuracyGuarantee": False, "notes": [
        "Output-label eligibility is not a per-species accuracy guarantee.",
        "App plant set is the visible mapped subset, not a guarantee of per-image acceptance.",
        *([note] if note else []),
      ],
    }
    revision = digest(body)
    relative = Path("solutions") / identifier / f"{revision}.json"
    immutable(self.directory / relative, {**body, "revision": revision})
    self.index[identifier] = {
      "file": str(relative), "revision": revision, "task": task,
      "plantCount": plant_set["count"], "appPlantCount": app_set["count"],
      "serverSelectable": runtime,
      **({"supportedPlantCount": len(declared_support)} if declared_support is not None else {}),
    }
    return body

  def mapped_classes(self, prefix, pairs):
    names = [scientific_binomial(name) for _, name in pairs]
    counts = Counter(names)
    output = []
    for index, ((source_id, raw), name) in enumerate(zip(pairs, names, strict=True)):
      matches = self.by_name[name] if name and counts[name] == 1 else []
      match = matches[0] if len(matches) == 1 else {}
      output.append({
        "id": f"{prefix}:{source_id}", "sourceSpeciesId": source_id, "classIndex": index,
        "sourceScientificName": raw, "scientificName": name or raw,
        "catalogSpeciesId": match.get("id"), "chineseName": match.get("chineseName"),
        "chineseAliases": match.get("chineseAliases", []), "visible": match.get("visible", False),
      })
    return output

  def plantclef(self, server):
    directory = server / ".cache/models/plantclef-2024/prepared"
    fallback = server / ".cache/research/plantclef-coverage-20260921"
    paths = {
      name: directory / name if (directory / name).is_file()
      else fallback / f"tar-pretrained_models__{name}"
      for name in ("class_mapping.txt", "species_id_to_name.txt")
    }
    source = {"name": "PlantCLEF 2024", "record": "10.5281/zenodo.10848263",
              "declaredClassCount": 7806, "labelHashes": {}}
    plants = None
    if all(path.is_file() for path in paths.values()):
      for name, path in paths.items():
        _, size, expected = MEMBERS[name]
        if path.stat().st_size != size or sha256(path) != expected:
          raise ValueError("PlantCLEF metadata is not the pinned official class map.")
        source["labelHashes"][name] = expected
      ids = paths["class_mapping.txt"].read_text().splitlines()
      with paths["species_id_to_name.txt"].open() as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
      names = {row["species_id"]: row["species"] for row in rows}
      if len(ids) != 7806 or len(set(ids)) != 7806 or len(names) != len(rows):
        raise ValueError("Invalid PlantCLEF class mapping.")
      plants = self.mapped_classes("plantclef", [(key, names[key]) for key in ids])
    artifact = {"weightsPresent": (directory / "model.safetensors").is_file(),
                "weightsIntegrityCheckedThisExport": False}
    result = self.add(
      "plantclef-2024", "PlantCLEF 2024 DINOv2", "species-identification", plants,
      source=source, artifact=artifact, unit="source-output-class",
      note="Published labels can be known while the prepared weights are still unavailable.",
    )
    return plants, result

  def export(self, server):
    fallback, fallback_meta = self.plantclef(server)
    recall = self.reference if self.catalog.available else None
    recall_source = {"name": "BioCLIP v1 published reference",
                     "catalog": self.catalog_version,
                     "referenceAssetSha256": {name: info[1] for name, info in ASSETS.items()}}
    recall_set = self.plant_set(recall, reason="Reference catalog unavailable.")
    for solution in SOLUTIONS.values():
      if solution.plant_set_source == "trained-plantnet":
        path = server / ".artifacts/models" / solution.model_directory / "model.json"
        plants = None
        source = {"expectedModelSha256": solution.model_sha256, "solutionVersion": solution.version}
        artifact = {"modelPresent": path.is_file(), "file": str(path.relative_to(server))}
        stages = []
        if path.is_file():
          if sha256(path) != solution.model_sha256:
            raise ValueError("Registered Pl@ntNet model metadata checksum mismatch.")
          sidecar = write_artifact_metadata(path)
          if sha256(path.parent / "classifier.safetensors") != sidecar["weightsSha256"]:
            raise ValueError("Registered Pl@ntNet trained head checksum mismatch.")
          plants = sidecar["plantSet"]["plants"]
          source.update(sidecar)
          stages = [
            {"role": "plant-presence", "plantSet": not_applicable(),
             "source": sidecar["presenceGate"]},
            {"role": "species-classification", "plantSet": self.plant_set(plants),
             "weightsSha256": sidecar["weightsSha256"]},
          ]
        supported = [
          self.catalog_rows[row["catalogSpeciesId"]] for row in plants
          if row["catalogSpeciesId"] in self.catalog_rows
          and self.catalog_rows[row["catalogSpeciesId"]]["visible"]
        ] if plants is not None and self.catalog.available else None
        self.add(
          solution.id, solution.title, "species-identification", plants,
          source=source, artifact=artifact, stages=stages, runtime=True,
          declared_support=supported,
          note="Pinned trained binary + species cascade; limited scope, no accuracy guarantee.",
        )
        continue
      if solution.plant_set_source == "trained-bioclip":
        directory = server / ".artifacts/models" / solution.model_directory
        path = directory / "production.metadata.json"
        plants = None
        source = {
          "solutionVersion": solution.version,
          "expectedProductionMetadataSha256": solution.model_sha256,
        }
        artifact = {"metadataPresent": path.is_file(), "file": str(path.relative_to(server))}
        stages = []
        if path.is_file():
          if sha256(path) != solution.model_sha256:
            raise ValueError("Registered target BioCLIP metadata checksum mismatch.")
          metadata = read_json(path, 8 * 1024 * 1024)
          if (
            metadata.get("status") != "validated-independent"
            or metadata.get("productionApproved") is not True
            or metadata.get("finalIndependentGoalMet") is not True
          ):
            raise ValueError("Registered target BioCLIP model is not approved.")
          plants = metadata["plantSet"]["plants"]
          source.update(metadata)
          stages = [
            {
              "role": "plant-presence", "plantSet": not_applicable(),
              "source": metadata["presenceGate"],
            },
            {
              "role": "species-classification",
              "plantSet": self.plant_set(plants),
              "weightsSha256": metadata["classifier"]["sha256"],
            },
          ]
        supported = [
          self.catalog_rows[row["catalogSpeciesId"]]
          for row in plants
          if row["catalogSpeciesId"] in self.catalog_rows
          and self.catalog_rows[row["catalogSpeciesId"]]["visible"]
        ] if plants is not None and self.catalog.available else None
        self.add(
          solution.id, solution.title, "species-identification", plants,
          source=source, artifact=artifact, stages=stages, runtime=True,
          declared_support=supported,
          note="Independent-validated binary gate plus finite BioCLIP 2 classifier.",
        )
        continue
      if solution.plant_set_source not in {"catalog-reference", "catalog-reference+plantclef"}:
        raise ValueError("Registered solution needs an explicit plant-set implementation.")
      if solution.cascade != (solution.plant_set_source == "catalog-reference+plantclef"):
        raise ValueError("Cascade stages do not match the declared plant-set source.")
      plants = recall
      stages = [{
        "role": "recall", "plantSet": recall_set,
        "model": MODEL_ID, "revision": MODEL_REVISION, "weightsSha256": MODEL_SHA256,
      }]
      if solution.rerank_view:
        stages.append({"role": "rerank", "plantSet": recall_set,
                       "model": RERANK_MODEL_ID, "revision": RERANK_REVISION,
                       "weightsSha256": RERANK_SHA256,
                       "shortlistLimit": solution.shortlist_limit,
                       "note": "Reranking cannot add species outside the recalled reference set."})
      if solution.cascade:
        stages.append({"role": "fallback", "metadataId": "plantclef-2024",
                       "plantSet": fallback_meta["plantSet"], "artifact": fallback_meta["artifact"]})
        if recall is not None and fallback is not None:
          union = {row["id"]: row for row in recall}
          for row in fallback:
            key = row.get("catalogSpeciesId") or row["id"]
            union.setdefault(key, {**row, "id": key})
          plants = list(union.values())
        else:
          plants = None  # Do not pretend the known primary stage is the full union.
      self.add(
        solution.id, solution.title, "species-identification", plants,
        source={**recall_source, "solutionVersion": solution.version},
        runtime=True, stages=stages,
        note="Configured capability, not a statement that this solution is active or its weights ready.",
      )
    label_version, labels = load_labels()
    probe = [label for label in labels if label.kind == "plant"]
    self.add(
      "bioclip-label-probe", "BioCLIP configured-label probe", "species-identification",
      self.mapped_classes("bioclip", [(label.id, label.name) for label in probe]),
      source={"model": MODEL_ID, "revision": MODEL_REVISION, "labelVersion": label_version,
              "nonPlantLabelCount": sum(label.kind == "nonplant" for label in labels)},
      unit="configured-label",
      note="Default probe labels only; supplying a different --labels file defines a different set.",
    )
    pretrained = server / ".cache/models/plantnet-300k-pretrained"
    for architecture in ("squeezenet1_0", "resnet50"):
      manifest_path = pretrained / f"{architecture}.manifest.json"
      plants, source = None, {"name": "Pl@ntNet-300K", "architecture": architecture,
                              "datasetVariant": "official-1081-class-pretrained-not-V2"}
      artifact = {"manifestPresent": manifest_path.is_file()}
      declared_support = None
      if manifest_path.is_file():
        manifest = read_json(manifest_path, 32768)
        if (
          manifest.get("format") != 1 or manifest["architecture"] != architecture
          or manifest["classCount"] != 1081
          or manifest.get("datasetVariant") != "official-1081-class-pretrained-not-V2"
        ):
          raise ValueError("Unexpected Pl@ntNet manifest.")
        for name in ("class_idx_to_species_id.json", "plantnet300K_species_id_2_name.json"):
          if sha256(pretrained / name) != manifest["files"][name]["sha256"]:
            raise ValueError("Pl@ntNet class map checksum mismatch.")
        source["manifestSha256"] = sha256(manifest_path)
        source["files"] = manifest["files"]
        plants = self.mapped_classes("plantnet", read_class_map(pretrained))
        artifact["weightsIntegrityCheckedThisExport"] = False
      contract_path = self.directory / "contracts" / f"plantnet-300k-{architecture}.json"
      if contract_path.exists():
        contract = read_json(contract_path, 2 * 1024 * 1024)
        if (
          not manifest_path.is_file() or contract.get("schemaVersion") != 1
          or contract.get("status") != "pretrained"
          or contract.get("solutionId") != f"plantnet-300k-{architecture}"
          or contract.get("sourceFiles") != source["files"]
        ):
          raise ValueError("Declared support is not bound to this pretrained model.")
        declared_support = []
        outputs = {row["classIndex"]: row for row in plants}
        for row in contract["supportedSpecies"]:
          if not row.get("sourceClasses") or any(
            ref["classIndex"] not in outputs
            or outputs[ref["classIndex"]]["catalogSpeciesId"] != row["speciesId"]
            or outputs[ref["classIndex"]]["sourceSpeciesId"] != ref["sourceSpeciesId"]
            or outputs[ref["classIndex"]]["sourceScientificName"] != ref["sourceScientificName"]
            for ref in row["sourceClasses"]
          ):
            raise ValueError("Declared species is not a mapped trained output class.")
          declared_support.append({
            "id": row["speciesId"], "catalogSpeciesId": row["speciesId"],
            "scientificName": row["scientificName"], "chineseName": row["chineseName"],
            "chineseAliases": row.get("chineseAliases", []),
            "sourceClasses": row["sourceClasses"],
          })
        source["supportContractSha256"] = sha256(contract_path)
      self.add(
        f"plantnet-300k-{architecture}", f"Pl@ntNet {architecture}", "species-identification",
        plants, source=source, artifact=artifact, unit="source-output-class",
        note="Keep source class IDs/indices and unmapped names; no invented synonym bridging.",
        declared_support=declared_support,
      )
    # Discover saved local models; do not rename or mutate historical head/checkpoint files.
    paths = sorted((server / ".cache/models/plant-presence").rglob("*.json"))
    for path in paths:
      if path.name.endswith(".metadata.json"):
        continue
      model = read_json(path, 8 * 1024 * 1024)
      if model.get("format") not in {1, 2} or model.get("task") != "plant-presence":
        continue
      if model["format"] == 2 and model.get("headType") != "mlp32":
        raise ValueError("Unknown binary head architecture in metadata export.")
      sidecar = write_artifact_metadata(path)
      relative = path.relative_to(server / ".cache/models/plant-presence").with_suffix("")
      identifier = "plant-presence-" + "--".join(relative.parts)
      self.add(identifier, f"Plant presence: {relative}", "plant-presence", None,
               source=sidecar, artifact={"file": str(path.relative_to(server)),
                                         "sha256": sha256(path)})
    for path in sorted((server / ".artifacts/models").glob("*/model.json")):
      architecture = read_json(path).get("architecture")
      if architecture not in {
        "normalized-bioclip2-linear-v1",
        "normalized-plantnet-resnet50-linear-v1",
        "plantnet-resnet50-retained-plus-new-v1",
      }:
        # Historical scratch/transfer experiments have their own ledgers and do
        # not declare the production plant-set contract required by this index.
        continue
      sidecar = write_artifact_metadata(path)
      transfer = architecture == "normalized-plantnet-resnet50-linear-v1"
      incremental = architecture == "plantnet-resnet50-retained-plus-new-v1"
      prefix = (
        "plantnet-resnet50-incremental" if incremental
        else "plantnet-resnet50-linear" if transfer else "bioclip2-linear"
      )
      title = (
        "Pl@ntNet retained + new" if incremental
        else "Pl@ntNet ResNet50 linear" if transfer else "BioCLIP 2 linear"
      )
      self.add(
        f"{prefix}-{path.parent.name}", f"{title}: {path.parent.name}",
        "species-identification", sidecar["plantSet"]["plants"],
        source={"metadataSha256": sha256(path), "encoder": sidecar["encoder"],
                **{key: sidecar[key] for key in (
                  "dataPolicy", "trainingScope", "dataLimitations", "presenceGate",
                  "requestedClassCount", "trainingMode",
                  "retainedClassCount", "trainedAdditionalClassCount", "retainedSourceIndices",
                  "retainedParametersUnchanged", "retentionAccuracyGuaranteed",
                  "retainedReplay",
                )
                   if key in sidecar}},
        artifact={"file": str(path.relative_to(server)), "sha256": sha256(path)},
      )
    index = {"schemaVersion": 1, "generatedAt": datetime.now(UTC).isoformat(),
             "solutions": self.index}
    write_json(self.directory / "index.json", index)
    return index


class MetadataStore:
  def __init__(self, directory=None):
    self.directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY

  def _path(self, relative):
    path = (self.directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(self.directory.resolve()):
      raise ValueError("Metadata path escapes its snapshot directory.")
    return path

  def index(self):
    return read_json(self.directory / "index.json", 1024 * 1024)

  def get(self, identifier):
    reference = self.index()["solutions"].get(identifier)
    if reference is None:
      raise ValueError("Unknown solution metadata ID; refresh snapshots after registering models.")
    result = read_json(self._path(reference["file"]))
    body = {key: value for key, value in result.items() if key != "revision"}
    if result["id"] != identifier or result["revision"] != reference["revision"] \
        or digest(body) != result["revision"]:
      raise ValueError("Solution metadata revision mismatch.")
    return result

  def plants(self, identifier, *, scope="model", q="", offset=0, limit=30):
    if scope not in {"model", "app", "supported"} or not 0 <= offset <= 500000 or not 1 <= limit <= 100:
      raise ValueError("Invalid metadata plant-set pagination.")
    if not isinstance(q, str) or len(q) > 160:
      raise ValueError("Invalid plant query.")
    metadata = self.get(identifier)
    reference = (
      metadata.get("supportedPlantSet", metadata["appPlantSet"])
      if scope == "supported" else metadata["plantSet" if scope == "model" else "appPlantSet"]
    )
    if reference["kind"] != "finite":
      return {"kind": reference["kind"], "total": None, "items": [],
              "reason": reference.get("reason")}
    path = self._path(reference["file"])
    if sha256(path) != reference["sha256"]:
      raise ValueError("Plant-set content hash mismatch.")
    members = read_json(path)["plants"]
    if len(members) != reference["count"]:
      raise ValueError("Plant-set count mismatch.")
    query = q.strip().casefold()
    if query:
      members = [row for row in members if query in " ".join(filter(None, [
        row["id"], row.get("scientificName"), row.get("sourceScientificName"),
        row.get("chineseName"), *(row.get("chineseAliases") or []),
      ])).casefold()]
    return {"kind": "finite", "total": len(members), "items": members[offset:offset + limit],
            "offset": offset, "limit": limit, "revision": metadata["revision"]}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=(
    "refresh", "list", "show", "plants", "initialize-plantnet-support",
  ))
  parser.add_argument("--solution")
  parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
  parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
  parser.add_argument("--scope", choices=("model", "app", "supported"), default="model")
  parser.add_argument("--plant-sets", type=Path,
                      default=SERVER.parent / "docs/research/plantnet-plant-sets-20260923/sets.json")
  parser.add_argument("--q", default="")
  parser.add_argument("--offset", type=int, default=0)
  parser.add_argument("--limit", type=int, default=30)
  args = parser.parse_args()
  store = MetadataStore(args.directory)
  if args.action == "initialize-plantnet-support":
    from yizhu_server.plantnet_plant_sets import register_pretrained_support

    result = register_pretrained_support(args.directory, args.plant_sets)
  elif args.action == "refresh":
    result = Exporter(args.directory, PlantCatalog(args.catalog)).export(SERVER)
  elif args.action == "list":
    result = store.index()
  elif not args.solution:
    parser.error("--solution is required.")
  elif args.action == "show":
    result = store.get(args.solution)
  else:
    result = store.plants(args.solution, scope=args.scope, q=args.q,
                          offset=args.offset, limit=args.limit)
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
