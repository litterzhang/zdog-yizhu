"""Collect licensed research-grade photos for an explicit resolved species plan."""

import argparse
import json
from pathlib import Path

from yizhu_server.training_data import (
  PublicClient,
  collect,
  dataset_lock,
  now,
  write_json,
)


def run(plan_path, output):
  plan = json.loads(plan_path.read_text())
  species = plan.get("species")
  targets = plan.get("splitTargets")
  if (
    plan.get("policy") != "target162-missing-research-cc0-ccby-v1"
    or not isinstance(species, list) or not species
    or targets != {"train": 70, "validation": 15, "test": 15}
    or any(
      not r.get("speciesId") or not r.get("scientificName")
      or type(r.get("inatTaxonId")) is not int
      for r in species
    )
  ):
    raise ValueError("Require a resolved fixed target collection plan.")
  output.mkdir(parents=True, exist_ok=False)
  write_json(output / "species.json", plan)
  try:
    with dataset_lock(output):
      result = collect(
        output, PublicClient(), plan, class_limit=len(species),
        split_targets=targets, max_pages=20,
      )
    write_json(output / "result.json", {
      **result, "planSha256": __import__("hashlib").sha256(plan_path.read_bytes()).hexdigest(),
      "automaticTraining": False, "onlineChanged": False,
    })
  except BaseException as exc:
    write_json(output / "stopped.json", {
      "stoppedAt": now(), "errorType": type(exc).__name__,
      "automaticRetry": False, "onlineChanged": False,
    })
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  run(args.plan.resolve(), args.output.resolve())


if __name__ == "__main__":
  main()
