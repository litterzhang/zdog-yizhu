"""Run the two frozen-backbone trials; select on validation before opening test results."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from yizhu_server.plant_presence import BACKBONES, evaluate, train
from yizhu_server.plantnet_pretrained import sha256, write_report
from yizhu_server.training_data import dataset_lock, require_dataset_active, write_json


def choose_on_validation(heads, minimum_recall=0.95):
  eligible = [name for name, head in heads.items()
              if head["validation"]["plantRecall"] >= minimum_recall]
  if not eligible:
    raise ValueError("No candidate meets the validation recall target.")
  return min(eligible, key=lambda name: (
    heads[name]["validation"]["falsePositiveRate"],
    -heads[name]["validation"]["plantRecall"], name,
  ))


def run(data, models, reports, *, threads=2, progress=None):
  require_dataset_active(data)
  reports.mkdir(parents=True, exist_ok=True)
  models.mkdir(parents=True, exist_ok=True)
  with dataset_lock(reports), dataset_lock(data):
    plan = {
      "datasetSha256": sha256(data / "samples.json"), "backbones": list(BACKBONES),
      "epochs": 100, "minimumValidationPlantRecall": 0.95, "threads": threads,
      "selectionRule": "validation recall >= 95%, then minimum FPR; no test-based selection",
      "trainingScope": "frozen encoders and identical linear head recipe; not fine-tuning",
    }
    plan_path = reports / "plan.json"
    if plan_path.exists():
      if json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing comparison plan differs; use a new experiment directory.")
    else:
      write_report(plan_path, plan)
    heads = {}
    for name in BACKBONES:
      path = models / f"{name}-resnet50-linear.json"
      if not path.exists():
        print(f"Training {name} frozen backbone + linear head", flush=True)
        if progress:
          progress({"stage": "training", "backbone": name, "phase": "checking-data"})
        started = time.monotonic()
        train(
          data, path, backbone=name, threads=threads,
          progress=(lambda details: progress({
            "stage": "training", "backbone": name, **details,
          })) if progress else None,
        )
        print(f"Finished {name} in {time.monotonic() - started:.1f}s", flush=True)
      head = json.loads(path.read_text())
      if (
        head.get("backbone") != name or head.get("datasetSha256") != plan["datasetSha256"]
        or head.get("minimumValidationPlantRecall") != plan["minimumValidationPlantRecall"]
      ):
        raise ValueError("Cached head does not match the frozen comparison plan.")
      heads[name] = head
    chosen = choose_on_validation(heads)
    decision = {
      "chosenBackbone": chosen, "selectionRule": plan["selectionRule"],
      "datasetSha256": plan["datasetSha256"],
      "heads": {name: {
        "sha256": sha256(models / f"{name}-resnet50-linear.json"),
        "validation": head["validation"], "threshold": head["threshold"],
      } for name, head in heads.items()},
      "testUsedForSelection": False, "productionApproved": False,
    }
    decision_path = reports / "validation-selection.json"
    if decision_path.exists():
      if json.loads(decision_path.read_text()) != decision:
        raise ValueError("Frozen pre-test selection changed; refusing to reinterpret test results.")
    else:
      # This file must exist BEFORE either test set is scored.
      write_report(decision_path, decision)
    print(f"Validation-only selection frozen: {chosen}. Starting held-out tests.", flush=True)
    tests = {}
    for name in BACKBONES:
      path = reports / f"{name}-test.json"
      head_path = models / f"{name}-resnet50-linear.json"
      if not path.exists():
        if progress:
          progress({"stage": "evaluating", "backbone": name})
        evaluate(
          data, head_path, path, threads=threads,
          progress=(lambda details: progress({
            "stage": "evaluating", "backbone": name, **details,
          })) if progress else None,
        )
      result = json.loads(path.read_text())
      if (
        result["datasetSha256"] != plan["datasetSha256"]
        or result["modelVersion"] != decision["heads"][name]["sha256"]
      ):
        raise ValueError("Cached test report does not match this experiment.")
      tests[name] = result["metrics"]
    summary = {
      **decision, "testMetrics": tests,
      "note": "Validation selected the candidate; test results are reported without reselection.",
      "productionChanged": False,
    }
    output = reports / "summary.json"
    if output.exists():
      if json.loads(output.read_text()) != summary:
        raise ValueError("Existing final summary differs.")
    else:
      write_report(output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_pipeline(data, models, reports, *, threads=2, prepare_data=False):
  """No polling queue: download failure stops this attempt before training starts."""
  def progress(details):
    write_json(reports / "run-status.json", {
      **details, "updatedAt": datetime.now(UTC).isoformat(), "productionChanged": False,
    })

  try:
    if prepare_data and not (data / "samples.json").exists():
      from yizhu_server.plant_presence_data import collect

      if not (data / "selection.json").exists():
        raise ValueError("--prepare-data requires an existing frozen selection.json.")
      progress({"stage": "preparing-data"})
      collect(
        data, workers=4,
        progress=lambda details: progress({"stage": "preparing-data", **details}),
      )
    result = run(data, models, reports, threads=threads, progress=progress)
    progress({"stage": "complete", "chosenBackbone": result["chosenBackbone"],
              "testMetrics": result["testMetrics"]})
    return result
  except Exception as exc:
    progress({"stage": "failed", "errorType": type(exc).__name__, "error": str(exc)})
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True)
  parser.add_argument("--reports", type=Path, required=True)
  parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
  parser.add_argument("--prepare-data", action="store_true",
                      help="Resume the frozen image selection, then train; no waiting queue")
  args = parser.parse_args()
  run_pipeline(
    args.data, args.models, args.reports, threads=args.threads, prepare_data=args.prepare_data,
  )


if __name__ == "__main__":
  main()
