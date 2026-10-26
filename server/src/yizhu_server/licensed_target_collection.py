"""Collect a reviewed fixed species plan from licensed iNaturalist observations."""

import argparse
import hashlib
import json
from pathlib import Path

from yizhu_server.training_data import PublicClient, collect, dataset_lock, now, write_json

POLICY = "reviewed-species-inat-cc0-ccby-v1"


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    species = plan.get("species")
    targets = plan.get("splitTargets")
    if (
        plan.get("policy") != POLICY
        or not isinstance(species, list) or not species
        or not isinstance(targets, dict) or set(targets) != {"train", "validation", "test"}
        or any(type(value) is not int or not 1 <= value <= 100 for value in targets.values())
        or any(
            not row.get("speciesId") or not row.get("scientificName")
            or type(row.get("inatTaxonId")) is not int
            or row.get("taxonRank") != "species"
            for row in species
        )
        or len({row["speciesId"] for row in species}) != len(species)
    ):
        raise ValueError("Require a distinct, reviewed species-level collection plan.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "species.json", plan)
    try:
        with dataset_lock(output):
            result = collect(
                output, PublicClient(), plan, class_limit=len(species),
                split_targets=targets, max_pages=plan.get("maxPages", 30),
            )
        write_json(output / "result.json", {
            **result, "planSha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
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
