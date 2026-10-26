"""Create immutable production metadata only after a passing independent evaluation."""

import argparse
import json
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def promote(model, evaluation):
    source_path = model / "model.metadata.json"
    destination = model / "production.metadata.json"
    if destination.exists():
        raise ValueError("Production metadata is immutable; use a new model version.")
    source = json.loads(source_path.read_text())
    result_path = evaluation / "results.json"
    result = json.loads(result_path.read_text())
    class_count = source.get("plantSet", {}).get("count")
    if (
        source.get("status") != "trained-offline-development-candidate"
        or source.get("productionApproved") is not False
        or source.get("finalIndependentGoalMet") is not False
        or result.get("status") != "final-independent-evaluation"
        or result.get("goalMet") is not True
        or result.get("evaluatedOnce") is not True
        or result.get("representedSpecies") != class_count
        or result.get("modelMetadataSha256") != sha256(source_path)
        or result.get("classifierSha256") != source["classifier"]["sha256"]
        or result.get("macroEndToEndTop1", 0) <= .8
        or result.get("macroEndToEndTop5", 0) <= .9
        or result.get("productionChanged") is not False
    ):
        raise ValueError("Independent acceptance evidence is incomplete or below target.")
    production = {
        **source,
        "createdAt": now(),
        "status": "validated-independent",
        "productionApproved": True,
        "finalIndependentGoalMet": True,
        "candidateMetadataSha256": sha256(source_path),
        "independentEvaluation": {
            "resultsFile": str(result_path),
            "resultsSha256": sha256(result_path),
            **{key: result[key] for key in (
                "samples", "representedSpecies", "top1", "top5",
                "macroTop1", "macroTop5", "plantPassRate",
                "endToEndTop1", "endToEndTop5",
                "macroEndToEndTop1", "macroEndToEndTop5", "goalMet",
            )},
        },
    }
    write_json(destination, production)
    return production


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    result = promote(args.model.resolve(), args.evaluation.resolve())
    print(json.dumps({
        "solutionId": result["solutionId"],
        "classCount": result["plantSet"]["count"],
        "status": result["status"],
        "productionApproved": result["productionApproved"],
    }))


if __name__ == "__main__":
    main()
