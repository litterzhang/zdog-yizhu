"""Seal an interrupted GBIF candidate prefix without claiming the full name list completed."""

import argparse
import json
from collections import Counter
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def finalize(root, names_path):
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text())
    names = json.loads(names_path.read_text())["scientificNames"]
    completed = progress.get("completedSpecies")
    if (
        progress.get("stage") != "collecting"
        or not isinstance(completed, int) or not 0 <= completed < len(names)
        or progress.get("totalSpecies") != len(names)
    ):
        raise ValueError("Only an interrupted in-progress GBIF collection can be finalized.")
    rows = []
    for path in sorted((root / "records").glob("*.json")):
        row = json.loads(path.read_text())
        image = root / row["path"]
        if (
            row.get("trainingEligible") is not False
            or not image.is_file() or sha256(image) != row["imageSha256"]
        ):
            raise ValueError("Partial GBIF record or image changed.")
        rows.append(row)
    if len({row["candidateId"] for row in rows}) != len(rows):
        raise ValueError("Partial GBIF records contain duplicate candidate IDs.")
    write_json(root / "candidates.json", rows)
    write_json(root / "partial.json", {
        "finalizedAt": now(), "partialScope": True,
        "completedSpecies": completed, "totalSpecies": len(names),
        "completedScientificNames": names[:completed],
        "remainingScientificNames": names[completed:],
        "originalProgress": progress,
        "candidateImages": len(rows),
        "note": "Valid prefix evidence only; remaining names are collected separately.",
    })
    write_json(progress_path, {
        "completedAt": now(), "stage": "collection-partial-finalized",
        "partialScope": True, "completedSpecies": completed,
        "totalSpecies": len(names), "candidateImages": len(rows),
        "counts": dict(Counter(row["speciesId"] for row in rows)),
        "rejected": progress.get("rejected", {}),
        "trainingEligible": False, "automaticAdmission": False,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--names-file", type=Path, required=True)
    args = parser.parse_args()
    rows = finalize(args.data.resolve(), args.names_file.resolve())
    print(json.dumps({"candidateImages": len(rows), "partialScope": True}))


if __name__ == "__main__":
    main()
