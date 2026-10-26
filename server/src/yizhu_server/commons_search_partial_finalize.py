"""Seal an interrupted Commons search prefix without claiming all targets completed."""

import argparse
import json
from collections import Counter
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def finalize(root, completed):
    plan = json.loads((root / "plan.json").read_text())
    species = plan["species"]
    rows = json.loads((root / "candidates.json").read_text())
    if not 0 <= completed < len(species):
        raise ValueError("Completed prefix must be smaller than the search plan.")
    for row in rows:
        image = root / row["path"]
        if (
            row.get("trainingEligible") is not False
            or not image.is_file() or sha256(image) != row["imageSha256"]
        ):
            raise ValueError("Commons partial candidate changed.")
    write_json(root / "partial.json", {
        "finalizedAt": now(), "partialScope": True,
        "completedSpecies": completed, "totalSpecies": len(species),
        "completedScientificNames": [row["scientificName"] for row in species[:completed]],
        "remainingScientificNames": [row["scientificName"] for row in species[completed:]],
        "candidateImages": len(rows),
    })
    write_json(root / "progress.json", {
        "completedAt": now(), "stage": "collection-partial-finalized",
        "partialScope": True, "completedSpecies": completed,
        "totalSpecies": len(species), "candidateImages": len(rows),
        "counts": dict(Counter(row["speciesId"] for row in rows)),
        "trainingEligible": False, "onlineChanged": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--completed", type=int, required=True)
    args = parser.parse_args()
    finalize(args.data.resolve(), args.completed)


if __name__ == "__main__":
    main()
