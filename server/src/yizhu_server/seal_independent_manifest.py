"""Copy a selected independent manifest into a portable immutable image bundle."""

import argparse
import json
import shutil
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json


def seal(source_path, output):
    if output.exists():
        raise ValueError("Use a new sealed independent directory.")
    source = json.loads(source_path.read_text())
    if (
        source.get("status") != "complete"
        or source.get("purpose") != "new-independent-test-never-used-for-training-or-selection"
        or source.get("imagesVerified") is not True
        or source.get("modelOutputsReadDuringSelection") is not False
    ):
        raise ValueError("Independent selection manifest is incomplete.")
    output.mkdir(parents=True)
    images = output / "images"
    images.mkdir()
    rows = []
    for row in source["samples"]:
        image = Path(row["path"])
        if not image.is_file() or sha256(image) != row["imageSha256"]:
            raise ValueError("Selected independent image changed.")
        destination = images / f"{row['imageSha256']}.jpg"
        if not destination.exists():
            shutil.copyfile(image, destination)
        if sha256(destination) != row["imageSha256"]:
            raise ValueError("Sealed independent image differs.")
        rows.append({**row, "path": f"images/{destination.name}"})
    sealed = {
        **source, "createdAt": now(), "samples": rows,
        "sealedFromManifestSha256": sha256(source_path),
        "portableRelativePaths": True,
    }
    write_json(output / "manifest.json", sealed)
    write_json(output / "audit.json", {
        "createdAt": now(), "sourceManifestSha256": sha256(source_path),
        "sealedManifestSha256": sha256(output / "manifest.json"),
        "images": len(rows), "representedSpecies": len({r["speciesId"] for r in rows}),
        "allImagesVerified": True, "modelOutputsRead": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
