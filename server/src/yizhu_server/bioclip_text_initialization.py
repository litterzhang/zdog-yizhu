"""Build frozen BioCLIP 2 scientific-name text prototypes for a reviewed target list."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from yizhu_server.recognition_rerank import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    TEXT_TEMPLATES,
    BioClipReranker,
)
from yizhu_server.training_data import now, sha256, write_json


def build(classes_path, output, *, threads=4):
    import torch
    from safetensors.torch import save_file

    if output.exists():
        raise ValueError("Use a new text-initialization directory.")
    document = json.loads(classes_path.read_text())
    classes = document["classes"]
    if len(classes) < 2 or len({row["speciesId"] for row in classes}) != len(classes):
        raise ValueError("Require a reviewed target list with unique species IDs.")
    encoder = BioClipReranker(threads=threads)
    output.mkdir(parents=True)
    started = time.monotonic()
    vectors = []
    with torch.inference_mode():
        for index, row in enumerate(classes, 1):
            vectors.append(encoder.text_features(row["scientificName"]))
            if index % 20 == 0 or index == len(classes):
                write_json(output / "progress.json", {
                    "updatedAt": now(), "stage": "encoding",
                    "completed": index, "total": len(classes),
                    "elapsedSeconds": time.monotonic() - started,
                    "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                })
    weight = torch.stack(vectors).contiguous()
    if weight.shape != (len(classes), 768) or not torch.isfinite(weight).all():
        raise ValueError("Invalid BioCLIP text prototype matrix.")
    temporary = output / "classifier.safetensors.tmp"
    save_file({
        "weight": weight,
        "bias": torch.zeros(len(classes), dtype=torch.float32),
    }, temporary)
    os.replace(temporary, output / "classifier.safetensors")
    write_json(output / "index.json", {
        "createdAt": now(), "status": "text-prototypes-extracted",
        "architecture": MODEL_ID, "revision": MODEL_REVISION,
        "checkpointSha256": MODEL_SHA256, "dimension": 768,
        "templates": list(TEXT_TEMPLATES),
        "classes": [{
            "classIndex": index,
            **{key: row[key] for key in ("speciesId", "scientificName", "chineseName")},
        } for index, row in enumerate(classes)],
        "classesManifestSha256": sha256(classes_path),
        "weightsSha256": sha256(output / "classifier.safetensors"),
        "productionChanged": False,
    })
    write_json(output / "progress.json", {
        "updatedAt": now(), "stage": "complete",
        "completed": len(classes), "total": len(classes),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    build(args.classes.resolve(), args.output.resolve(), threads=args.threads)


if __name__ == "__main__":
    main()
