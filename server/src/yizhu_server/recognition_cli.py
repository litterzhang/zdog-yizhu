"""Local smoke benchmark, deliberately separate from the public API."""

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

from yizhu_server.catalog import PlantCatalog
from yizhu_server.catalog_recognition import CatalogRecognitionProvider
from yizhu_server.recognition import (
    DEFAULT_WEIGHTS,
    MODEL_SHA256,
    BioClipProvider,
    read_photo,
)
from yizhu_server.recognition_solutions import SOLUTIONS, create_provider


def case_image(case_id, path):
    if path is not None:
        return read_photo(path)
    import numpy as np
    from PIL import Image

    return Image.new("RGB", (224, 224), "white") if case_id.endswith("blank") else (
        Image.fromarray(np.random.default_rng(0).integers(
            0, 256, (224, 224, 3), dtype=np.uint8,
        ))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline plant-recognition probe; never switches the API")
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--labels", type=Path, help="Optional custom subset for comparison only")
    parser.add_argument("--solution", choices=tuple(SOLUTIONS),
                        help="Run a registered solution locally, without changing the API selection")
    parser.add_argument("--rerank", action="store_true",
                        help="Offline only: v1 top-50 retrieval followed by local BioCLIP 2 reranking")
    parser.add_argument("--rerank-view", choices=("official", "whole"), default="official",
                        help="Official center crop or full-image square resize; requires --rerank")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--repeat", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--include-controls", action="store_true", help="Add blank/noise negatives")
    parser.add_argument("--output", type=Path, help="Write a new JSON report; never overwrite")
    args = parser.parse_args()
    if len(args.photos) > 50:
        parser.error("At most 50 photos per local probe.")
    if args.solution and (args.labels or args.rerank or args.weights != DEFAULT_WEIGHTS):
        parser.error("--solution cannot be combined with --labels, --rerank or custom --weights.")
    if args.rerank and args.labels:
        parser.error("--rerank requires full-reference retrieval, not custom labels.")
    if args.rerank_view != "official" and not args.rerank:
        parser.error("--rerank-view requires --rerank.")

    # Fail closed if a library attempts a hub fetch; this command never uploads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    started = time.perf_counter()
    try:
        if args.output and args.output.exists():
            raise ValueError("Report already exists; choose a new output path.")
        if args.rerank:
            from yizhu_server import recognition_rerank

            recognition_rerank.verify_weights(recognition_rerank.DEFAULT_WEIGHTS)
        if args.solution:
            catalog = PlantCatalog()
            provider = create_provider(args.solution, catalog, threads=args.threads)
            label_count = catalog.metadata["labelCount"]
        else:
            provider = (BioClipProvider(args.weights, args.labels, threads=args.threads)
                        if args.labels else CatalogRecognitionProvider(weights=args.weights, threads=args.threads))
            label_count = len(provider.labels) if args.labels else provider.catalog.metadata["labelCount"]
        initialization = time.perf_counter() - started
        results = []
        cases = [(f"input-{index + 1}", path) for index, path in enumerate(args.photos)]
        if args.include_controls:
            cases.extend([("synthetic-blank", None), ("synthetic-noise", None)])
        for case_id, path in cases:
            image = case_image(case_id, path)
            timings = []
            for _ in range(args.repeat):
                before = time.perf_counter()
                prediction = (provider.retrieve(image, limit=50) if args.rerank
                              else provider.recognize(image))
                timings.append(round(time.perf_counter() - before, 4))
            image.close()
            results.append({
                "case": case_id,  # No private paths, filenames or image URLs in output.
                "inference_seconds": timings,
                "median_inference_seconds": statistics.median(timings),
                "result": prediction,
            })
        reranker_report = None
        if args.rerank:
            # Sequential loading limits memory during evaluation. It does not
            # represent the RAM cost of serving both models concurrently.
            del provider
            gc.collect()
            before = time.perf_counter()
            reranker = recognition_rerank.BioClipReranker(threads=args.threads)
            reranker_report = {
                "model": recognition_rerank.MODEL_ID, "revision": recognition_rerank.MODEL_REVISION,
                "weights_sha256": recognition_rerank.MODEL_SHA256, "shortlist_limit": 50,
                "image_view": args.rerank_view,
                "initialization_seconds": round(time.perf_counter() - before, 4),
                "execution": "sequential_offline_not_live_latency_or_memory",
            }
            for (case_id, path), entry in zip(cases, results, strict=True):
                image = case_image(case_id, path)
                retrieval = entry["result"]
                before = time.perf_counter()
                reranker.rerank(image, retrieval, view=args.rerank_view)
                entry["rerank_first_call_seconds"] = round(time.perf_counter() - before, 4)
                timings = []
                for _ in range(args.repeat):
                    before = time.perf_counter()
                    prediction = reranker.rerank(image, retrieval, view=args.rerank_view)
                    timings.append(round(time.perf_counter() - before, 4))
                image.close()
                entry["rerank_warm_seconds"] = timings
                entry["rerank_warm_median_seconds"] = statistics.median(timings)
                entry["result"] = prediction
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            "purpose": "reference candidate smoke benchmark; not a measured product accuracy",
            "device": "cpu",
            "threads": args.threads,
            "weights_sha256": MODEL_SHA256 if not args.solution else None,
            "solution": args.solution,
            "label_count": label_count,
            "versions": {name: version(name) for name in (
                "pybioclip", "open_clip_torch", "torch", "torchvision", "pillow",
            )},
            "initialization_seconds": round(initialization, 4),
            "total_seconds": round(time.perf_counter() - started, 4),
            "peak_process_rss_mib": round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 2),
            "cases": results,
        }
        if reranker_report:
            report["reranker"] = reranker_report
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                os.chmod(args.output, 0o600)
                stream.write(text + "\n")
        print(text)
    except ImportError:
        parser.exit(1, "Install the recognition extra: uv sync --extra recognition --locked\n")
    except (ValueError, OSError, RuntimeError):
        # Avoid echoing private image paths or arbitrary library payloads.
        parser.exit(1, "Probe failed: check model checksum, labels, image limits and output path.\n")


if __name__ == "__main__":
    main()
