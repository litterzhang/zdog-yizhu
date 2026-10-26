"""Image validation, pinned model constants, and the optional small-label probe.

The live full-reference provider is in catalog_recognition.py. This older custom
label provider remains useful for comparisons; neither supplies calibrated confidence.
"""

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict

MODEL_ID = "imageomics/bioclip"
MODEL_REVISION = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c"
MODEL_FILENAME = "open_clip_pytorch_model.bin"
MODEL_BYTES = 598_599_013
MODEL_SHA256 = "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2] / ".cache/models/bioclip-v1" / MODEL_FILENAME
)
DEFAULT_LABELS = Path(__file__).parent / "data/bioclip-labels.json"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True)
class Label:
    id: str
    kind: Literal["plant", "nonplant"]
    name: str


class Candidate(TypedDict):
    label_id: NotRequired[str]
    species_id: NotRequired[str | None]
    kind: Literal["plant", "nonplant"]
    scientific_name: str | None
    label: str
    score: float
    confidence: None


class RecognitionResult(TypedDict):
    source: str
    model: str
    revision: str
    label_version: str
    is_demo: bool
    status: Literal["requires_review", "uncertain"]
    scope: str
    score_type: str
    candidates: list[Candidate]
    reason: NotRequired[str | None]
    catalog_plant_count: NotRequired[int]
    provider_version: NotRequired[str]
    global_leading_candidate: NotRequired[dict]


def load_labels(path: Path = DEFAULT_LABELS) -> tuple[str, list[Label]]:
    if path.stat().st_size > 256 * 1024:
        raise ValueError("Label file is too large.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or not isinstance(data.get("version"), str)
        or not data["version"].strip()
    ):
        raise ValueError("Labels must include a version.")
    rows = data.get("labels")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 100:
        raise ValueError("The CPU probe requires 2–100 labels.")
    labels = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in {"plant", "nonplant"}:
            raise ValueError("Each label needs a plant/nonplant kind.")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 160
            for key in ("id", "name")
        ):
            raise ValueError("Each label needs a short ID and name.")
        labels.append(Label(row["id"], row["kind"], row["name"].strip()))
    if len({label.id for label in labels}) != len(labels):
        raise ValueError("Duplicate label IDs.")
    if len({label.name for label in labels}) != len(labels):
        raise ValueError("Duplicate label names.")
    return data["version"], labels


def verify_weights(path: Path) -> None:
    """Verify the publisher's pinned LFS SHA before any deserialization."""
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise ValueError("Pinned BioCLIP v1 weights are missing or have the wrong size.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("BioCLIP weight checksum mismatch; refusing to load.")


def read_photo(path: Path):
    """Bound local input, apply orientation, and discard metadata before inference."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError("Photo must be a local file of at most 8 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Only JPEG, PNG and WebP are supported.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated photos are not supported.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Photo exceeds the 20-megapixel limit.")
                image.load()
                oriented = ImageOps.exif_transpose(image).convert("RGB")
                # Fresh pixel-only image: no EXIF, filename, comment or location.
                clean = Image.new("RGB", oriented.size)
                clean.paste(oriented)
                return clean
    except (
        UnidentifiedImageError, OSError, Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError("Photo cannot be decoded safely.") from None


class RecognitionProvider(Protocol):
    def recognize(self, image, *, top_k: int = 5) -> RecognitionResult: ...


def format_candidates(
    rows: list[dict], labels: list[Label], *, label_version: str
) -> RecognitionResult:
    by_name = {label.name: label for label in labels}
    candidates: list[Candidate] = []
    for row in rows:
        label = by_name.get(row.get("classification"))
        score = row.get("score")
        if (
            label is None or isinstance(score, bool) or not isinstance(score, (float, int))
            or not math.isfinite(score) or not 0 <= score <= 1
        ):
            raise ValueError("Unexpected classifier output.")
        candidates.append({
            "label_id": label.id,
            "kind": label.kind,
            "scientific_name": label.name if label.kind == "plant" else None,
            "label": label.name,
            "score": float(score),
            "confidence": None,
        })
    if not candidates:
        raise ValueError("Classifier returned no candidates.")
    # Never promote a forced choice to a confirmed species, even at a high score.
    return {
        "source": "BioCLIP",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "label_version": label_version,
        "is_demo": False,
        "status": "requires_review",
        "scope": "closed_set_baseline_only",
        "score_type": "softmax_over_supplied_labels_not_calibrated_confidence",
        "candidates": candidates,
    }


class BioClipProvider:
    def __init__(
        self, weights: Path = DEFAULT_WEIGHTS, labels_path: Path = DEFAULT_LABELS,
        *, threads: int = 2,
    ):
        if not 1 <= threads <= 4:
            raise ValueError("Use 1–4 CPU threads for the baseline.")
        self.label_version, self.labels = load_labels(labels_path)
        verify_weights(weights)
        # Optional imports remain out of API startup and ordinary server installs.
        import open_clip
        import torch
        from bioclip import CustomLabelsClassifier
        from bioclip.predict import preprocess_img

        torch.set_num_threads(threads)

        class LocalClassifier(CustomLabelsClassifier):
            def load_pretrained_model(
                self, model_str: str, pretrained_str: str | None = None,
            ):
                # Explicit local architecture/checkpoint: no HF network access and
                # no accidental default to BioCLIP 2. Eager CPU avoids JIT startup.
                self.model_str = model_str
                self.pretrained_str = pretrained_str
                self.model, _ = open_clip.create_model_from_pretrained(
                    model_str, pretrained=pretrained_str, device="cpu", return_transform=True,
                    weights_only=True,
                )
                self.model.eval()
                # Preserve pybioclip's BioCLIP-v1 square resize and normalization,
                # even though loading by architecture rather than hf-hub alias.
                self.preprocess = preprocess_img

        self.classifier = LocalClassifier(
            [label.name for label in self.labels],
            model_str="ViT-B-16", pretrained_str=str(weights.resolve()), device="cpu",
        )

    def recognize(self, image, *, top_k: int = 5) -> RecognitionResult:
        if not 1 <= top_k <= len(self.labels):
            raise ValueError("top_k must be within the label count.")
        rows = self.classifier.predict([image], k=top_k, batch_size=1, callback=lambda *_: None)
        return format_candidates(rows, self.labels, label_version=self.label_version)
