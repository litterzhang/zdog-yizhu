"""Explicitly selected local BioCLIP 2 reranking; never enabled implicitly."""

import copy
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

MODEL_ID = "imageomics/bioclip-2"
MODEL_REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"
MODEL_BYTES = 1_710_517_724
MODEL_SHA256 = "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2] / ".cache/models/bioclip-v2/open_clip_model.safetensors"
)
# Fixed photographic subset of pybioclip's templates; no species-specific prompts.
# Keep this CPU comparison bounded instead of encoding 80 prompts per candidate.
TEXT_TEMPLATES = (
    "a photo of a {name}.",
    "a close-up photo of a {name}.",
    "a photo of the {name}.",
)


def verify_weights(path):
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise ValueError("Pinned BioCLIP 2 safetensors are missing or incomplete.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("BioCLIP 2 checksum mismatch.")


def validate_candidates(candidates):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 100:
        raise ValueError("Reranking requires 1–100 retrieved candidates.")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("kind") != "plant":
            raise ValueError("Reranking only accepts retrieved plant references.")
        identifier, name = candidate.get("species_id"), candidate.get("scientific_name")
        if (not isinstance(identifier, str) or not identifier or identifier in seen
                or not isinstance(name, str) or not name.strip() or len(name) > 160):
            raise ValueError("Invalid or duplicate retrieved reference.")
        seen.add(identifier)


def reranked_result(first_stage, scores, *, view="official"):
    """No expected species input; never inject a label or turn abstention into success."""
    candidates = first_stage.get("candidates")
    validate_candidates(candidates)
    if view not in {"official", "whole"}:
        raise ValueError("Unknown reranking image view.")
    if first_stage.get("status") not in {"requires_review", "uncertain"}:
        raise ValueError("Unexpected first-stage status.")
    if len(scores) != len(candidates) or any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("Invalid reranking scores.")
    ranked = [
        {**candidate, "score": float(score), "confidence": None,
         "retrieval_rank": index + 1, "retrieval_score": candidate["score"]}
        for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
    ]
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return {
        "source": "BioCLIP", "model": MODEL_ID, "revision": MODEL_REVISION,
        "provider_version": "bioclip-v1-retrieve-v2-rerank-1", "is_demo": False,
        "scope": "retrieved_shortlist_not_full_bioclip2_reference",
        "score_type": "cosine_similarity_not_calibrated_confidence",
        "text_strategy": "scientific-name-photo-templates-v1",
        "text_template_count": len(TEXT_TEMPLATES),
        "image_view": view,
        "status": first_stage["status"], "reason": first_stage.get("reason"),
        "label_version": first_stage.get("label_version"),
        "catalog_plant_count": first_stage.get("catalog_plant_count"),
        "first_stage": copy.deepcopy(first_stage), "candidates": ranked,
    }


class BioClipReranker:
    def __init__(self, weights=DEFAULT_WEIGHTS, *, threads=2, text_cache_path=None):
        if not 1 <= threads <= 4:
            raise ValueError("Use 1–4 inference threads.")
        verify_weights(weights)
        import open_clip
        import torch
        from open_clip.transform import PreprocessCfg, image_transform_v2

        torch.set_num_threads(threads)
        # Exact architecture matches the pinned official open_clip_config.json.
        # Local safetensors + built-in tokenizer: no hub calls or remote code.
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            "ViT-L-14", pretrained=str(weights.resolve()), device="cpu",
            return_transform=True, weights_only=True,
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
            image_interpolation="bicubic", image_resize_mode="shortest",
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.transforms = {
            "official": self.preprocess,
            "whole": image_transform_v2(
                PreprocessCfg(**{**self.model.visual.preprocess_cfg, "resize_mode": "squash"}),
                is_train=False,
            ),
        }
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.text_cache = OrderedDict()
        self.text_cache_limit = 1024
        self.disk_cache = None
        if text_cache_path is not None:
            from importlib.metadata import version

            from yizhu_server.recognition_text_cache import TextVectorCache

            namespace = json.dumps({
                "weights": MODEL_SHA256, "templates": TEXT_TEMPLATES,
                "tokenizer": "ViT-L-14", "open_clip": version("open_clip_torch"),
                "torch": version("torch"), "format": "normalized-f32-768-v1",
            }, sort_keys=True)
            self.disk_cache = TextVectorCache(Path(text_cache_path), namespace)

    def text_features(self, name):
        import torch

        if name in self.text_cache:
            self.text_cache.move_to_end(name)
            return self.text_cache[name]
        cached = self.disk_cache.get(name) if self.disk_cache is not None else None
        if cached is not None:
            feature = torch.from_numpy(cached)
            self.text_cache[name] = feature
            if len(self.text_cache) > self.text_cache_limit:
                self.text_cache.popitem(last=False)
            return feature
        prompts = [template.format(name=name) for template in TEXT_TEMPLATES]
        features = []
        with torch.inference_mode():
            # Small batches limit peak CPU RAM on the existing 8 GiB host.
            for start in range(0, len(prompts), 8):
                tokens = self.tokenizer(prompts[start:start + 8])
                features.append(torch.nn.functional.normalize(self.model.encode_text(tokens), dim=-1))
            feature = torch.nn.functional.normalize(torch.cat(features).mean(dim=0), dim=0)
        self.text_cache[name] = feature
        if self.disk_cache is not None:
            self.disk_cache.put(name, feature.detach().cpu().numpy())
        if len(self.text_cache) > self.text_cache_limit:
            self.text_cache.popitem(last=False)
        return feature

    def rerank(self, image, first_stage, *, view="official"):
        import torch

        candidates = first_stage.get("candidates")
        validate_candidates(candidates)
        if view not in self.transforms:
            raise ValueError("Unknown reranking image view.")
        with torch.inference_mode():
            # Re-encode with v2; never compare v2 image features against v1 vectors.
            texts = torch.stack([self.text_features(row["scientific_name"]) for row in candidates])
            image_feature = self.model.encode_image(self.transforms[view](image).unsqueeze(0))
            image_feature = torch.nn.functional.normalize(image_feature, dim=-1)[0]
            scores = (texts @ image_feature).tolist()
        return reranked_result(first_stage, scores, view=view)
