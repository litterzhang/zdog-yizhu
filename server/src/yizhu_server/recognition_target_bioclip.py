"""Serve a validated finite-target BioCLIP 2 classifier behind the existing API."""

import json
import math
from pathlib import Path

from yizhu_server.recognition_candidates import close_candidates, policy_info
from yizhu_server.recognition_rerank import MODEL_ID, BioClipReranker
from yizhu_server.training_data import sha256

SERVER = Path(__file__).resolve().parents[2]
PRESENCE_VIEW_POLICY = "whole-then-center-square-v1"
CENTER_FALLBACK_MIN_ASPECT_RATIO = 1.25


def ranked_candidates(logits, classes, *, limit=5):
    import torch

    if (
        logits.shape != (len(classes),)
        or not torch.isfinite(logits).all()
        or not 1 <= limit <= min(20, len(classes))
    ):
        raise ValueError("Invalid finite-target BioCLIP logits.")
    scores = logits.softmax(0)
    indices = torch.argsort(logits, descending=True, stable=True)[:limit]
    return [{
        "species_id": classes[index]["id"],
        "scientific_name": classes[index]["scientificName"],
        "label": classes[index]["scientificName"],
        "kind": "plant", "score": float(scores[index]), "confidence": None,
    } for index in indices.tolist()]


def presence_with_center_fallback(classifier, image, threshold):
    """Retry a rejected non-square photo on its center square without lowering the gate."""
    whole = classifier.predict(image)
    scores = {"whole": whole["score"]}
    selected_view = "whole"
    score = whole["score"]
    width, height = image.size
    aspect_ratio = max(width, height) / min(width, height)
    if score < threshold and aspect_ratio >= CENTER_FALLBACK_MIN_ASPECT_RATIO:
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        with image.crop((left, top, left + side, top + side)) as center:
            center_score = classifier.predict(center)["score"]
        scores["center_square"] = center_score
        if center_score > score:
            score, selected_view = center_score, "center_square"
    return {
        **whole,
        "score": score,
        "containsPlant": score >= threshold,
        "viewPolicy": PRESENCE_VIEW_POLICY,
        "selectedView": selected_view,
        "viewScores": scores,
    }


class TargetBioClipRecognitionProvider:
    def __init__(self, solution, catalog, *, threads=2):
        import torch
        from safetensors.torch import load_file

        from yizhu_server.plant_presence import PlantPresenceClassifier

        if not catalog.available:
            raise ValueError("Install the Chinese catalog before starting recognition.")
        if not solution.model_directory or not solution.model_sha256:
            raise ValueError("Select a pinned validated target BioCLIP model.")
        directory = (SERVER / ".artifacts/models" / solution.model_directory).resolve()
        if not directory.is_relative_to((SERVER / ".artifacts/models").resolve()):
            raise ValueError("Model directory escapes the local artifact store.")
        metadata_path = directory / "production.metadata.json"
        if sha256(metadata_path) != solution.model_sha256:
            raise ValueError("Registered target BioCLIP metadata checksum mismatch.")
        metadata = json.loads(metadata_path.read_text())
        plants = metadata.get("plantSet", {}).get("plants", [])
        classifier = directory / metadata.get("classifier", {}).get("file", "")
        if (
            metadata.get("status") != "validated-independent"
            or metadata.get("productionApproved") is not True
            or metadata.get("finalIndependentGoalMet") is not True
            or metadata.get("encoder", {}).get("model") != MODEL_ID
            or sha256(classifier) != metadata.get("classifier", {}).get("sha256")
            or len(plants) != metadata.get("plantSet", {}).get("count")
            or [row["classIndex"] for row in plants] != list(range(len(plants)))
        ):
            raise ValueError("Target BioCLIP model is not independently approved.")
        state = load_file(classifier, device="cpu")
        if set(state) not in ({"weight", "bias"}, {"weight", "bias", "feature_mean"}):
            raise ValueError("Unsupported target BioCLIP classifier transform.")
        self.encoder = BioClipReranker(threads=threads)
        self.head = torch.nn.Linear(768, len(plants))
        self.head.load_state_dict(
            {"weight": state["weight"], "bias": state["bias"]}, strict=True,
        )
        self.head.eval().requires_grad_(False)
        self.feature_mean = state.get("feature_mean")
        gate = metadata.get("presenceGate", {})
        gate_path = SERVER / gate.get("headFile", "")
        if (
            not gate_path.is_file() or sha256(gate_path) != gate.get("headSha256")
            or not isinstance(gate.get("threshold"), (int, float))
            or isinstance(gate.get("threshold"), bool)
            or not math.isfinite(gate["threshold"]) or not 0 <= gate["threshold"] <= 1
        ):
            raise ValueError("Validated target model has no usable plant-presence gate.")
        self.presence = PlantPresenceClassifier(gate_path, threads=threads)
        if self.presence.encoder.identity != gate.get("encoder"):
            raise ValueError("Plant-presence encoder identity changed.")
        self.threshold = gate["threshold"]
        self.solution, self.catalog = solution, catalog
        self.directory, self.metadata = directory, metadata
        self.classes = plants
        self.class_ids = {row["id"] for row in plants}
        if len(self.class_ids) != len(plants):
            raise ValueError("Duplicate trained target species IDs.")

    def public_model_info(self):
        return {
            "modelVersion": self.solution.model_sha256,
            "supportedPlantCount": len(self.classes),
            "plantPresenceGate": True,
            "candidateSelection": policy_info(),
            "independentEvaluation": self.metadata["independentEvaluation"],
        }

    def recognize(self, image, *, top_k=5):
        import torch

        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        presence = presence_with_center_fallback(self.presence, image, self.threshold)
        passed = presence["score"] >= self.threshold
        candidates = []
        if passed:
            with torch.inference_mode():
                vector = torch.nn.functional.normalize(
                    self.encoder.model.encode_image(
                        self.encoder.preprocess(image).unsqueeze(0)
                    ),
                    dim=1,
                )[0]
                if self.feature_mean is not None:
                    vector = torch.nn.functional.normalize(
                        vector - self.feature_mean, dim=0,
                    )
                candidates = ranked_candidates(
                    self.head(vector), self.classes, limit=min(top_k, 5),
                )
        hidden_leader = bool(candidates and self.catalog.get(candidates[0]["species_id"]) is None)
        reason = (
            "nonplant_gate" if not passed
            else "hidden_leading_candidate" if hidden_leader
            else "no_trained_candidates" if not candidates else None
        )
        displayed = close_candidates(candidates) if not reason else candidates
        return {
            "source": "BioCLIP", "model": MODEL_ID,
            "revision": self.solution.model_sha256,
            "label_version": self.metadata["classifier"]["sha256"],
            "provider_version": f"target-bioclip-{self.solution.version}",
            "is_demo": False,
            "status": "uncertain" if reason else "requires_review",
            "reason": reason, "scope": "validated_chinese_species_closed_set",
            "score_type": "softmax_not_calibrated_confidence",
            "catalog_plant_count": len(self.classes),
            "candidates": displayed,
            "candidate_selection": {
                **policy_info(), "applied": reason is None,
                "originalCount": len(candidates), "returnedCount": len(displayed),
            },
            "solution": {**self.solution.public_info(), **self.public_model_info()},
            "presence_gate": {
                "score": presence["score"], "threshold": self.threshold,
                "passed": passed, "confidence": None,
                "model_version": presence["modelVersion"],
                "view_policy": presence["viewPolicy"],
                "selected_view": presence["selectedView"],
                "view_scores": presence["viewScores"],
            },
        }
