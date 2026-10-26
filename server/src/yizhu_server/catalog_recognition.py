"""CPU ranking against all published BioCLIP-v1 taxon vectors, without hub calls."""

from yizhu_server.catalog import CATALOG_DIRECTORY, PLANT_KINGDOMS, PlantCatalog, verify_asset
from yizhu_server.recognition import DEFAULT_WEIGHTS, MODEL_ID, MODEL_REVISION, verify_weights


def ranked_result(catalog, indices, scores, *, top_k, insufficient_detail=False):
    rows = catalog.labels(indices)
    candidates, seen = [], set()
    for row, score in zip(rows, scores, strict=True):
        identifier = row["species_id"] or f"nonplant:{row['idx']}"
        if identifier in seen:
            continue
        seen.add(identifier)
        candidates.append({
            "species_id": row["species_id"], "kind": "plant" if row["species_id"] else "nonplant",
            "scientific_name": row["name"], "label": row["name"],
            "score": float(score), "confidence": None,
        })
        if len(candidates) == top_k:
            break
    if not candidates:
        raise ValueError("No candidates returned.")
    uncertain = insufficient_detail or candidates[0]["kind"] != "plant"
    return {
        "source": "BioCLIP", "model": MODEL_ID, "revision": MODEL_REVISION,
        "label_version": catalog.metadata["version"], "is_demo": False,
        "status": "uncertain" if uncertain else "requires_review",
        "scope": "published_treeoflife_reference_taxa",
        "provider_version": "bioclip-v1-reference-1",
        "score_type": "cosine_similarity_not_calibrated_confidence",
        "reason": "insufficient_visual_detail" if insufficient_detail else (
            "nonplant_leading_candidate" if uncertain else None),
        "catalog_plant_count": catalog.metadata["plantCount"],
        "candidates": candidates,
    }


class CatalogRecognitionProvider:
    def __init__(self, catalog=None, *, directory=CATALOG_DIRECTORY, weights=DEFAULT_WEIGHTS, threads=2):
        self.catalog = catalog or PlantCatalog(directory / "catalog.sqlite3")
        if not self.catalog.available:
            raise ValueError("Install the verified plant catalog before starting recognition.")
        if not 1 <= threads <= 4:
            raise ValueError("Use 1–4 inference threads.")
        verify_weights(weights)
        verify_asset(directory / "txt_emb_species.npy", "txt_emb_species.npy")
        import numpy as np
        import open_clip
        import torch
        from bioclip.predict import preprocess_img

        torch.set_num_threads(threads)
        self.vectors = np.load(directory / "txt_emb_species.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.dtype != np.float32 or self.vectors.shape != (512, self.catalog.metadata["labelCount"]):
            raise ValueError("Taxonomy/embedding shape mismatch.")
        self.plant_indices = np.array(self.catalog.plant_indices(), dtype=np.int32)
        self.model, _ = open_clip.create_model_from_pretrained(
            "ViT-B-16", pretrained=str(weights.resolve()), device="cpu",
            return_transform=True, weights_only=True,
        )
        self.model.eval()
        self.preprocess = preprocess_img

    def recognize(self, image, *, top_k=5):
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        return self.retrieve(image, limit=top_k)

    def retrieve(self, image, *, limit=50):
        """Bounded shortlist for configured reranking; default recognition is unchanged."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        top_k = limit
        import numpy as np
        import torch
        from PIL import ImageStat

        with torch.inference_mode():
            feature = self.model.encode_image(self.preprocess(image).unsqueeze(0))
            feature = torch.nn.functional.normalize(feature, dim=-1)[0].cpu().numpy()
        # Memory-map the ~751 MiB reference matrix; no duplicate tensor or startup
        # text encoding. einsum avoids spawning an uncontrolled BLAS thread pool.
        scores = np.einsum("d,dn->n", feature, self.vectors, optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError("Invalid embedding scores.")
        # Plant-specific retrieval, with a separate all-kingdom counter-signal.
        # An animal/fungus leading globally must not be presented as plant success.
        count = min(top_k * 10, len(self.plant_indices))
        selected = np.argpartition(scores[self.plant_indices], -count)[-count:]
        indices = self.plant_indices[selected]
        indices = indices[np.argsort(scores[indices])[::-1]]
        # A low-detail check is a capture-quality heuristic, not a probability or
        # a universal unknown-species detector. All other matches need review too.
        detail = max(ImageStat.Stat(image.resize((64, 64))).stddev)
        result = ranked_result(self.catalog, indices.tolist(), scores[indices].tolist(),
                               top_k=top_k, insufficient_detail=detail < 2)
        leading_index = int(np.argmax(scores))
        leading = self.catalog.labels([leading_index])[0]
        result["global_leading_candidate"] = {
            "name": leading["name"], "kingdom": leading["kingdom"],
            "species_id": leading["species_id"], "score": float(scores[leading_index]),
        }
        if leading["species_id"] is None and detail >= 2:
            result.update(status="uncertain", reason=(
                "unresolved_reference_taxon" if leading["kingdom"] in PLANT_KINGDOMS
                else "nonplant_leading_candidate"))
        return result
