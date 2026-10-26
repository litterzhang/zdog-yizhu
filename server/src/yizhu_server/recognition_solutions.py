"""Versioned server-side recognition choices; no model imports until selected."""

import copy
from dataclasses import dataclass
from types import MappingProxyType

DEFAULT_SOLUTION = "bioclip-v1"


@dataclass(frozen=True)
class RecognitionSolution:
    id: str
    title: str
    plant_set_source: str
    version: str = "1"
    rerank_view: str | None = None
    shortlist_limit: int = 50
    experimental: bool = False
    cascade: bool = False
    model_directory: str | None = None
    model_sha256: str | None = None

    def public_info(self):
        return {"id": self.id, "version": self.version, "title": self.title,
                "experimental": self.experimental, "metadataId": self.id}

    def metadata(self, directory=None):
        from yizhu_server.solution_metadata import MetadataStore

        return MetadataStore(directory).get(self.id)


SOLUTIONS = MappingProxyType({
    DEFAULT_SOLUTION: RecognitionSolution(
        DEFAULT_SOLUTION, "BioCLIP v1 全参考检索", "catalog-reference",
    ),
    "bioclip-v2-rerank": RecognitionSolution(
        "bioclip-v2-rerank", "BioCLIP v1 召回 + v2 中心裁剪重排",
        plant_set_source="catalog-reference", rerank_view="official", experimental=True,
    ),
    "bioclip-v2-whole": RecognitionSolution(
        "bioclip-v2-whole", "BioCLIP v1 召回 + v2 整图重排",
        plant_set_source="catalog-reference", rerank_view="whole", experimental=True,
    ),
    "bioclip-multiview-plantclef": RecognitionSolution(
        "bioclip-multiview-plantclef", "多视图 BioCLIP + PlantCLEF 两级识别",
        plant_set_source="catalog-reference+plantclef",
        rerank_view="official", shortlist_limit=100, experimental=True, cascade=True,
    ),
    "plantnet-china": RecognitionSolution(
        "plantnet-china", "Pl@ntNet 中文植物增量识别",
        plant_set_source="trained-plantnet", version="20260924.1", experimental=True,
        model_directory="plantnet-china-replay-v2",
        model_sha256="25279397d94533a437ef0110eb4bbc0817d510ebff1ce394e211fdb64b6e210c",
    ),
    "target249-bioclip2": RecognitionSolution(
        "target249-bioclip2", "BioCLIP 2 中国常见植物识别（249 类）",
        plant_set_source="trained-bioclip", version="20261001.1",
        experimental=False,
        model_directory="target249-replay249-center-l2-inverse-v1",
        model_sha256="da8acb5597add5aa43ee20c6daa601aff59397567436be48cb738efd561bacb7",
    ),
    "target285-paraguayense-bioclip2": RecognitionSolution(
        "target285-paraguayense-bioclip2",
        "BioCLIP 2 中国常见植物与多肉识别（285 类，含胧月）",
        plant_set_source="trained-bioclip", version="20261006.1",
        experimental=False,
        model_directory="target285-paraguayense-frozen-old-v1",
        model_sha256="ba07fb5ca90d497fead259a35420bcda0c7bc84262edfad8afda3bcdafd80bd1",
    ),
    "target284-succulent-bioclip2-replay284": RecognitionSolution(
        "target284-succulent-bioclip2-replay284",
        "BioCLIP 2 中国常见植物与多肉识别（284 类）",
        plant_set_source="trained-bioclip", version="20261005.2",
        experimental=False,
        model_directory="target284-succulent-replay284-center-l2-inverse-lowreg-v1",
        model_sha256="71b771d929e89de72d0c6dff9e5db27e845dd8b385dff59a94fd11d65a222e2a",
    ),
    "plantnet-china-431": RecognitionSolution(
        "plantnet-china-431", "Pl@ntNet 中文植物增量识别（431 类归档）",
        plant_set_source="trained-plantnet", version="20260923.1", experimental=True,
        model_directory="plantnet-china-incremental-v1",
        model_sha256="4d750ba0a6b83b8498b7739e2680c367f5af6284edc99aa3a6c9dd7257d8967e",
    ),
})


def get_solution(identifier):
    if not isinstance(identifier, str) or identifier not in SOLUTIONS:
        raise ValueError("Unknown YIZHU_RECOGNITION_SOLUTION; select a registered solution.")
    return SOLUTIONS[identifier]


class SolutionProvider:
    """Keep the existing recognition contract while retaining each stage for audit."""

    def __init__(self, solution, retrieval, reranker=None):
        if solution.cascade:
            raise ValueError("Use CascadeProvider for a cascaded solution.")
        if bool(solution.rerank_view) != (reranker is not None):
            raise ValueError("The selected solution requires its configured model stages.")
        self.solution = solution
        self.retrieval = retrieval
        self.reranker = reranker

    def recognize(self, image, *, top_k=5):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        if self.reranker is None:
            result = copy.deepcopy(self.retrieval.recognize(image, top_k=top_k))
        else:
            first_stage = self.retrieval.retrieve(image, limit=self.solution.shortlist_limit)
            result = copy.deepcopy(self.reranker.rerank(
                image, first_stage, view=self.solution.rerank_view,
            ))
            # Preserve all 50 reranked references internally, but keep the same
            # bounded candidate list consumed by the existing client.
            result["reranking_candidates"] = result["candidates"]
            result["candidates"] = result["candidates"][:top_k]
            if first_stage["status"] == "uncertain":
                result["status"] = "uncertain"
                result["reason"] = first_stage.get("reason")
        result["solution"] = self.solution.public_info()
        return result


def create_provider(identifier, catalog, *, threads=2):
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 4:
        raise ValueError("Use 1–4 inference threads.")
    solution = get_solution(identifier)
    if solution.plant_set_source == "trained-plantnet":
        from yizhu_server.recognition_plantnet import PlantNetRecognitionProvider

        return PlantNetRecognitionProvider(solution, catalog, threads=threads)
    if solution.plant_set_source == "trained-bioclip":
        from yizhu_server.recognition_target_bioclip import (
            TargetBioClipRecognitionProvider,
        )

        return TargetBioClipRecognitionProvider(solution, catalog, threads=threads)
    from yizhu_server.catalog_recognition import CatalogRecognitionProvider

    if solution.cascade:
        from pathlib import Path

        from yizhu_server.recognition_cascade import CascadeProvider
        from yizhu_server.recognition_plantclef import PlantCLEFProvider
        from yizhu_server.recognition_rerank import BioClipReranker

        # Validate fallback files before loading primary weights. The fallback
        # model itself stays lazy until a genuine uncertain result reaches it.
        fallback = PlantCLEFProvider(catalog, threads=threads)
        retrieval = CatalogRecognitionProvider(catalog, threads=threads)
        cache = Path(__file__).resolve().parents[2] / ".cache/models/bioclip-v2/text-vectors.sqlite3"
        reranker = BioClipReranker(threads=threads, text_cache_path=cache)
        return CascadeProvider(solution, retrieval, reranker, fallback, catalog)

    retrieval = CatalogRecognitionProvider(catalog, threads=threads)
    reranker = None
    if solution.rerank_view is not None:
        from yizhu_server.recognition_rerank import BioClipReranker

        # Missing/incompatible weights fail explicitly, never silently run v1.
        reranker = BioClipReranker(threads=threads)
    return SolutionProvider(solution, retrieval, reranker)
