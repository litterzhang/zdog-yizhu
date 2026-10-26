"""Build a conservative species-only expansion from a horticultural name atlas."""

import argparse
import json
from pathlib import Path

from yizhu_server.training_data import now, sha256, write_json

POLICY = "succulent-species-only-v1"
# These article labels denote horticultural hybrids/cultivars whose pictured plant
# must not be turned into the unrelated same-named catalog species.
REJECTED_EXACT_NAMES = {
    "秋丽": "园艺杂交名称，不能可靠归入文章目录命中的 Graptopetalum bartramii。",
    "蜡牡丹": "园艺杂交或品种名称，不能因同名直接归入 Echeveria nuda。",
    "红化妆": "园艺杂交或品种名称，不能因同名直接归入 Sedum palmeri。",
    "百惠": "卷绢属园艺品种名，文章未提供可核验的物种身份。",
}
# GBIF resolves the catalog scientific spelling/name as a synonym. Preserve the
# stable catalog ID, but train and report the accepted species name.
DEFERRED_NO_OPEN_TRAINING = {
    "红稚莲": "已核对到 Echeveria macdougallii，虽有训练图片，但开放来源没有取得与训练/开发图片独立的终验图片，暂不进入模型支持集。",
    "小红衣": "已核对到 Echeveria globulosa，但三类开放来源均无可用于训练的许可兼容图片，暂不进入模型支持集。",
}

ACCEPTED_NAME_OVERRIDES = {
    "大和锦": "Echeveria purpusiorum",
    "黑法师": "Aeonium arboreum",
}


def build(seed_path: Path, gbif_path: Path, production_path: Path, catalog_path: Path):
    import sqlite3

    seed = json.loads(seed_path.read_text())
    verified = {row["sourceName"]: row for row in json.loads(gbif_path.read_text())}
    production = json.loads(production_path.read_text())
    current = production["plantSet"]["plants"]
    current_ids = {row["id"] for row in current}
    with sqlite3.connect(catalog_path) as db:
        db.row_factory = sqlite3.Row
        catalog = {
            row["id"]: json.loads(row["data"])
            for row in db.execute("SELECT id,data FROM species WHERE is_visible=1")
        }
        scientific = {
            item["scientificName"]: (identifier, item)
            for identifier, item in catalog.items()
        }
    decisions, additions = [], []
    for row in seed["records"]:
        decision = {
            "index": row["index"], "sourceName": row["sourceName"],
            "sourceImageIndex": row["index"], "policy": POLICY,
        }
        if row["status"] != "candidate-catalog-exact":
            decision.update(
                status="excluded", reason=row["reason"], speciesId=None,
                scientificName=None, officialChineseName=None,
            )
            decisions.append(decision)
            continue
        if row["sourceName"] in REJECTED_EXACT_NAMES:
            decision.update(
                status="excluded-horticultural-name",
                reason=REJECTED_EXACT_NAMES[row["sourceName"]], speciesId=None,
                scientificName=None, officialChineseName=None,
            )
            decisions.append(decision)
            continue
        if row["sourceName"] in DEFERRED_NO_OPEN_TRAINING:
            decision.update(
                status="mapped-species-deferred-no-training-data",
                reason=DEFERRED_NO_OPEN_TRAINING[row["sourceName"]],
                speciesId=row["speciesId"], scientificName=row["scientificName"],
                officialChineseName=row["officialChineseName"],
            )
            decisions.append(decision)
            continue
        evidence = verified[row["sourceName"]]["gbifMatch"]
        if not (
            evidence.get("confidence") == 100
            and evidence.get("matchType") == "EXACT"
            and evidence.get("rank") == "SPECIES"
            and evidence.get("status") in {"ACCEPTED", "SYNONYM"}
        ):
            raise ValueError(f"GBIF species verification failed for {row['sourceName']}.")
        accepted_name = ACCEPTED_NAME_OVERRIDES.get(
            row["sourceName"], evidence.get("species") or row["scientificName"],
        )
        species_id = row["speciesId"]
        item = catalog.get(species_id)
        if row["sourceName"] == "黑法师":
            species_id, item = scientific[accepted_name]
        if item is None:
            raise ValueError("Reviewed species is absent from the visible catalog.")
        if species_id in current_ids:
            status = "mapped-existing-species"
        else:
            status = "accepted-new-species"
            additions.append({
                "speciesId": species_id, "id": species_id,
                "catalogSpeciesId": species_id,
                "scientificName": accepted_name,
                "catalogScientificName": item["scientificName"],
                "chineseName": item["chineseName"],
                "chineseAliases": list(dict.fromkeys([
                    *(item.get("chineseAliases") or []), row["sourceName"],
                ])),
                "sourceInputs": [{
                    "inputList": "wechat-succulent-atlas-20261005",
                    "position": row["index"], "inputName": row["sourceName"],
                }],
                "mappingEvidence": {
                    "catalogNameSources": item.get("nameSources", []),
                    "gbifMatch": evidence,
                    "policy": POLICY,
                },
            })
        decision.update(
            status=status, reason="目录中文物种名与GBIF物种级精确匹配共同支持。",
            speciesId=species_id, scientificName=accepted_name,
            catalogScientificName=item["scientificName"],
            officialChineseName=item["chineseName"], gbifMatch=evidence,
        )
        decisions.append(decision)
    if len(decisions) != 263 or len({row["sourceName"] for row in decisions}) != 263:
        raise ValueError("Species-only review must cover all 263 source labels exactly once.")
    additions.sort(key=lambda row: row["speciesId"])
    classes = []
    for row in current:
        classes.append({
            "speciesId": row["id"], "id": row["id"],
            "catalogSpeciesId": row.get("catalogSpeciesId", row["id"]),
            "scientificName": row["scientificName"],
            "chineseName": row["chineseName"],
            "chineseAliases": row.get("chineseAliases", []),
            "sourceInputs": row.get("sourceInputs", []),
        })
    classes.extend(additions)
    if len({row["speciesId"] for row in classes}) != len(classes):
        raise ValueError("Expansion contains duplicate catalog species IDs.")
    return {
        "createdAt": now(), "policy": POLICY,
        "sourceAtlasSha256": sha256(seed_path),
        "gbifVerificationSha256": sha256(gbif_path),
        "baseProductionSha256": sha256(production_path),
        "sourceLabelCount": len(decisions),
        "acceptedNewSpeciesCount": len(additions),
        "mappedExistingSpeciesCount": sum(
            row["status"] == "mapped-existing-species" for row in decisions
        ),
        "excludedLabelCount": sum(row["status"].startswith("excluded") for row in decisions),
        "deferredSpeciesCount": sum("deferred" in row["status"] for row in decisions),
        "targetClassCount": len(classes),
        "decisions": decisions, "additions": additions, "classes": classes,
        "trainingImagesFromArticle": 0,
        "articleImagesUsage": "reference-only-not-training-eligible",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--gbif", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable mapping output path.")
    result = build(
        args.seed.resolve(), args.gbif.resolve(), args.production.resolve(),
        args.catalog.resolve(),
    )
    write_json(args.output.resolve(), result)
    print(json.dumps({
        key: result[key] for key in (
            "sourceLabelCount", "acceptedNewSpeciesCount",
            "mappedExistingSpeciesCount", "excludedLabelCount", "deferredSpeciesCount",
            "targetClassCount",
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
