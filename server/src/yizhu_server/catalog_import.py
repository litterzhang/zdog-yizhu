"""Build a versioned catalog from verified public taxonomy; no training images."""

import argparse
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from yizhu_server.catalog import (
    ASSETS,
    CATALOG_DIRECTORY,
    CATALOG_PATH,
    CATALOG_REVISION,
    CATALOG_SOURCE,
    CATALOG_URL,
    PLANT_KINGDOMS,
    RANKS,
    legacy_species,
    stable_id,
    verify_asset,
)
from yizhu_server.catalog_schema import SCHEMA_VERSION, create_catalog_schema
from yizhu_server.colchina_names import HAN, build_national_index, workbook_rows
from yizhu_server.colchina_names import PATH as NATIONAL_PATH
from yizhu_server.colchina_names import VERSION as NATIONAL_VERSION
from yizhu_server.species_knowledge import load_knowledge


def build_catalog(names, output, *, growth_forms=None, chinese_names=None, names_version=None, knowledge=None):
    """Separate from file verification so tiny synthetic fixtures can exercise it."""
    growth_forms = growth_forms or {}
    chinese_names = chinese_names or {}
    knowledge = knowledge or {}
    if output.exists():
        raise ValueError("Catalog already exists; use a new output path instead of overwriting.")
    output.parent.mkdir(parents=True, exist_ok=True)
    legacy = {item["scientificName"]: item for item in legacy_species().values()}
    species, labels, taxa, membership = {}, [], {}, set()
    for index, entry in enumerate(names):
        if (not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], list)
                or len(entry[0]) != 7 or not all(isinstance(value, str) for value in entry[0])
                or not isinstance(entry[1], str)):
            raise ValueError("Unexpected taxonomy label format.")
        lineage, english_name = entry
        scientific = " ".join(lineage[5:7]).strip()
        is_plant = lineage[0] in PLANT_KINGDOMS
        identifier = None
        # Labels such as "Genus sp. ABC" are unresolved references, not a species
        # users should be able to confirm. Keep their vector as counter-evidence.
        resolved = not re.match(r"^(sp|spp|cf|aff|indet)(?:[.\s_0-9]|$)", lineage[6], re.I)
        if is_plant and resolved:
            identifier = legacy.get(scientific, {}).get("id") or stable_id("tol10m", scientific)
        labels.append((index, identifier, lineage[0], scientific))
        if identifier is None:
            continue
        taxonomy = dict(zip(RANKS, lineage[:6], strict=True))
        if identifier in species:
            existing = species[identifier]
            if taxonomy != existing["taxonomy"] and taxonomy not in existing["taxonomyAlternatives"]:
                existing["taxonomyAlternatives"].append(taxonomy)
            continue  # Same scientific name: preserve alternate lineages, not duplicate species.
        trait = growth_forms.get(scientific)
        chinese = chinese_names.get(("species", scientific))
        # Historical labels preserve IDs only; never supply names or aliases.
        display_name = chinese["name"] if chinese else None
        display_name = display_name.strip() if display_name else None
        if display_name and not HAN.search(display_name):
            display_name = None
        chinese_taxonomy = {}
        for rank, name in taxonomy.items():
            translation = chinese_names.get((rank, name))
            if translation:
                chinese_taxonomy[rank] = translation["name"]
        item = {
            "id": identifier, "scientificName": scientific,
            "commonName": display_name or scientific,
            # The pinned national workbook has no alias column.
            "chineseName": display_name, "chineseAliases": [],
            "isVisible": bool(display_name),
            "chineseNameOrigin": "colchina" if display_name else None,
            "nameSources": [chinese["source"]] if chinese else [],
            "taxonomyChinese": chinese_taxonomy,
            "englishName": english_name, "taxonomy": taxonomy, "taxonomyAlternatives": [],
            "knowledge": knowledge.get(scientific), "growthForms": [trait["form"]] if trait else [],
            "sources": [{"title": CATALOG_SOURCE, "url": CATALOG_URL, "version": CATALOG_REVISION}],
            "growthFormSources": trait.get("sources", []) if trait else [],
        }
        species[identifier] = item
        # Browse order -> family -> genus. Full lineage remains available in details.
        parent = ""
        for position, rank in enumerate(RANKS[3:6], start=3):
            # The browsing tree starts at order. Historical class/phylum labels
            # must not produce indistinguishable duplicate order buttons.
            node_id = stable_id(f"taxon-{rank}", "|".join(lineage[3:position + 1]))
            translated = chinese_names.get((rank, lineage[position]))
            taxa[node_id] = (
                node_id, rank, lineage[position], parent,
                translated["name"] if translated else None,
                json.dumps(translated["source"], ensure_ascii=False) if translated else None,
            )
            membership.add((identifier, node_id))
            parent = node_id
    if not species:
        raise ValueError("No plant taxa in this catalog.")
    metadata = {
        "storageSchemaVersion": SCHEMA_VERSION,
        "available": True, "source": CATALOG_SOURCE, "version": CATALOG_REVISION,
        "sourceUrl": CATALOG_URL, "plantCount": len(species), "labelCount": len(labels),
        "growthFormCount": sum(bool(item["growthForms"]) for item in species.values()),
        "chineseNameCount": sum(bool(item["chineseName"]) for item in species.values()),
        "visiblePlantCount": sum(item["isVisible"] for item in species.values()),
        "hiddenPlantCount": sum(not item["isVisible"] for item in species.values()),
        "visibilityPolicy": "chinese_name_required",
        "chineseTaxonCount": sum(bool(node[4]) for node in taxa.values()),
        "chineseNamesVersion": (names_version or NATIONAL_VERSION) if chinese_names else None,
        "knowledgeCount": sum(item["knowledge"] is not None for item in species.values()),
        "unresolvedPlantLabelCount": sum(
            row[1] is None and row[2] in PLANT_KINGDOMS for row in labels),
        "scope": "published_reference_taxa_not_all_plants",
        "taxonomyPolicy": "first_source_lineage_with_alternatives_preserved",
    }
    descriptor, temporary = tempfile.mkstemp(prefix="catalog-", suffix=".sqlite3", dir=output.parent)
    os.close(descriptor)
    try:
        with sqlite3.connect(temporary) as db:
            create_catalog_schema(db)
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
            db.executemany("INSERT INTO species VALUES (?,?,?,?,?,?,?)", [
                (item["id"], item["scientificName"],
                 " ".join([item["scientificName"], item["commonName"], item["englishName"],
                           *item["chineseAliases"]]),
                 item["growthForms"][0] if item["growthForms"] else "unknown",
                 json.dumps(item, ensure_ascii=False), int(bool(item["chineseName"])),
                 int(item["isVisible"]))
                for item in species.values()
            ])
            db.executemany("INSERT INTO labels VALUES (?,?,?,?)", labels)
            db.executemany("INSERT INTO taxa VALUES (?,?,?,?,?,?)", taxa.values())
            db.executemany("INSERT INTO membership VALUES (?,?)", membership)
            db.execute("ANALYZE")
        os.link(temporary, output)  # Atomic publish; refuses to replace an existing catalog.
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Import pinned BioCLIP taxonomy and verify vectors")
    parser.add_argument("--source", type=Path, default=CATALOG_DIRECTORY)
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--national-source", type=Path, default=NATIONAL_PATH)
    args = parser.parse_args()
    try:
        for filename in ASSETS:
            verify_asset(args.source / filename, filename)
        names = json.loads((args.source / "txt_emb_species.json").read_text())
        traits_path = Path(__file__).parent / "data/growth-forms.json"
        traits = json.loads(traits_path.read_text()) if traits_path.exists() else {}
        chinese_names = build_national_index(workbook_rows(args.national_source))
        print(json.dumps(build_catalog(
            names, args.output, growth_forms=traits, chinese_names=chinese_names,
            names_version=NATIONAL_VERSION, knowledge=load_knowledge(),
        ), ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
