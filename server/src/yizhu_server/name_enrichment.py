"""Build an offline, pending-review name dataset; never modify the live catalog."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from yizhu_server.colchina_names import HAN, NS

SCIDB_SIZE = 1_408_491
SCIDB_SHA256 = "f01d5f9f915cee8343a90321e122f8fedbadcd54b746bf21852937c714594341"
SCIDB_SOURCE = {
    "id": "sciencedb-01711-v1",
    "title": "中国外来植物名录数据集",
    "version": "V1",
    "url": "https://doi.org/10.57760/sciencedb.01711",
    "license": "CC-BY-4.0",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "林秦文、崔晓、马金双，2022，Science Data Bank",
    "snapshotSha256": SCIDB_SHA256,
    "modifications": "Extracted name fields; aliases split on list separators; not a full copy.",
}
HEADERS = (
    "类别-Taxon", "中文科名-Family_chinese", "科名-Family", "中文属名",
    "属名-Genus", "中文名-Chinese name", "别名-variant name", "学名-scientific name",
    "命名人-authors", "生存状态-living status", "生存时间-duration",
    "生活型-Growth Habit", "原产国家或地区-native countries or regions",
    "引入省份-introduced Provinces",
)
BINOMIAL = re.compile(r"[A-Z][a-z]+ [a-z][a-z-]+")
LANGUAGES = {"zh", "zh-hans", "zh-cn"}
# Bibliographic provenance, not per-species name mappings. Other sources stay blocked.
FRPS_SERIES = "Q5862814"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scidb_rows(path: Path):
    """Accept only the verified DOI workbook, not the different journal attachment."""
    if path.stat().st_size != SCIDB_SIZE or digest(path) != SCIDB_SHA256:
        raise ValueError("ScienceDB workbook checksum/size mismatch.")
    with zipfile.ZipFile(path) as archive:
        for name, limit in (
            ("xl/sharedStrings.xml", 10_000_000),
            ("xl/worksheets/sheet1.xml", 15_000_000),
        ):
            if archive.getinfo(name).file_size > limit:
                raise ValueError("Unexpected workbook XML size.")
        strings = [
            "".join(node.text or "" for node in item.iter(NS + "t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        count = 0
        with archive.open("xl/worksheets/sheet1.xml") as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != NS + "row":
                    continue
                values = [""] * len(HEADERS)
                for cell in row:
                    if cell.find(NS + "f") is not None:
                        raise ValueError("Formula cells are not accepted.")
                    address = re.fullmatch(r"([A-N])\d+", cell.get("r", ""))
                    if not address:
                        raise ValueError("Unexpected workbook column.")
                    value = cell.find(NS + "v")
                    text = value.text if value is not None else ""
                    if cell.get("t") == "s":
                        text = strings[int(text)]
                    elif cell.get("t") == "inlineStr":
                        text = "".join(node.text or "" for node in cell.iter(NS + "t"))
                    values[ord(address[1]) - ord("A")] = (text or "").strip()
                count += 1
                if count == 1:
                    if tuple(values) != HEADERS:
                        raise ValueError("ScienceDB headers changed.")
                else:
                    yield {
                        "rowNumber": int(row.get("r")),
                        **dict(zip(HEADERS, values, strict=True)),
                    }
                row.clear()
        if count != 14_711:
            raise ValueError("ScienceDB row count changed.")


def scidb_records(rows):
    for row in rows:
        scientific = row["学名-scientific name"]
        primary = row["中文名-Chinese name"]
        names = [(primary, "source_primary")]
        for alias in re.split(r"[、,，;；]", row.get("别名-variant name", "")):
            alias = alias.strip()
            if alias and alias != primary:
                names.append((alias, "alias"))
        yield {
            "id": f"sciencedb-01711-v1:row-{row['rowNumber']}",
            "sourceId": SCIDB_SOURCE["id"],
            "scientificName": scientific,
            "rank": "species" if BINOMIAL.fullmatch(scientific) else None,
            "flags": ["rank_inferred_from_binomial"],
            # Do not import distribution text, life-form codes, or generate descriptions.
            "evidence": {
                "rowNumber": row["rowNumber"], "scientificName": scientific,
                "sourceChineseName": primary, "sourceAliasText": row.get("别名-variant name", ""),
                "family": row["科名-Family"], "genus": row["属名-Genus"],
                "authors": row["命名人-authors"],
            },
            "names": [
                {"name": name, "kind": kind, "language": "zh"}
                for name, kind in dict.fromkeys(names) if name
            ],
        }


def values(entity: dict, prop: str) -> list:
    return [
        claim["mainsnak"]["datavalue"]["value"]
        for claim in entity.get("claims", {}).get(prop, [])
        if claim.get("rank") != "deprecated" and "datavalue" in claim.get("mainsnak", {})
    ]


def load_entities(directory: Path) -> tuple[dict, list]:
    entities, snapshots = {}, []
    paths = sorted(directory.glob("Q*.json"))
    if len(paths) > 2000:
        raise ValueError("Use a smaller Wikidata batch (at most 2000 files).")
    total = 0
    for path in paths:
        size = path.stat().st_size
        total += size
        if size > 3_000_000 or total > 100_000_000:
            raise ValueError("Wikidata snapshot batch is too large.")
        items = json.loads(path.read_text(encoding="utf-8"))["entities"]
        for key, item in items.items():
            if key in entities or item.get("id") != key or not re.fullmatch(r"Q\d+", key):
                raise ValueError("Duplicate or invalid Wikidata entity.")
            entities[key] = item
        snapshots.append({"file": path.name, "sha256": digest(path)})
    return entities, snapshots


def wikidata_records(entities: dict):
    for qid, entity in sorted(entities.items()):
        scientific = values(entity, "P225")
        if not scientific:
            continue  # Bibliographic entities are evidence, not plants.
        ranks = values(entity, "P105")
        is_species = len(ranks) == 1 and ranks[0].get("id") == "Q7432"
        for claim in entity.get("claims", {}).get("P1843", []):
            name = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if claim.get("rank") == "deprecated" or name.get("language") not in LANGUAGES:
                continue
            flags, references = [], []
            for ref in claim.get("references", []):
                for snak in ref.get("snaks", {}).get("P248", []):
                    ref_id = snak.get("datavalue", {}).get("value", {}).get("id")
                    book = entities.get(ref_id, {})
                    if any(v.get("id") == FRPS_SERIES for v in values(book, "P361")):
                        references.append({
                            "id": ref_id, "revision": book.get("lastrevid"),
                            "title": book.get("labels", {}).get("zh", {}).get("value"),
                            "reference": ref,
                        })
            if not references:
                flags.append("unverified_reference")
            if claim.get("qualifiers"):
                flags.append("qualified_statement")
            if len(scientific) != 1:
                flags.append("ambiguous_scientific_name")
            if not entity.get("lastrevid"):
                flags.append("missing_source_revision")
            yield {
                "id": claim["id"], "sourceId": "wikidata",
                "scientificName": scientific[0] if len(scientific) == 1 else "",
                "rank": "species" if is_species else None,
                "flags": flags,
                "evidence": {
                    "entity": qid, "revision": entity.get("lastrevid"),
                    "scientificNames": scientific, "references": references,
                    "statementId": claim["id"], "qualifiers": claim.get("qualifiers", {}),
                    "sourceUrl": f"https://www.wikidata.org/wiki/{qid}",
                },
                "names": [{
                    "name": name["text"], "kind": "vernacular_statement",
                    "language": name["language"],
                }],
            }


def build_candidates(catalog: Path, output: Path, records, sources: list) -> dict:
    """Publish a separate dataset with all names pending; no apply/deploy operation."""
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a new path.")
    catalog_hash = digest(catalog)
    names, existing_chinese = defaultdict(list), defaultdict(set)
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for identifier, scientific, visible, data in db.execute(
            "SELECT id,scientific_name,is_visible,data FROM species"
        ):
            item = json.loads(data)
            names[scientific].append((identifier, bool(visible), item))
            for name in [item.get("chineseName"), *item.get("chineseAliases", [])]:
                if name:
                    existing_chinese[name].add(identifier)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="names-", suffix=".sqlite3", dir=output.parent)
    os.close(descriptor)
    counts = Counter()
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE metadata(data TEXT NOT NULL);
                CREATE TABLE sources(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE records(id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                    scientific_name TEXT NOT NULL, evidence TEXT NOT NULL);
                CREATE TABLE candidates(id TEXT PRIMARY KEY, record_id TEXT NOT NULL,
                    species_id TEXT, scientific_name TEXT NOT NULL, name TEXT NOT NULL,
                    language TEXT NOT NULL, kind TEXT NOT NULL, match_status TEXT NOT NULL,
                    current_visible INTEGER, current_chinese_name TEXT,
                    review_status TEXT NOT NULL CHECK(review_status='pending'),
                    flags TEXT NOT NULL);
                CREATE INDEX candidates_name ON candidates(name, species_id);
                CREATE INDEX candidates_species ON candidates(species_id);
                CREATE INDEX candidates_record ON candidates(record_id);
            """)
            db.executemany("INSERT INTO sources VALUES (?,?)", [
                (source["id"], json.dumps(source, ensure_ascii=False)) for source in sources
            ])
            source_ids = {source["id"] for source in sources}
            for record in records:
                if record["sourceId"] not in source_ids:
                    raise ValueError("Name record has no source manifest.")
                scientific = record["scientificName"]
                matches = names.get(scientific, [])
                status = "exact" if len(matches) == 1 else "ambiguous" if matches else "unmatched"
                if record["rank"] != "species" or not BINOMIAL.fullmatch(scientific):
                    status = "unresolved_rank"
                match = matches[0] if status == "exact" else None
                db.execute("INSERT INTO records VALUES (?,?,?,?)", (
                    record["id"], record["sourceId"], scientific,
                    json.dumps(record, ensure_ascii=False),
                ))
                counts["records"] += 1
                counts[f"records_{status}"] += 1
                for name in record["names"]:
                    text = name["name"].strip()
                    if not HAN.search(text) or len(text) > 80:
                        counts["invalid_names_skipped"] += 1
                        continue
                    flags = list(record["flags"])
                    if re.search(r"[、，,;；/()（）]", text):
                        flags.append("compound_or_qualified_name")
                    if re.search(r"提取物|制剂|根$|枝$|露$", text):
                        flags.append("possible_non_species_term")
                    if match and text == match[2].get("chineseName"):
                        flags.append("already_primary_name")
                    if match and not match[1]:
                        flags.append("hidden_requires_policy_approval")
                    if existing_chinese.get(text, set()) - ({match[0]} if match else set()):
                        flags.append("matches_other_catalog_name")
                    if (match and name["kind"] == "source_primary"
                            and match[2].get("chineseName")
                            and text != match[2]["chineseName"]):
                        flags.append("primary_name_difference")
                    key = json.dumps([record["id"], name["kind"], name["language"], text])
                    identifier = hashlib.sha256(key.encode()).hexdigest()
                    db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                        identifier, record["id"], match[0] if match else None, scientific,
                        text, name["language"], name["kind"], status,
                        int(match[1]) if match else None,
                        match[2].get("chineseName") if match else None,
                        "pending", json.dumps(sorted(set(flags))),
                    ))
                    counts["candidates"] += 1
                    counts[f"candidates_{status}"] += 1
            # An alias is not a unique key. Preserve all matches and flag ambiguity.
            shared_names = db.execute("""
                SELECT name FROM candidates WHERE species_id IS NOT NULL
                GROUP BY name HAVING count(DISTINCT species_id)>1
            """).fetchall()
            for (name,) in shared_names:
                for identifier, flags in db.execute(
                    "SELECT id,flags FROM candidates WHERE name=?", (name,)
                ).fetchall():
                    db.execute("UPDATE candidates SET flags=? WHERE id=?", (
                        json.dumps(sorted(set(json.loads(flags)) | {"shared_name"})), identifier,
                    ))
            counts["names_shared_across_species"] = len(shared_names)
            counts["records_without_names"] = counts["records"] - db.execute(
                "SELECT count(DISTINCT record_id) FROM candidates"
            ).fetchone()[0]
            for visible in (0, 1):
                counts[f"matched_{'visible' if visible else 'hidden'}_species"] = db.execute(
                    "SELECT count(DISTINCT species_id) FROM candidates WHERE current_visible=?",
                    (visible,),
                ).fetchone()[0]
            if digest(catalog) != catalog_hash:
                raise ValueError("Input catalog changed during the build; retry with a snapshot.")
            summary = {
                "schemaVersion": 1, "purpose": "pending_name_enrichment",
                "createdAt": datetime.now(UTC).isoformat(),
                "catalogSha256": catalog_hash, "counts": dict(counts),
                "productionChanges": False, "allCandidatesRequireReview": True,
            }
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(summary),))
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sciencedb", type=Path)
    parser.add_argument("--wikidata-dir", type=Path)
    parser.add_argument("--apply-from", type=Path, help="Apply eligible aliases from a candidate DB.")
    parser.add_argument("--reveal-sciencedb", type=Path,
                        help="Fill missing Chinese names from this verified workbook and reveal them.")
    parser.add_argument("--expected-count", type=int,
                        help="Abort unless --reveal-sciencedb changes exactly this many species.")
    args = parser.parse_args()
    if args.expected_count is not None and (
        args.expected_count < 0 or not args.reveal_sciencedb
    ):
        parser.error("--expected-count requires --reveal-sciencedb and must be nonnegative.")
    if args.reveal_sciencedb:
        if args.sciencedb or args.wikidata_dir or args.apply_from:
            parser.error("--reveal-sciencedb cannot be combined with other source actions.")
        from yizhu_server.catalog_names import reveal_sciencedb
        try:
            print(json.dumps(reveal_sciencedb(
                args.catalog, args.reveal_sciencedb, args.output,
                expected_count=args.expected_count,
            ), ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile, sqlite3.Error) as error:
            parser.exit(1, f"{error}\n")
        return
    if args.apply_from:
        if args.sciencedb or args.wikidata_dir:
            parser.error("--apply-from cannot be combined with source inputs.")
        from yizhu_server.catalog_names import apply_aliases
        try:
            print(json.dumps(apply_aliases(
                args.catalog, args.apply_from, args.output,
            ), ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, sqlite3.Error) as error:
            parser.exit(1, f"{error}\n")
        return
    if not args.sciencedb and not args.wikidata_dir:
        parser.error("Provide --sciencedb and/or --wikidata-dir.")
    try:
        records, sources = [], []
        if args.sciencedb:
            records.extend(scidb_records(scidb_rows(args.sciencedb)))
            sources.append(SCIDB_SOURCE)
        if args.wikidata_dir:
            entities, snapshots = load_entities(args.wikidata_dir)
            records.extend(wikidata_records(entities))
            sources.append({
                "id": "wikidata", "title": "Wikidata taxon common-name statements",
                "url": "https://www.wikidata.org", "license": "CC0-1.0",
                "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
                "snapshots": snapshots,
                "modifications": "Chinese P1843 statements only; pending review, not all aliases.",
            })
        if not records:
            raise ValueError("No source records found.")
        print(json.dumps(build_candidates(
            args.catalog, args.output, records, sources,
        ), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
