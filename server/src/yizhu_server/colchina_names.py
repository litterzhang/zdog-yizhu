"""Import the user-provided national checklist snapshot, without network or Excel execution."""

import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / ".cache/catalog/colchina-2026/plantae-2026-48356.xlsx"
VERSION = "CoL-China-2026-v2"
SIZE = 3_661_260
SHA256 = "bfae790b1bda9ba103e68d6b028fe7f5322eaadeb4873b249ccbc24fdd0d6582"
SOURCE = {
    "title": "中国生物物种名录 2026 · 植物界",
    "url": "http://www.sp2000.org.cn/download",
    "version": "2026",
    "snapshotSha256": SHA256,
    "license": "工作簿未附开放许可声明；按来源使用要求引用，不转载完整工作簿",
}
HEADERS = (
    "物种拉丁名", "物种中文名", "界拉丁名", "界中文名", "门拉丁名", "门中文名",
    "纲拉丁名", "纲中文名", "目拉丁名", "目中文名", "科拉丁名", "科中文名",
    "属拉丁名", "属中文名", "审核专家/数据源",
)
RANK_COLUMNS = (
    ("species", "物种"), ("kingdom", "界"), ("phylum", "门"),
    ("class", "纲"), ("order", "目"), ("family", "科"), ("genus", "属"),
)
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
HAN = re.compile(r"[\u3400-\u9fff]")


def workbook_rows(path=PATH):
    if not path.is_file() or path.stat().st_size != SIZE:
        raise ValueError("National checklist file is missing/incomplete.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("National checklist snapshot checksum mismatch.")
    # This is a pinned one-sheet snapshot reader, not a general spreadsheet engine.
    with zipfile.ZipFile(path) as archive:
        for name, limit in [("xl/sharedStrings.xml", 10_000_000), ("xl/worksheets/sheet1.xml", 40_000_000)]:
            if archive.getinfo(name).file_size > limit:
                raise ValueError("Unexpected national checklist size.")
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
                        raise ValueError("Formula cells are not accepted as taxonomy data.")
                    column = re.match(r"[A-Z]+", cell.get("r", ""))
                    if not column:
                        raise ValueError("Invalid spreadsheet cell address.")
                    index = 0
                    for letter in column[0]:
                        index = index * 26 + ord(letter) - ord("A") + 1
                    value = cell.find(NS + "v")
                    raw = value.text if value is not None else ""
                    if cell.get("t") == "s":
                        raw = strings[int(raw)]
                    elif cell.get("t") == "inlineStr":
                        raw = "".join(node.text or "" for node in cell.iter(NS + "t"))
                    if index <= len(values):
                        values[index - 1] = (raw or "").strip()
                if count == 0:
                    if tuple(values) != HEADERS:
                        raise ValueError("National checklist headers changed; inspect before importing.")
                else:
                    yield {"rowNumber": count + 1, **dict(zip(HEADERS, values, strict=True))}
                count += 1
                row.clear()
        if count != 48_357:
            raise ValueError("National checklist row count changed.")


def build_national_index(rows):
    matches = defaultdict(dict)
    for row in rows:
        if row.get("界拉丁名") != "Plantae":
            continue
        for rank, label in RANK_COLUMNS:
            scientific, chinese = row.get(f"{label}拉丁名", ""), row.get(f"{label}中文名", "")
            if not scientific or not HAN.search(chinese):
                continue
            key = (rank, scientific)
            if chinese not in matches[key]:
                matches[key][chinese] = {
                    "name": chinese, "aliases": [], "origin": "colchina",
                    "source": {
                        **SOURCE, "recordId": f"植物界-2026-48356!row-{row['rowNumber']}",
                        "sourceName": chinese, "matchedScientificName": scientific,
                        "acceptedScientificName": scientific, "matchType": "exact",
                        "attribution": row.get("审核专家/数据源", ""),
                    },
                }
    # Different names for the same scientific label require curation, not first-row wins.
    return {key: next(iter(values.values())) for key, values in matches.items() if len(values) == 1}
