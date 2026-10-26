"""Cross-source scientific synonym evidence; never fuzzy-match names or infer parent taxa."""


def accepted_key(record, expected_name):
    if (
        record.get("matchType") != "EXACT" or record.get("rank") != "SPECIES"
        or record.get("canonicalName") != expected_name
        or type(record.get("usageKey")) is not int or record["usageKey"] < 1
    ):
        raise ValueError("Require an exact species-level GBIF name match.")
    if record.get("status") == "ACCEPTED":
        if record.get("acceptedUsageKey") not in (None, record["usageKey"]):
            raise ValueError("Inconsistent accepted taxon record.")
        return record["usageKey"]
    if record.get("status") == "SYNONYM":
        key = record.get("acceptedUsageKey")
        if type(key) is int and key > 0:
            return key
    raise ValueError("Require an accepted taxon or explicit synonym, not a higher-rank match.")


def build_bridge(species, source_taxon, target_match, source_match):
    result = {
        **species, "inatTaxonId": source_taxon["id"],
        "sourceScientificName": source_taxon["name"], "sourceTaxonRank": "species",
        "sourceNameEvidence": {
            "matchType": "gbif-exact-synonym",
            "queryScientificName": species["scientificName"],
            "sourceScientificName": source_taxon["name"], "sourceTaxonId": source_taxon["id"],
            "gbifTargetMatch": target_match, "gbifSourceMatch": source_match,
            "inatTaxon": {k: source_taxon[k] for k in ("id", "name", "rank", "is_active", "ancestor_ids")},
            "note": "Source does not list this synonym itself; GBIF exact records share "
                    "the same accepted taxon key. Original catalog identity is unchanged.",
        },
    }
    validate_bridge(result)
    return result


def validate_bridge(species):
    evidence = species.get("sourceNameEvidence", {})
    target_name, source_name = species["scientificName"], species.get("sourceScientificName")
    taxon = evidence.get("inatTaxon", {})
    if (
        evidence.get("matchType") != "gbif-exact-synonym"
        or evidence.get("queryScientificName") != target_name
        or evidence.get("sourceScientificName") != source_name
        or evidence.get("sourceTaxonId") != species.get("inatTaxonId")
        or type(taxon.get("id")) is not int or taxon["id"] < 1
        or species.get("sourceTaxonRank", "species") != "species"
        or taxon.get("id") != species.get("inatTaxonId") or taxon.get("name") != source_name
        or taxon.get("rank") != "species" or taxon.get("is_active") is not True
        or 47126 not in taxon.get("ancestor_ids", [])
    ):
        raise ValueError("Source taxon identity does not match the scientific synonym bridge.")
    target_key = accepted_key(evidence.get("gbifTargetMatch", {}), target_name)
    source_key = accepted_key(evidence.get("gbifSourceMatch", {}), source_name)
    if target_key != source_key:
        raise ValueError("Scientific names do not share an explicitly accepted taxon.")
