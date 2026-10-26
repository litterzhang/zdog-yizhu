import json

import pytest

from yizhu_server.licensed_target_collection import POLICY, run


def test_rejects_non_species_plan(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "policy": POLICY,
        "splitTargets": {"train": 40, "validation": 10, "test": 10},
        "species": [{
            "speciesId": "one", "scientificName": "Example cultivar",
            "inatTaxonId": 1, "taxonRank": "variety",
        }],
    }))
    with pytest.raises(ValueError, match="species-level"):
        run(plan, tmp_path / "output")
