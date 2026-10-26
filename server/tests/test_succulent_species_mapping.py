from yizhu_server.succulent_species_mapping import (
    ACCEPTED_NAME_OVERRIDES,
    DEFERRED_NO_OPEN_TRAINING,
    REJECTED_EXACT_NAMES,
)


def test_horticultural_collisions_are_not_promoted_as_species():
    assert {"秋丽", "蜡牡丹", "红化妆", "百惠"} <= set(REJECTED_EXACT_NAMES)


def test_verified_synonyms_use_accepted_species_names():
    assert ACCEPTED_NAME_OVERRIDES == {
        "大和锦": "Echeveria purpusiorum",
        "黑法师": "Aeonium arboreum",
    }


def test_species_without_required_open_evidence_are_deferred():
    assert {"小红衣", "红稚莲"} <= set(DEFERRED_NO_OPEN_TRAINING)
