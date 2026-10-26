import pytest

from yizhu_server.species_expansion_snapshot import (
    _commons_rows,
    _gbif_rows,
    _inat_additional_rows,
    _perceptual,
)


def test_module_exposes_reviewed_candidate_projections():
    assert callable(_gbif_rows)
    assert callable(_commons_rows)
    assert callable(_inat_additional_rows)


def test_perceptual_hash_accepts_source_integer_or_manifest_hex():
    assert _perceptual(15) == 15
    assert _perceptual("000000000000000f") == 15
    with pytest.raises(ValueError):
        _perceptual(None)
