from yizhu_server.target_feature_expansion import _identity


def test_feature_identity_normalizes_scalar_id_types():
    row = {
        "sampleId": "one", "speciesId": "species", "imageSha256": "hash",
        "observerGroup": "observer", "photoId": 1, "observationId": "1",
        "split": "train",
    }
    assert _identity(row) == _identity({**row, "photoId": "1"})
