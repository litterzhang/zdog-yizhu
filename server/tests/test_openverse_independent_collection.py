from yizhu_server.openverse_independent_collection import (
    ALLOWED_HOSTS,
    download_variant,
    explicit_name,
    normalized,
)


def test_exact_scientific_name_is_required_in_metadata():
    result = {"title": "Echeveria cante in cultivation", "tags": []}
    assert explicit_name(result, ["Echeveria cante"])
    assert not explicit_name({"title": "pretty succulent", "tags": []}, ["Echeveria cante"])
    assert normalized("Ligustrum × vicaryi") == "ligustrum x vicaryi"


def test_smithsonian_delivery_is_explicitly_allowlisted_for_specimens():
    assert "ids.si.edu" in ALLOWED_HOSTS


def test_inaturalist_original_is_bounded_to_medium_download():
    url = "https://inaturalist-open-data.s3.amazonaws.com/photos/123/original.jpeg"
    assert download_variant(url).endswith("/123/medium.jpeg")
