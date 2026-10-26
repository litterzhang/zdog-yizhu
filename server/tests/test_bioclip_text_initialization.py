import pytest

from yizhu_server.bioclip_text_initialization import build


def test_text_initialization_refuses_existing_output(tmp_path):
    classes = tmp_path / "classes.json"
    classes.write_text('{"classes": [{"speciesId": "a"}, {"speciesId": "b"}]}')
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="new text-initialization"):
        build(classes, output)
