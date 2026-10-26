"""Synthetic checklist rows for tests; never used by production import."""

import pytest

from yizhu_server.colchina_names import build_national_index


@pytest.fixture
def national_names():
    return build_national_index([
        {"rowNumber": index + 2, "界拉丁名": "Plantae",
         "物种拉丁名": scientific, "物种中文名": chinese, "审核专家/数据源": "合成测试资料"}
        for index, (scientific, chinese) in enumerate([
            ("Ginkgo biloba", "银杏"), ("Jasminum sambac", "茉莉"),
            ("Oxalis corniculata", "酢浆草"),
        ])
    ])
