import copy

import pytest

from yizhu_server.trained_species_contract import (
  build_trained_contract,
  rank_trained_candidates,
  validate_trained_contract,
)


def build(**changes):
  kwargs = {
    "classes": [
      {"speciesId": "own:1", "scientificName": "Example one", "chineseName": "合成甲"},
      {"speciesId": "own:2", "scientificName": "Example two", "chineseName": "合成乙"},
    ],
    "target_species_ids": ["own:1", "own:2", "planned:3"],
    "training_examples": {"own:1": 70, "own:2": 70},
    "completed_epochs": 2, "backbone_optimizer_steps": 20,
    "checkpoint_sha256": "a" * 64,
  }
  kwargs.update(changes)
  return build_trained_contract(**kwargs)


def test_output_never_includes_planned_but_untrained_species():
  torch = pytest.importorskip("torch")
  contract = build()
  assert validate_trained_contract(contract) == contract
  results = rank_trained_candidates(torch.tensor([2., 1.]), contract)
  assert [row["speciesId"] for row in results] == ["own:1", "own:2"]
  assert len(results) == 2  # No filling to five candidates from a global/pretrained list.
  assert "planned:3" not in {row["speciesId"] for row in results}
  with pytest.raises(ValueError, match="exactly"):
    rank_trained_candidates(torch.tensor([2., 1., 100.]), contract)


def test_self_trained_species_outside_user_targets_is_still_forbidden():
  rows = [
    {"speciesId": "own:1", "scientificName": "Example one"},
    {"speciesId": "old:horse", "scientificName": "Aesculus hippocastanum"},
  ]
  with pytest.raises(ValueError, match="outside"):
    build(classes=rows, training_examples={"own:1": 70, "old:horse": 70})


@pytest.mark.parametrize("changes", [
  {"completed_epochs": 0}, {"completed_epochs": True},
  {"backbone_optimizer_steps": 0},
  {"initialization": "pretrained"},
  {"checkpoint_sha256": "not-a-checkpoint"},
  {"training_examples": {"own:1": 70, "own:2": 0}},
  {"training_examples": {"own:1": 70, "own:2": True}},
  {"training_examples": {"own:1": 70, "own:2": 70, "planned:3": 0}},
])
def test_incomplete_or_inherited_training_cannot_publish_support(changes):
  with pytest.raises(ValueError):
    build(**changes)


@pytest.mark.parametrize("role", ["retained-pretrained", "finetune-addition"])
def test_old_solution_classes_cannot_be_reused_as_clean_training_proof(role):
  classes = [
    {"speciesId": "own:1", "scientificName": "Example one", "classRole": role},
    {"speciesId": "own:2", "scientificName": "Example two"},
  ]
  with pytest.raises(ValueError, match="inherited"):
    build(classes=classes)


def test_contract_tampering_is_rejected():
  contract = copy.deepcopy(build())
  contract["externalSpeciesFallbackAllowed"] = True
  with pytest.raises(ValueError, match="changed"):
    validate_trained_contract(contract)
