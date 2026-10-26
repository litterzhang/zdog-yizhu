import pytest

from yizhu_server.random_init_classifier import build_random_classifier


@pytest.mark.parametrize("architecture", ["resnet18", "resnet50"])
def test_scratch_factory_never_requests_pretrained_weights(monkeypatch, architecture):
  torch = pytest.importorskip("torch")
  from torchvision.models._api import WeightsEnum

  def forbidden(*args, **kwargs):
    raise AssertionError("Pretrained checkpoint loading is forbidden.")

  monkeypatch.setattr(WeightsEnum, "get_state_dict", forbidden)
  monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
  model = build_random_classifier(162, architecture=architecture)
  assert model.fc.out_features == 162
  assert model.training
  assert all(parameter.requires_grad for parameter in model.parameters())
  assert model.conv1.weight.device.type == "cpu"


def test_random_initialization_is_reproducible_and_backbone_learns():
  torch = pytest.importorskip("torch")
  torch.set_num_threads(1)
  rng = torch.random.get_rng_state().clone()
  model = build_random_classifier(3, seed=24)
  assert torch.equal(torch.random.get_rng_state(), rng)
  again = build_random_classifier(3, seed=24)
  assert torch.equal(model.conv1.weight, again.conv1.weight)
  different = build_random_classifier(3, seed=25)
  assert not torch.equal(model.conv1.weight, different.conv1.weight)
  optimizer = torch.optim.SGD(model.parameters(), lr=.01)
  original = model.conv1.weight.detach().clone()
  logits = model(torch.randn(2, 3, 64, 64))
  assert logits.shape == (2, 3)
  torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
  assert model.conv1.weight.grad is not None and model.conv1.weight.grad.abs().sum() > 0
  optimizer.step()
  assert not torch.equal(model.conv1.weight, original)


@pytest.mark.parametrize("kwargs", [
  {"class_count": True}, {"class_count": 1}, {"class_count": 4097},
  {"class_count": 2, "architecture": "pretrained-resnet50"},
  {"class_count": 2, "seed": -1}, {"class_count": 2, "seed": True},
])
def test_random_factory_rejects_invalid_settings(kwargs):
  with pytest.raises(ValueError):
    build_random_classifier(**kwargs)
