"""Offline architecture factory: no pretrained weights, cached features, or deployment."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import torch

ARCHITECTURES = ("resnet18", "resnet50")
DEFAULT_ARCHITECTURE = "resnet18"


def build_random_classifier(
  class_count: int, *, architecture: str = DEFAULT_ARCHITECTURE, seed: int = 20260924,
) -> "torch.nn.Module":
  """Build a fully trainable CPU model; output width is not a claim of trained support."""
  if type(class_count) is not int or not 2 <= class_count <= 4096:
    raise ValueError("Require 2–4096 output classes.")
  if architecture not in ARCHITECTURES:
    raise ValueError("Use a registered random-initialization architecture.")
  if type(seed) is not int or not 0 <= seed < 2**63:
    raise ValueError("Use a nonnegative integer seed below 2**63.")

  import torch
  from torchvision import models

  builder = {"resnet18": models.resnet18, "resnet50": models.resnet50}[architecture]
  # This deliberately does not call the transfer-learning encoders or accept a
  # checkpoint argument. Only the architecture is reused, not learned parameters.
  with torch.random.fork_rng(), torch.device("cpu"):
    torch.manual_seed(seed)
    model = builder(weights=None, num_classes=class_count)
  model.requires_grad_(True)
  model.train()
  return model
