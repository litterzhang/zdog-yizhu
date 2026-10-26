import copy
import io
import json

import pytest
from PIL import Image

from yizhu_server import training_depth_collection as depth
from yizhu_server.training_data import group_split, observer_group


def fixture_data():
  user = next(i for i in range(1, 100) if group_split(observer_group(i)) == "train")
  lead = {
    "sampleId": "inat:123:456", "observationId": 123, "photoId": 456,
    "speciesId": "test:species", "observerGroup": observer_group(user), "split": "train",
    "species": {"speciesId": "test:species", "scientificName": "Synthetic plant",
                "inatTaxonId": 789},
  }
  observation = {
    "id": 123, "quality_grade": "research", "user": {"id": user},
    "taxon": {"id": 789, "name": "Synthetic plant", "rank": "species"},
    "photos": [{
      "id": 456, "license_code": "cc-by", "attribution": "Fixture author",
      "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/456/square.jpg",
    }],
  }
  return lead, observation


@pytest.mark.parametrize("change", [
  "license", "grade", "taxon", "name", "rank", "author", "photo", "id", "host",
])
def test_refresh_rejects_changed_or_unauthorized_source(change):
  lead, observation = fixture_data()
  assert depth.refresh_candidate({"results": [observation]}, lead) is not None
  if change == "license":
    observation["photos"][0]["license_code"] = "cc-by-nc"
  elif change == "grade":
    observation["quality_grade"] = "casual"
  elif change == "taxon":
    observation["taxon"]["id"] += 1
  elif change == "name":
    observation["taxon"]["name"] = "Wrong species"
  elif change == "rank":
    observation["taxon"]["rank"] = "genus"
  elif change == "author":
    observation["user"]["id"] += 1
  elif change == "photo":
    observation["photos"][0]["id"] += 1
  elif change == "id":
    observation["id"] += 1
  elif change == "host":
    observation["photos"][0]["url"] = "https://example.org/456/square.jpg"
  assert depth.refresh_candidate({"results": [observation]}, lead) is None


def test_refresh_projects_only_public_provenance_and_no_location():
  lead, observation = fixture_data()
  observation.update(location="31.1,121.1", positional_accuracy=1, description="private note")
  row = depth.refresh_candidate({"results": [observation]}, lead)
  assert row["license"] == "cc-by"
  assert row["imageUrl"] == (
    "https://inaturalist-open-data.s3.amazonaws.com/photos/456/medium.jpg"
  )
  assert set(row).isdisjoint({"location", "positional_accuracy", "description"})


class FakeClient:
  def __init__(self, observation, *, blocked=False):
    self.observation, self.blocked, self.calls = observation, blocked, []

  def json(self, endpoint, params):
    self.calls.append(endpoint)
    if self.blocked:
      raise PermissionError("Public source paused: HTTP 429")
    return {"results": [self.observation]}

  def get(self, url, **kwargs):
    self.calls.append(url)
    image = Image.new("RGB", (120, 130))
    for x in range(120):
      for y in range(130):
        image.putpixel((x, y), ((x * 17) % 255, (y * 13) % 255, (x + y) % 255))
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def make_plan(tmp_path, *, baseline=None):
  lead, observation = fixture_data()
  plan = {
    "policy": depth.POLICY, "classes": [{"speciesId": lead["speciesId"]}],
    "candidates": [lead], "baseline": baseline or [],
  }
  path = tmp_path / "plan.json"
  path.write_text(json.dumps(plan))
  return path, observation


def test_collect_fresh_source_and_hold_for_consolidated_audit(tmp_path, monkeypatch):
  monkeypatch.setattr(depth, "reserve_space", lambda *args: None)
  plan, observation = make_plan(tmp_path)
  client = FakeClient(observation)
  output = tmp_path / "output"
  depth.collect(plan, output, client=client)
  row = json.loads((output / "samples/456.json").read_text())["samples"][0]
  assert row["split"] == "train"
  assert row["trainingEligible"] is False
  assert row["automatedSourceChecksPassed"] is True
  assert row["sourceQualityGrade"] == "research"
  assert len(client.calls) == 2
  progress = json.loads((output / "progress.json").read_text())
  assert progress["downloadedImages"] == 1
  assert progress["trainingImagesAdmitted"] == 0
  assert progress["stage"] == "collection-pass-complete"
  assert not (output / "images/current.download").exists()
  with pytest.raises(ValueError, match="implicit resume"):
    depth.collect(plan, output, client=client)


def test_429_stops_durably_without_retry_or_bypass(tmp_path, monkeypatch):
  monkeypatch.setattr(depth, "reserve_space", lambda *args: None)
  plan, observation = make_plan(tmp_path)
  output = tmp_path / "output"
  client = FakeClient(observation, blocked=True)
  with pytest.raises(PermissionError):
    depth.collect(plan, output, client=client)
  assert len(client.calls) == 1
  assert json.loads((output / "stopped.json").read_text())["automaticRetry"] is False
  assert json.loads((output / "progress.json").read_text())["stage"] == "stopped"


def test_existing_observation_is_not_redownloaded(tmp_path, monkeypatch):
  monkeypatch.setattr(depth, "reserve_space", lambda *args: None)
  lead, _ = fixture_data()
  baseline = [{**lead, "imageSha256": "a" * 64, "differenceHash": "0" * 16}]
  plan, observation = make_plan(tmp_path, baseline=baseline)
  client = FakeClient(observation)
  depth.collect(plan, tmp_path / "output", client=client)
  assert client.calls == []


def test_author_cap_includes_existing_dataset(tmp_path, monkeypatch):
  monkeypatch.setattr(depth, "reserve_space", lambda *args: None)
  lead, _ = fixture_data()
  baseline = []
  for i in range(3):
    row = copy.deepcopy(lead)
    row.update(photoId=i, observationId=i, imageSha256=str(i) * 64, differenceHash="0" * 16)
    baseline.append(row)
  plan, observation = make_plan(tmp_path, baseline=baseline)
  client = FakeClient(observation)
  depth.collect(plan, tmp_path / "output", client=client)
  assert client.calls == []
