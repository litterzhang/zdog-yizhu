from copy import deepcopy

import pytest

from yizhu_server.end_to_end_evaluation import score_responses

CLASSES = [{"speciesId": "a"}, {"speciesId": "b"}]


def sample(identifier, species="a"):
    return {"sampleId": identifier, "isPlant": species is not None, "speciesId": species}


def response(identifier, candidates=None, *, passed=True, success=True):
    return {
        "sampleId": identifier, "technicalSuccess": success, "gatePassed": passed,
        "candidateStage": "final-returned",
        "candidates": ["a"] if candidates is None else candidates,
    }


def test_failures_rejections_and_untrained_targets_stay_in_denominator():
    samples = [sample("hit"), sample("rejected"), sample("missing"), sample("untrained", "c")]
    outputs = [response("hit"), response("rejected", [], passed=False),
               response("untrained", ["a", "b"])]
    result = score_responses(samples, outputs, CLASSES, ["a", "b", "c"])
    assert result["inScopePlantImages"] == 4
    assert result["endToEndTop5"] == .25
    assert result["technicalFailures"] == 1
    assert result["inScopePlantGateRejections"] == 1
    assert result["uncoveredTestPhotos"] == 1
    assert result["missingResponseIds"] == ["missing"]
    assert result["fullTargetCoverage"] is False
    assert "speciesTop5" not in result
    assert result["ungatedSpeciesMetricsAvailable"] is False


def test_only_final_returned_candidates_count_after_truncation():
    samples = [sample("a"), sample("b", "b")]
    outputs = [response("a", ["b"]), response("b", ["b"])]
    outputs[0]["internalTop5"] = ["b", "a"]
    result = score_responses(samples, outputs, CLASSES, ["a", "b"])
    assert result["endToEndTop5"] == .5


def test_nonplants_and_outside_target_plants_are_explicit_not_silent_drops():
    samples = [sample("p"), sample("n1", None), sample("n2", None), sample("n3", None),
               sample("outside", "outside-scope")]
    outputs = [response("p"), response("n1", ["a"]), response("n2", [], passed=False),
               response("outside", [])]
    result = score_responses(samples, outputs, CLASSES, ["a", "b"])
    assert result["benchmarkSampleCount"] == 5
    assert result["nonplant"] == {
        "images": 3, "successfulResponses": 2, "technicalFailures": 1,
        "falsePositive": 1, "trueNegative": 1, "falsePositiveRateAmongSuccessful": .5,
    }
    assert result["outOfScopePlants"]["images"] == 1
    assert result["outOfScopePlants"]["returnedNoSpecies"] == 1


def test_perfect_synthetic_metrics_do_not_claim_independent_acceptance():
    result = score_responses([sample("a"), sample("b", "b")],
                             [response("a"), response("b", ["b"])], CLASSES, ["a", "b"])
    assert result["endToEndTop5"] == 1 and result["fullTargetCoverage"]
    assert result["goalAchieved"] is False and result["productionApproved"] is False
    assert result["independentFinalBenchmark"] is False


@pytest.mark.parametrize("mutation", [
    {"candidates": ["outside"]}, {"candidates": ["a", "a"]},
    {"candidateStage": "internal-logits"}, {"technicalSuccess": "true"},
    {"gatePassed": None}, {"gatePassed": False}, {"technicalSuccess": False},
])
def test_invalid_or_untrained_outputs_are_not_accepted(mutation):
    row = {**response("a"), **mutation}
    with pytest.raises(ValueError):
        score_responses([sample("a")], [row], CLASSES, ["a", "b"])


def test_duplicate_extra_and_unlabelled_records_are_rejected():
    with pytest.raises(ValueError):
        score_responses([sample("a"), sample("a")], [], CLASSES, ["a", "b"])
    with pytest.raises(ValueError):
        score_responses([sample("a")], [response("a"), response("a")], CLASSES, ["a", "b"])
    with pytest.raises(ValueError):
        score_responses([sample("a")], [response("other")], CLASSES, ["a", "b"])
    with pytest.raises(ValueError):
        score_responses([{"sampleId": "a", "isPlant": True}], [], CLASSES, ["a", "b"])


def test_metric_adapter_does_not_mutate_inputs():
    args = ([sample("a")], [response("a")], CLASSES, ["a", "b"])
    before = deepcopy(args)
    score_responses(*args)
    assert args == before
