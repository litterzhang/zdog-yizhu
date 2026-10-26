"""Score final pipeline responses; no inference, model loading, threshold tuning or deployment."""

from yizhu_server.plantnet_transfer import summarize as summarize_plant_cases


def score_responses(samples, responses, classes, requested_ids):
    """Missing responses are failures; never substitute internal logits for returned candidates.

    samples: sampleId, isPlant, speciesId (required for plant images).
    responses: sampleId, technicalSuccess, gatePassed, candidateStage='final-returned',
               candidates (ordered speciesId strings; zero to five).
    """
    requested = set(requested_ids)
    class_ids = {r["speciesId"] for r in classes}
    if (
        not samples or not requested or len(requested) != len(requested_ids)
        or len(class_ids) != len(classes) or not class_ids <= requested
        or any(not isinstance(sid, str) or not sid for sid in requested)
    ):
        raise ValueError("Require unique trained/target identities and nonempty benchmark.")
    truths = {}
    for row in samples:
        identifier = row.get("sampleId")
        if (
            not isinstance(identifier, str) or not identifier or identifier in truths
            or type(row.get("isPlant")) is not bool
            or (row["isPlant"] and (
                not isinstance(row.get("speciesId"), str) or not row["speciesId"]
            ))
            or (not row["isPlant"] and row.get("speciesId") is not None)
        ):
            raise ValueError("Benchmark labels must be explicit and sample IDs unique.")
        truths[identifier] = row
    outputs = {}
    for row in responses:
        identifier = row.get("sampleId")
        candidates = row.get("candidates")
        if (
            identifier not in truths or identifier in outputs
            or type(row.get("technicalSuccess")) is not bool
            or row.get("candidateStage") != "final-returned"
            or not isinstance(candidates, list)
            or any(not isinstance(sid, str) or sid not in class_ids for sid in candidates)
            or len(candidates) > 5 or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("Require unique final responses with only trained candidate IDs.")
        if row["technicalSuccess"]:
            if type(row.get("gatePassed")) is not bool:
                raise ValueError("Successful response requires an actual binary gate decision.")
            if not row["gatePassed"] and candidates:
                raise ValueError("Rejected plant image must not contain classification candidates.")
        elif candidates:
            raise ValueError("Failed response cannot reuse stale classification candidates.")
        outputs[identifier] = row
    plant_cases, nonplants, outside_targets, missing = [], [], [], []
    for identifier, truth in truths.items():
        response = outputs.get(identifier)
        if response is None:
            missing.append(identifier)
            response = {"technicalSuccess": False, "gatePassed": None, "candidates": []}
        success = response["technicalSuccess"]
        gate = success and response.get("gatePassed") is True
        candidates = response["candidates"] if success else []
        entry = {
            "sampleId": identifier, "technicalSuccess": success, "gatePassed": gate,
            "returnedCandidates": candidates,
        }
        if not truth["isPlant"]:
            nonplants.append(entry)
        elif truth["speciesId"] not in requested:
            outside_targets.append(entry)
        else:
            sid = truth["speciesId"]
            plant_cases.append({
                **entry, "speciesId": sid, "top1Hit": bool(candidates) and candidates[0] == sid,
                "top5Hit": sid in candidates,
            })
    if not plant_cases:
        raise ValueError("At least one in-scope plant case is required for species accuracy.")
    result = summarize_plant_cases(plant_cases, classes, requested_ids)
    # Gate-rejected images have no returned species candidates. The independent,
    # ungated species classifier accuracy cannot be reconstructed from these responses.
    for metrics in [result, result["coveredSpeciesOnly"], *result["perSpecies"].values()]:
        if metrics is not None:
            metrics.pop("speciesTop1", None)
            metrics.pop("speciesTop5", None)
    result.pop("macroSpeciesTop5", None)
    successful_nonplants = sum(r["technicalSuccess"] for r in nonplants)
    false_positives = sum(r["technicalSuccess"] and r["gatePassed"] for r in nonplants)
    result.update({
        "benchmarkSampleCount": len(samples),
        "inScopePlantImages": len(plant_cases),
        "inScopePlantGateRejections": sum(
            r["technicalSuccess"] and not r["gatePassed"] for r in plant_cases
        ),
        "missingResponseIds": missing,
        "nonplant": {
            "images": len(nonplants), "successfulResponses": successful_nonplants,
            "technicalFailures": len(nonplants) - successful_nonplants,
            "falsePositive": false_positives,
            "trueNegative": sum(r["technicalSuccess"] and not r["gatePassed"] for r in nonplants),
            "falsePositiveRateAmongSuccessful": (
                false_positives / successful_nonplants if successful_nonplants else None
            ),
        },
        "outOfScopePlants": {
            "images": len(outside_targets),
            "technicalFailures": sum(not r["technicalSuccess"] for r in outside_targets),
            "returnedNoSpecies": sum(
                r["technicalSuccess"] and not r["returnedCandidates"] for r in outside_targets
            ),
            "returnedKnownSpecies": sum(
                r["technicalSuccess"] and bool(r["returnedCandidates"]) for r in outside_targets
            ),
        },
        "candidateStage": "final-returned", "metricsOnly": True,
        "ungatedSpeciesMetricsAvailable": False,
        "goalAchieved": False, "productionApproved": False,
        "independentFinalBenchmark": False,
        "scopeNote": "In-scope plant denominator includes untrained targets, gate rejections "
                     "and missing/failed responses. Explicit outside-target species are "
                     "reported separately, not silently merged into the supported-only score.",
    })
    return result
