"""Exact scene enumeration and maximum-probability review selection."""
import math
from ..shared.grounding import LOW_SCORE


def class_observations(evidence):
    """Use every class once at its highest returned confidence; retain all boxes elsewhere."""
    observations = {}
    for item in evidence:
        if "box_xyxy_pixels" in item:
            label = item["label"].casefold()
            observations[label] = max(observations.get(label, 0), item["confidence"])
    return observations


def label_likelihood(data, label):
    return data["likelihoods"].get(label, data["default_likelihood"])


def infer_setting(evidence, model):
    """Enumerate P(S,E) and normalize. Every observed label participates.

    Each label contributes a positive observation likelihood; absent labels
    remain unobserved. Low-score observations use a weaker, explicit CPT mixture
    with the prior-predictive mean, not the YOLO score as a probability of truth.
    """
    scenes, observations = model["scenes"], class_observations(evidence)
    means = {label: sum(row["prior"] * label_likelihood(row, label) for row in scenes.values())
             for label in observations}
    log_joint, factors = {}, {}
    for setting, data in scenes.items():
        value = math.log(data["prior"])
        factors[setting] = {}
        for label, score in observations.items():
            weight = 1.0 if score >= LOW_SCORE else model["low_score_weight"]
            likelihood = weight * label_likelihood(data, label) + (1-weight) * means[label]
            factors[setting][label] = likelihood
            value += math.log(likelihood)
        log_joint[setting] = value
    offset = max(log_joint.values())
    weights = {name: math.exp(value-offset) for name, value in log_joint.items()}
    normalizer = sum(weights.values())
    posterior = {name: value/normalizer for name, value in weights.items()}
    ranked = sorted(posterior, key=lambda name: (-posterior[name], name))
    top, second = ranked[:2]
    informative = [label for label, score in observations.items() if score >= LOW_SCORE
                   and len({label_likelihood(row, label) for row in scenes.values()}) > 1]
    accepted = (len(informative) >= 2 and top != "unclear" and posterior[top] >= .55
                and posterior[top]-posterior[second] >= .15)
    return {"setting": top if accepted else "unclear", "posterior": posterior,
            "observed_labels": sorted(observations), "label_confidences": observations,
            "priors": {name: row["prior"] for name, row in scenes.items()},
            "likelihood_factors": factors,
            "raw_likelihoods": {name: {label: label_likelihood(row, label) for label in observations} for name, row in scenes.items()},
            "evidence_weights": {label: 1.0 if score >= LOW_SCORE else model["low_score_weight"] for label, score in observations.items()},
            "predictive_means": means, "log_joint": log_joint,
            "normalization_offset": offset, "scaled_normalizer": normalizer,
            "custom_labels": sorted(set(observations)-set(model["labels"])),
            "candidates": ranked[:3], "method": "Inference by Enumeration",
            "calibrated": False, "table_version": model["version"],
            "assumption": "Distinct observed labels are conditionally independent given the setting. Missing labels are unobserved. Custom labels have equal neutral likelihoods."}


def infer_review(assessments, model):
    """Conditional P(review useful | observed label, score band), then MAP choice.

    These hand-authored probabilities express a review policy, not object
    correctness or danger. All boxes are evaluated; presence-only stays below
    the recommendation threshold. No review planning/search is performed.
    """
    table = model["review_model"]
    outcomes = [{**item, "review_probability": table["probabilities"][item["kind"]]}
                for item in assessments]
    chosen = max(outcomes, key=lambda item: item["review_probability"], default=None)
    recommend = chosen is not None and chosen["review_probability"] >= table["threshold"]
    key = f"{chosen['kind']}:{chosen['detection_id']}" if recommend else "no_action"
    return {"next_step": key, "assessments": outcomes, "threshold": table["threshold"],
            "selected_probability": chosen["review_probability"] if chosen else None,
            "calibrated": False, "method": "Conditional probability / maximum-posterior review choice",
            "meaning": table["meaning"]}
