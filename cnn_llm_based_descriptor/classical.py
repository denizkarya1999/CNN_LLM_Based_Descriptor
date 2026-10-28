"""Transparent label associations, exact probability inference and search.

No image pixels or language model are used. All eligible YOLO boxes are
assessed, including custom labels and low-score predictions. The knowledge
base and conditional probabilities are illustrative, not calibrated accuracy.
"""

import json
import math
from pathlib import Path

from .grounding import LOW_SCORE, action_choices, render_description
from .methods import DEFAULT_SEARCH, method_details
from .scenes import SETTINGS, environment_text
from .search import plan_reviews
from .explanations import search_explanation, probability_explanation

MODEL_PATH = Path(__file__).with_name("scene_model.json")


def load_scene_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    labels, scenes = model["labels"], model["scenes"]
    if len(labels) != len(set(labels)) or set(scenes) != set(SETTINGS):
        raise ValueError("Knowledge base needs unique labels and every supported setting.")
    if set(model["label_actions"]) != set(labels):
        raise ValueError("Every knowledge-base label needs a decision association.")
    for data in scenes.values():
        if set(data["likelihoods"]) != set(labels):
            raise ValueError("Every setting must define likelihoods for every knowledge-base label.")
        for value in (data["prior"], data["default_likelihood"], *data["likelihoods"].values()):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("Scene priors and likelihoods must be strictly between zero and one.")
    if not math.isclose(sum(data["prior"] for data in scenes.values()), 1, abs_tol=1e-8):
        raise ValueError("Scene priors must sum to one.")
    if len({data["default_likelihood"] for data in scenes.values()}) != 1:
        raise ValueError("Custom labels need equal neutral likelihoods across settings.")
    if not 0 <= model["low_score_weight"] <= 1:
        raise ValueError("Low-score evidence weight must be between zero and one.")
    review = model["review_model"]
    if not 0 < review["threshold"] < 1 or any(not 0 <= p <= 1 for p in review["probabilities"].values()):
        raise ValueError("Review probabilities and threshold must be valid probabilities.")
    if set(review["probabilities"]) != {"no_action", "verify", "check_signal", "handle_carefully"}:
        raise ValueError("Unsupported review probability table.")
    if review["probabilities"]["no_action"] >= review["threshold"]:
        raise ValueError("Presence-only review probability must stay below the recommendation threshold.")
    if not set(model["label_actions"].values()) <= {"no_action", "check_signal", "handle_carefully"}:
        raise ValueError("Unsupported label action in knowledge base.")
    # Only the matching detected classes may receive these specific suggestions.
    for label, kind in model["label_actions"].items():
        if kind == "check_signal" and label not in {"traffic light", "stop sign"}:
            raise ValueError("Signal advice requires a signal label.")
        if kind == "handle_carefully" and label not in {"knife", "scissors"}:
            raise ValueError("Handling advice requires a supported detected label.")
    return model


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


def assess_detections(evidence, model):
    """Every eligible box gets a decision association, even if it needs no action."""
    assessments = []
    for item in evidence:
        if "box_xyxy_pixels" not in item:
            continue
        label = item["label"].casefold()
        kind = "verify" if item["confidence"] < LOW_SCORE else model["label_actions"].get(label, "no_action")
        rule = {"verify": "low_score", "check_signal": "signal_state_unconfirmed",
                "handle_carefully": "care_if_handled", "no_action": "presence_only"}[kind]
        assessments.append({"detection_id": item["id"], "label": item["label"], "kind": kind,
                            "rule": rule, "confidence": item["confidence"],
                            "knowledge_base_label": label in model["label_actions"]})
    return assessments


def review_targets(evidence, assessments):
    offered, by_id = action_choices(evidence), {item["id"]: item for item in evidence}
    targets = []
    for assessment in assessments:
        kind, ident = assessment["kind"], assessment["detection_id"]
        if kind == "no_action":
            continue
        item = by_id[ident]
        kinds = ((kind, 1.2 if kind == "handle_carefully" else 1.0),)
        if kind != "verify":
            kinds += (("inspect", 1.6),)
        options = [(f"{action}:{ident}", cost) for action, cost in kinds if f"{action}:{ident}" in offered]
        if options:
            x1, y1, x2, y2 = item["box_xyxy_normalized"]
            targets.append({"id": ident, "rule": assessment["rule"], "options": options,
                            "center": ((x1+x2)/2, (y1+y2)/2)})
    return targets


def search_review_plan(evidence, algorithm=DEFAULT_SEARCH, model=None):
    model = model or load_scene_model()
    assessments = assess_detections(evidence, model)
    result = plan_reviews(review_targets(evidence, assessments), algorithm)
    result["assessments"] = assessments
    return result


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


class ClassicalClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_scene_model()

    def describe(self, frame, records):
        from .descriptor import detection_evidence
        height, width = frame.shape[:2]
        classification = self.cfg.task == "Image classification"
        # Only LLM prompts have an object budget. Classical methods use all
        # predictions returned by YOLO after its confidence/class/NMS filters.
        evidence, total = detection_evidence(records, width, height, None,
                                            self.cfg.confidence, require_boxes=not classification)
        details = method_details(self.cfg)
        if classification or not evidence:
            result = render_description(evidence, total, width, height, self.cfg,
                                        status="classification" if classification else "no_detections")
            operation = "Search traversal" if self.cfg.describe_backend == "Search Algorithm" else "Probability calculation"
            result["text"] += f"\n\nDescription ({self.cfg.describe_backend}): {operation} was not performed because no eligible object-box evidence was available."
        else:
            if self.cfg.describe_backend == "Search Algorithm":
                scene = {"setting": "unclear", "custom_labels": []}
                plan = search_review_plan(evidence, self.cfg.describe_search, self.model)
                step = plan["steps"][0]["choice"] if plan["steps"] else "no_action"
                source = "knowledge-base"
                note = (f"{self.cfg.describe_search} assessed all {len(evidence)} detections and planned "
                        f"{len(plan['steps'])} review steps in {len(plan['batches'])} batch(es).")
                if plan["steps"]:
                    note += " The first suggestion is shown as the decision."
                if not plan["steps"]:
                    note += " No review rule was triggered by the detected labels and scores."
                extra = {"decision_search": plan}
            else:
                scene = infer_setting(evidence, self.model)
                decision = infer_review(assess_detections(evidence, self.model), self.model)
                step, source = decision["next_step"], "probability model"
                note = (f"Conditional review probabilities were evaluated for all {len(evidence)} detections. "
                        f"Highest review probability: {decision['selected_probability']*100:.0f}%; "
                        f"recommendation threshold: {decision['threshold']*100:.0f}%. "
                        "These are illustrative policy probabilities, not measured accuracy or danger.")
                extra = {"scene_inference": scene, "decision_inference": decision}
            setting = scene["setting"]
            environment = SETTINGS[setting][1]
            context = environment_text(environment, setting, source)
            if source == "probability model":
                ranking = "; ".join(f"{SETTINGS[name][0]} {scene['posterior'][name]*100:.1f}%"
                                    for name in scene["candidates"] if scene["posterior"][name] >= .01)
                context += f"\nPossible settings: {ranking}.\nIllustrative model probabilities, not measured accuracy."
            if scene["custom_labels"]:
                context += " Custom labels without scene associations use a neutral fallback."
            choice = {"environment": environment, "setting": setting, "next_step": step}
            result = render_description(evidence, total, width, height, self.cfg, choice=choice,
                                        context_override=context, reason_note=note)
            result.update(extra)
            reasoning = (search_explanation(plan, evidence) if self.cfg.describe_backend == "Search Algorithm"
                         else probability_explanation(scene, decision))
            result["reasoning_trace"] = reasoning
            result["text"] += "\n\n" + reasoning
            result["efficiency"] = ({"expanded_states": plan["expanded_states"],
                                     "generated_states": plan["generated_states"],
                                     "peak_frontier_nodes": plan["peak_frontier"],
                                     "planned_reviews": len(plan["steps"])}
                                    if self.cfg.describe_backend == "Search Algorithm" else
                                    {"scene_hypotheses": len(scene["posterior"]),
                                     "likelihood_factors": sum(map(len, scene["likelihood_factors"].values())),
                                     "review_assessments": len(decision["assessments"])})
        result.update(model=details["algorithm"], backend=self.cfg.describe_backend, **details,
                      detections_considered=len(evidence), knowledge_base_labels=len(self.model["labels"]))
        return result
