"""CIS 579 alternative: conditional probability, rules, and uniform-cost search.

This is an explicit teaching model, not a learned scene classifier. The table's
priors/likelihoods and the search costs are illustrative and uncalibrated.
"""

import heapq
import itertools
import json
import math
from pathlib import Path

from .grounding import LOW_SCORE, action_choices, render_description
from .scenes import SETTINGS, environment_text

MODEL_PATH = Path(__file__).with_name("scene_model.json")
MAX_REVIEW_TARGETS = 8


def load_scene_model(path=MODEL_PATH):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = model["scenes"]
    if set(scenes) != set(SETTINGS):
        raise ValueError("Scene table must include every supported setting, including unclear.")
    for data in scenes.values():
        for value in (data["prior"], data["default_likelihood"], *data["likelihoods"].values()):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("Scene priors and likelihoods must be strictly between zero and one.")
    if not math.isclose(sum(data["prior"] for data in scenes.values()), 1, abs_tol=1e-8):
        raise ValueError("Scene priors must sum to one.")
    return model


def infer_setting(evidence, model):
    """Enumerate P(S,E), then normalize to P(S|E) in log space.

Only distinct, sufficiently strong, supported positive class observations are
conditioned on. Missing classes are unobserved, not negative evidence. The
conditional independence assumption defines the illustrative joint model.
YOLO confidence gates observations; it is never substituted for P(E|S).
"""
    scenes = model["scenes"]
    supported = set().union(*(set(data["likelihoods"]) for data in scenes.values()))
    labels = sorted({item["label"].casefold() for item in evidence
                     if "box_xyxy_pixels" in item and item["confidence"] >= LOW_SCORE
                     and item["label"].casefold() in supported})
    log_joint = {}
    for setting, data in scenes.items():
        log_joint[setting] = math.log(data["prior"]) + sum(
            math.log(data["likelihoods"].get(label, data["default_likelihood"]))
            for label in labels)
    offset = max(log_joint.values())
    weights = {name: math.exp(value - offset) for name, value in log_joint.items()}
    normalizer = sum(weights.values())
    posterior = {name: value / normalizer for name, value in weights.items()}
    ranked = sorted(posterior, key=lambda name: (-posterior[name], name))
    top, second = ranked[:2]
    accepted = (len(labels) >= 2 and top != "unclear" and posterior[top] >= .55
                and posterior[top] - posterior[second] >= .15)
    return {
        "setting": top if accepted else "unclear", "posterior": posterior,
        "observed_labels": labels, "candidates": ranked[:3],
        "method": "Bayes rule / exact enumeration of scene hypotheses",
        "calibrated": False, "table_version": model["version"],
        "assumption": "Distinct observed classes are conditionally independent given the setting; missing classes are unobserved.",
    }


def review_targets(evidence):
    """Simple implication rules create review goals from current YOLO facts."""
    targets = []
    offered = action_choices(evidence)
    for item in evidence:
        if "box_xyxy_normalized" not in item:
            continue
        if item["confidence"] < LOW_SCORE:
            rule, kinds = "low_score", (("verify", 1.0),)
        elif item["label"].casefold() in {"traffic light", "stop sign"}:
            rule, kinds = "signal_state_unconfirmed", (("check_signal", 1.0), ("inspect", 1.6))
        elif item["label"].casefold() in {"knife", "scissors"}:
            rule, kinds = "care_if_handled", (("handle_carefully", 1.2), ("inspect", 1.6))
        else:
            continue
        options = [(f"{kind}:{item['id']}", cost) for kind, cost in kinds
                   if f"{kind}:{item['id']}" in offered]
        if options:
            x1, y1, x2, y2 = item["box_xyxy_normalized"]
            targets.append({"id": item["id"], "rule": rule, "options": options,
                            "center": ((x1+x2)/2, (y1+y2)/2)})
    # Input is already confidence-sorted. Bound search time without claiming
    # that the omitted objects have been reviewed.
    return targets[:MAX_REVIEW_TARGETS], max(0, len(targets)-MAX_REVIEW_TARGETS)


def search_review_plan(evidence):
    """UCS over (planned target bitmask, last image location).

Goal: a conditional review suggestion for every selected target. Edge cost is
illustrative review effort + normalized Manhattan image-attention shift.
This plans advice; it never executes an action or plans physical movement.
"""
    targets, omitted = review_targets(evidence)
    goal = (1 << len(targets)) - 1
    counter = itertools.count()
    frontier = [(0.0, next(counter), 0, -1, [])]
    best = {(0, -1): 0.0}
    expanded = 0
    while frontier:
        cost, _, mask, previous, path = heapq.heappop(frontier)
        if cost > best[(mask, previous)]:
            continue
        expanded += 1
        if mask == goal:
            return {"algorithm": "Uniform-cost search", "steps": path,
                    "cost": cost, "expanded_states": expanded,
                    "targets": [t["id"] for t in targets], "omitted_targets": omitted,
                    "goal": "Plan a review suggestion for each selected target; no actions executed.",
                    "cost_basis": "Illustrative effort plus attention shift between normalized image box centers; not distance or risk."}
        origin = (.5, .5) if previous == -1 else targets[previous]["center"]
        for index, target in enumerate(targets):
            if mask & (1 << index):
                continue
            shift = sum(abs(a-b) for a, b in zip(origin, target["center"]))
            for key, effort in target["options"]:
                next_cost = cost + effort + shift
                state = (mask | (1 << index), index)
                if next_cost < best.get(state, math.inf):
                    best[state] = next_cost
                    step = {"choice": key, "detection_id": target["id"], "rule": target["rule"],
                            "cost": effort + shift}
                    heapq.heappush(frontier, (next_cost, next(counter), *state, path + [step]))
    raise RuntimeError("No review plan found for the supplied choices.")


class ClassicalClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_scene_model()

    def describe(self, frame, records):
        # Import here to avoid a circular import; no server/model call is made.
        from .descriptor import detection_evidence
        height, width = frame.shape[:2]
        classification = self.cfg.task == "Image classification"
        evidence, total = detection_evidence(records, width, height, self.cfg.describe_max_objects,
                                            self.cfg.confidence, require_boxes=not classification)
        if classification or not evidence:
            result = render_description(evidence, total, width, height, self.cfg,
                                        status="classification" if classification else "no_detections")
        else:
            inference = infer_setting(evidence, self.model)
            setting = inference["setting"]
            environment = SETTINGS[setting][1]
            plan = search_review_plan(evidence)
            choice = {"environment": environment, "setting": setting,
                      "next_step": plan["steps"][0]["choice"] if plan["steps"] else "no_action"}
            context = environment_text(environment, setting, "probability model")
            if inference["observed_labels"]:
                ranking = "; ".join(f"{SETTINGS[name][0]} {inference['posterior'][name]*100:.1f}%"
                                    for name in inference["candidates"] if inference["posterior"][name] >= .01)
                context += f"\nPossible settings: {ranking}.\nIllustrative model probabilities, not measured accuracy."
            else:
                context += " No supported scene cues above 50% detector score."
            note = (f"CIS 579 rules triggered {len(plan['targets'])} review goal(s). Uniform-cost search chose "
                    f"a {len(plan['steps'])}-step plan with illustrative cost {plan['cost']:.2f}; "
                    "the first suggestion is shown. This is a review plan, not a physical route.") if plan["steps"] else (
                    "CIS 579 rules found no low-score detection, signal to check, or handling-related review goal. "
                    "No Action does not establish that the scene is safe.")
            if plan["omitted_targets"]:
                note += f" {plan['omitted_targets']} additional review targets are outside the search limit."
            result = render_description(evidence, total, width, height, self.cfg, choice=choice,
                                        context_override=context, reason_note=note)
            result.update(scene_inference=inference, decision_search=plan)
        result.update(model="CIS 579 · Probability + UCS", backend="CIS 579")
        return result
